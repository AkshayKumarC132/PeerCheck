import os
import re
import string
import tempfile
import csv
import logging
import threading
import textwrap
import json
from typing import Dict, List, Optional, Tuple

import boto3
import numpy as np
from django.conf import settings as django_settings
from peercheck import settings
from pdf2docx import Converter
import whisper
import PyPDF2
import docx
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from io import BytesIO
from docx.shared import RGBColor, Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.text.paragraph import Run
import docx.oxml
import uuid
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from docx.enum.text import WD_COLOR_INDEX, WD_ALIGN_PARAGRAPH
import fitz  # PyMuPDF
from docx2pdf import convert

from pyannote.audio import Pipeline, Inference, Model
from pyannote.core import Segment

from .models import SpeakerProfile
from .speaker_utils import match_speaker_embedding

# Load Whisper model once
model = whisper.load_model(getattr(settings, 'WHISPER_MODEL', 'small.en'))

_DIARIZATION_PIPELINE: Optional[Pipeline] = None
_EMBEDDING_INFERENCE: Optional[Inference] = None
_MODEL_LOCK = threading.Lock()


def _get_hf_token() -> Optional[str]:
    """Return the Hugging Face token if available."""
    token = getattr(settings, "HF_TOKEN", None) or getattr(django_settings, "HF_TOKEN", None)
    return token


def _get_diarization_pipeline() -> Pipeline:
    """Lazily initialise the diarization pipeline."""
    global _DIARIZATION_PIPELINE
    if _DIARIZATION_PIPELINE is None:
        with _MODEL_LOCK:
            if _DIARIZATION_PIPELINE is None:
                _DIARIZATION_PIPELINE = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=_get_hf_token(),
                )
    return _DIARIZATION_PIPELINE


def _get_embedding_inference() -> Inference:
    """Lazily initialise the speaker embedding inference model."""
    global _EMBEDDING_INFERENCE
    if _EMBEDDING_INFERENCE is None:
        with _MODEL_LOCK:
            if _EMBEDDING_INFERENCE is None:
                embedding_model = Model.from_pretrained(
                    "pyannote/embedding",
                    use_auth_token=_get_hf_token(),
                )
                _EMBEDDING_INFERENCE = Inference(embedding_model, window="whole")
    return _EMBEDDING_INFERENCE

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_S3_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_S3_SECRET_ACCESS_KEY,
    region_name=settings.AWS_S3_REGION_NAME
)

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def upload_file_to_s3(file_obj, s3_key):
    print("Upload File to S3 ", file_obj, s3_key)
    """Upload file to S3 and return the S3 URL"""
    try:
        s3_client.upload_fileobj(
            file_obj,
            settings.AWS_STORAGE_BUCKET_NAME,
            s3_key,
            ExtraArgs={'ACL': 'private'}
        )
        return f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
    except Exception as e:
        raise Exception(f"Failed to upload to S3: {str(e)}")

def download_file_from_s3(s3_key):
    """Download file from S3 to temporary file and return path"""
    try:
        response = s3_client.get_object(
            Bucket=settings.AWS_STORAGE_BUCKET_NAME,
            Key=s3_key
        )
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(response['Body'].read())
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        raise Exception(f"Failed to download from S3: {str(e)}")

def get_s3_key_from_url(s3_url):
    """Extract S3 key from S3 URL"""
    return s3_url.replace(f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/", "")

def extract_text_txt(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def extract_text_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ''
    return text

def extract_text_docx(file_path):
    document = docx.Document(file_path)
    return '\n'.join([para.text for para in document.paragraphs])

def extract_text_from_s3(s3_url):
    """Extract text from document stored in S3"""
    s3_key = get_s3_key_from_url(s3_url)
    temp_path = download_file_from_s3(s3_key)
    
    try:
        ext = s3_key.rsplit('.', 1)[1].lower()
        if ext == 'pdf':
            text = extract_text_pdf(temp_path)
        elif ext == 'docx':
            text = extract_text_docx(temp_path)
        elif ext == 'txt':
            text = extract_text_txt(temp_path)
        else:
            text = ''
        return text
    finally:
        # Clean up temp file
        os.unlink(temp_path)

def transcribe_audio_from_s3(s3_url):
    """Transcribe audio file stored in S3 and return full Whisper result (with word-level timestamps)"""
    s3_key = get_s3_key_from_url(s3_url)
    temp_path = download_file_from_s3(s3_key)
    try:
        # Use word_timestamps=True to get word-level info
        result = model.transcribe(temp_path, word_timestamps=True)
        return result  # Return the full result dict
    finally:
        os.unlink(temp_path)

def normalize_line(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\[\]\(\)\{\}\<\>]", "", s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(s.split())

def find_missing(text, transcript, threshold=0.6):
    """
    Fuzzy-match each original text line against all transcript lines.
    Returns: (matched_html, missing_html, matched_words, total_words, entire_html)
    """
    text_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    trans_lines = [ln.strip() for ln in transcript.splitlines() if ln.strip()]
    norm_text = [normalize_line(ln) for ln in text_lines]
    norm_trans = [normalize_line(ln) for ln in trans_lines]

    # Detect start offset
    start_idx = 0
    if norm_trans and norm_text:
        K = min(len(norm_trans), 3)
        best_avg, best_idx = 0.0, 0
        for i in range(len(norm_text) - K + 1):
            avg = sum(
                SequenceMatcher(None, norm_text[i + j], norm_trans[j]).ratio()
                for j in range(K)
            ) / K
            if avg > best_avg:
                best_avg, best_idx = avg, i
        if best_avg >= threshold:
            start_idx = best_idx

    matched_spans, missing_spans = [], []
    matched_words = 0
    total_words = sum(len(normalize_line(ln).split()) for ln in text_lines)

    # Mark lines before start offset as missing
    for ln in text_lines[:start_idx]:
        missing_spans.append(f'<span class="missing">{ln}</span>')

    # Fuzzy-match remaining lines
    for orig, norm in zip(text_lines[start_idx:], norm_text[start_idx:]):
        best = max(
            (fuzz.token_set_ratio(norm, tnorm) for tnorm in norm_trans),
            default=0
        ) / 100.0
        wcount = len(norm.split())
        if best >= threshold:
            matched_spans.append(f'<span class="match">{orig}</span>')
            matched_words += wcount
        else:
            missing_spans.append(f'<span class="missing">{orig}</span>')

    m_html = '<p>' + '</p><p>'.join(matched_spans) + '</p>' if matched_spans else ''
    x_html = '<p>' + '</p><p>'.join(missing_spans) + '</p>' if missing_spans else ''
    
    # Build entire document view
    entire_spans = []
    for orig, norm in zip(text_lines, norm_text):
        best = max((fuzz.token_set_ratio(norm, tnorm) for tnorm in norm_trans), default=0) / 100.0
        cls = "match" if best >= threshold else "missing"
        entire_spans.append(f'<span class="{cls}">{orig}</span>')
    entire_html = '<p>' + '</p><p>'.join(entire_spans) + '</p>'
    
    return m_html, x_html, matched_words, total_words, entire_html

def highlight_docx_cross_platform(docx_path, norm_trans, output_path, threshold=0.6):
    """
    Highlights text in a DOCX file using python-docx (cross-platform).

    Args:
        docx_path (str): Path to the input DOCX file.
        norm_trans (list): A list of normalized transcript strings.
        output_path (str): Path to save the highlighted DOCX.
        threshold (float): Similarity threshold for highlighting.
    """
    document = docx.Document(docx_path)
    
    # Define colors
    GREEN = RGBColor(0, 176, 80)
    RED = RGBColor(255, 0, 0)

    # 1. Highlight regular paragraphs
    for para in document.paragraphs:
        if not para.text.strip():
            continue
        norm_para_text = normalize_line(para.text)
        best_score = max((fuzz.token_set_ratio(norm_para_text, t) for t in norm_trans), default=0) / 100.0
        color = GREEN if best_score >= threshold else RED
        for run in para.runs:
            run.font.color.rgb = color

    # 2. Highlight text within shapes and text boxes by manipulating the underlying XML
    tree = document.element.body
    text_runs_in_shapes = tree.xpath('.//w:txbxContent//w:t') # Find all text runs in text boxes

    for text_run in text_runs_in_shapes:
        parent_paragraph = text_run.getparent().getparent()
        texts_in_p = parent_paragraph.xpath('.//w:t/text()')
        full_text = "".join(texts_in_p).strip()

        if not full_text:
            continue
        
        norm_shape_text = normalize_line(full_text)
        best_score = max((fuzz.token_set_ratio(norm_shape_text, t) for t in norm_trans), default=0) / 100.0
        color = GREEN if best_score >= threshold else RED
        
        # Apply color by creating/updating XML properties
        run_properties = text_run.getparent().find(docx.oxml.ns.qn('w:rPr'))
        if run_properties is None:
            run_properties = docx.oxml.OxmlElement('w:rPr')
            text_run.getparent().insert(0, run_properties)
        
        color_element = run_properties.find(docx.oxml.ns.qn('w:color'))
        if color_element is None:
            color_element = docx.oxml.OxmlElement('w:color')
            run_properties.append(color_element)
            
        color_element.set(docx.oxml.ns.qn('w:val'), str(color))

    document.save(output_path)


def _load_abbreviation_map(csv_path: str) -> dict:
    """Load abbreviation mappings from a CSV file.

    Supports rows that contain multiple abbreviations for the same meaning
    separated by characters such as commas, slashes, dashes, or spaces. Each
    abbreviation variant is normalized by stripping non-alphanumeric
    characters and upper-casing before being added to the map. Single-character
    entries and headers are skipped.
    """
    abbr_map: dict[str, dict] = {}
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            with open(csv_path, newline="", encoding=enc) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) < 2:
                        continue
                    abbr_field, full = row[0].strip(), row[1].strip()
                    if not (abbr_field and full) or abbr_field.upper() == "ABBREVIATION":
                        continue

                    # Split the abbreviation field into individual candidates
                    for part in re.split(r",", abbr_field):
                        part = part.strip()
                        if not part:
                            continue

                        # Consider sub-variants separated by '/', '-' or spaces
                        candidates = {part}
                        for sep in ("/", "-", " "):
                            if sep in part:
                                candidates.update(
                                    seg.strip() for seg in part.split(sep) if seg.strip()
                                )

                        for abbr in candidates:
                            norm = re.sub(r"[^A-Za-z0-9]", "", abbr).upper()
                            if len(norm) < 2:
                                continue
                            entry = abbr_map.setdefault(norm, {"abbrs": set(), "full": full})
                            entry["abbrs"].add(abbr)

            for data in abbr_map.values():
                data["abbrs"] = list(data["abbrs"])
            logging.info(
                "Loaded %d abbreviations from CSV using %s", len(abbr_map), enc
            )
            return abbr_map
        except Exception as exc:
            logging.warning("Failed to load acronyms CSV with %s: %s", enc, exc)
    return abbr_map


def _confirm_abbreviations(abbr_map: dict, transcript: str) -> dict:
    """Filter abbreviation map to only those whose full form appears in transcript."""
    transcript_lower = transcript.lower()
    validated = {}
    for norm, data in abbr_map.items():
        full = data["full"].lower()
        abbr_display = "/".join(data.get("abbrs", []))
        if full in transcript_lower:
            validated[norm] = data
            logging.debug(
                "Exact transcript match for abbreviation '%s' -> '%s'",
                abbr_display,
                data["full"],
            )
            continue
        ratio = fuzz.token_set_ratio(full, transcript_lower)
        if ratio >= 65:
            validated[norm] = data
            logging.debug(
                "Fuzzy transcript match for abbreviation '%s' -> '%s' with ratio %s",
                abbr_display,
                data["full"],
                ratio,
            )
    logging.info("Transcript validation enabled: %d abbreviations confirmed", len(validated))
    return validated


def _replace_validated_abbreviations(doc, validated_abbrs: dict) -> None:
    """Replace red highlights with dark green for validated abbreviations."""
    if not validated_abbrs:
        logging.info("No validated abbreviations to apply")
        return

    dark_green = (0.6, 1, 0.6)
    light_red = (1, 0.6, 0.6)

    def _is_light_red(color, target=light_red, tol=0.1):
        return all(abs(color[i] - target[i]) <= tol for i in range(3))

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        red_annots = []
        annot = page.first_annot
        while annot:
            colors = annot.colors or {}
            stroke = colors.get("stroke")
            if stroke and _is_light_red(stroke):
                red_annots.append((annot, fitz.Rect(annot.rect)))
            annot = annot.next

        if not red_annots:
            continue

        words_on_page = page.get_text("words")
        for w in words_on_page:
            word_text = w[4].strip()
            norm_word = re.sub(r"[^A-Za-z0-9]", "", word_text).upper()
            data = validated_abbrs.get(norm_word)
            if not data:
                continue
            rect = fitz.Rect(w[:4])
            for annot, a_rect in list(red_annots):
                if a_rect.intersects(rect):
                    page.delete_annot(annot)
                    red_annots.remove((annot, a_rect))
                    new_annot = page.add_highlight_annot(rect)
                    new_annot.set_colors(stroke=dark_green)
                    new_annot.set_opacity(0.6)
                    new_annot.update()
                    logging.info(
                        "Validated and highlighted abbreviation '%s' on page %d",
                        word_text,
                        page_num + 1,
                    )
                    break


def _hex_to_rgb_tuple(hex_color: str) -> tuple:
    hex_color = (hex_color or "").lstrip('#')
    if len(hex_color) != 6:
        return 0.6, 0.6, 0.6
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _format_timestamp(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def generate_highlighted_pdf(
    doc_path,
    query_text,
    output_path,
    require_transcript_match=True,
    spoken_summary: Optional[Dict] = None,
):
    """
    Opens a document, identifies relevant pages, highlights text on those pages based on semantic and numeric matching,
    and saves the result to a new PDF file.
    Adds robust validation for PDF input.
    """
    import logging
    pdf_path = doc_path
    if doc_path.lower().endswith('.docx'):
        temp_pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        convert(doc_path, temp_pdf_path)
        pdf_path = temp_pdf_path

    # --- Robust PDF Validation ---
    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
        raise ValueError(f"Input file '{pdf_path}' does not exist or is empty.")
    with open(pdf_path, 'rb') as f:
        header = f.read(5)
        if header != b'%PDF-':
            raise ValueError(f"Input file '{pdf_path}' is not a valid PDF (missing %PDF header).")
    try:
        target_doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Failed to open PDF '{pdf_path}': {e}")
        raise ValueError(f"Failed to open PDF '{pdf_path}': {e}")
    
    # --- Helper Functions ---
    def split_into_sentences(text):
        return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if len(s.strip()) > 5]

    def extract_numeric_patterns(text):
        return set(re.findall(r'\b(?:\d+(?:\.\d+)*|\w+-\w+|\w{3,})\b', text.lower()))

    # --- 1. Page Relevance Scoring ---
    from rapidfuzz import fuzz

    query_tokens = extract_numeric_patterns(query_text)
    
    token_freq = Counter()
    target_page_tokens = []
    for page in target_doc:
        page_text = page.get_text()
        tokens = extract_numeric_patterns(page_text)
        target_page_tokens.append(tokens)
        token_freq.update(tokens)

    COMMON_TOKEN_THRESHOLD = int(0.5 * len(target_doc))
    filtered_query_tokens = {tok for tok in query_tokens if token_freq[tok] < COMMON_TOKEN_THRESHOLD}

    def token_score(token):
        return len(token) / (1 + token_freq[token])

    def fuzzy_match_score(query_set, target_set):
        score = 0
        for q in query_set:
            for t in target_set:
                if fuzz.ratio(q, t) > 90:
                    score += token_score(q)
        return score

    page_scores = []
    for i, tgt_tokens in enumerate(target_page_tokens):
        score = fuzzy_match_score(filtered_query_tokens, tgt_tokens)
        page_scores.append((i, score))

    page_scores = sorted(page_scores, key=lambda x: x[1], reverse=True)

    top_k = 5
    top_pages = set()
    for i, score in page_scores[:top_k]:
        top_pages.update([i - 1, i, i + 1])
    relevant_page_nums = sorted([i for i in top_pages if 0 <= i < len(target_doc)])
    
    # --- 2. Semantic Matching Setup ---
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_chunks = split_into_sentences(query_text)
    if not query_chunks: query_chunks = [query_text]
    query_embeddings = model.encode(query_chunks, convert_to_tensor=True)

    # --- 3. Apply Highlighting to Relevant Pages ---
    transcript_word_set = {
        token.lower()
        for token in re.findall(r"\b\w+\b", query_text.lower())
        if token.strip()
    }

    for page_num in relevant_page_nums:
        page = target_doc.load_page(page_num)
        page_text = page.get_text("text")
        page_chunks = split_into_sentences(page_text)
        if not page_chunks:
            page_chunks = [page_text]

        page_embeddings = model.encode(page_chunks, convert_to_tensor=True)

        cosine_scores = util.cos_sim(page_embeddings, query_embeddings)

        matched_page_chunks = set()
        import torch
        for i, row in enumerate(cosine_scores):
            if len(row) > 0 and torch.max(row) > 0.7:
                if i < len(page_chunks):
                    matched_page_chunks.add(page_chunks[i].lower())

        green_rects = []
        words_on_page = page.get_text("words")
        for w in words_on_page:
            word_text = w[4].strip()
            word_text_lower = word_text.lower()
            rect = fitz.Rect(w[:4])

            is_numeric_match = any(
                word_text_lower == t or word_text_lower in t or t in word_text_lower
                for t in filtered_query_tokens
            )
            is_semantic_match = any(word_text_lower in chunk for chunk in matched_page_chunks)

            # Define light green and light red colors
            light_green = (0.6, 1, 0.6)
            light_red = (1, 0.6, 0.6)
            light_orange = (1, 0.85, 0.6)

            if is_numeric_match or is_semantic_match:
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors(stroke=light_green)
                highlight.set_opacity(0.3)
                highlight.update()
                green_rects.append(rect)
            else:
                if not any(rect.intersects(g) for g in green_rects):
                    highlight = page.add_highlight_annot(rect)
                    if word_text_lower in transcript_word_set:
                        highlight.set_colors(stroke=light_red)
                    else:
                        highlight.set_colors(stroke=light_orange)
                    highlight.set_opacity(0.3)
                    highlight.update()

    # --- Post-highlight Cleanup Pass for Abbreviations ---
    logging.info("Starting abbreviation cleanup pass")
    csv_path = os.path.join(settings.BASE_DIR, "Acronyms1.csv")
    abbreviations = _load_abbreviation_map(csv_path)

    if require_transcript_match:
        validated_abbrs = _confirm_abbreviations(abbreviations, query_text)
    else:
        validated_abbrs = abbreviations
        logging.info(
            "Transcript validation disabled: using %d abbreviations directly",
            len(validated_abbrs),
        )

    _replace_validated_abbreviations(target_doc, validated_abbrs)

    summary_payload = spoken_summary or {}

    def _summary_columns():
        return [
            {"key": "speaker", "header": "Speaker", "ratio": 0.15},
            {"key": "start", "header": "Start", "ratio": 0.1},
            {"key": "end", "header": "End", "ratio": 0.1},
            {"key": "status", "header": "Status", "ratio": 0.12},
            {"key": "confidence", "header": "Confidence", "ratio": 0.1},
            {"key": "document_text", "header": "Document Text", "ratio": 0.215},
            {"key": "spoken_text", "header": "Spoken Text", "ratio": 0.215},
        ]

    def _format_confidence(value: Optional[float]) -> str:
        if value is None:
            return "-"
        try:
            return f"{round(float(value) * 100, 1)}%"
        except (TypeError, ValueError):
            return str(value)

    def _prepare_row(segment: Dict, legend_map: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        status_key = (segment.get("status") or "").lower()
        legend_entry = legend_map.get(status_key, {})
        return {
            "speaker": segment.get("speaker") or "Unknown",
            "start": _format_timestamp(segment.get("start")),
            "end": _format_timestamp(segment.get("end")),
            "status": legend_entry.get("label") or status_key.title() or "-",
            "confidence": _format_confidence(segment.get("confidence")),
            "document_text": segment.get("document_text") or "-",
            "spoken_text": segment.get("spoken_text") or "-",
        }

    def _wrap_text(text: str, width: float, fontsize: float) -> List[str]:
        if width <= 0:
            return [text]
        avg_char_width = max(fontsize * 0.55, 1e-6)
        max_chars = max(1, int(width / avg_char_width))
        wrapped: List[str] = []
        for paragraph in str(text).splitlines() or [""]:
            segment = paragraph.strip() if paragraph else ""
            if not segment:
                wrapped.append("")
                continue
            wrapped.extend(textwrap.wrap(segment, max_chars) or [segment])
        return wrapped or [""]

    def _draw_summary_header(page: "fitz.Page", page_number: int, legend_map: Dict[str, Dict[str, str]], columns: List[Dict[str, str]], margin: float = 36.0) -> Tuple[float, List[float]]:
        page_rect = page.rect
        table_width = page_rect.width - 2 * margin
        x_positions = [page_rect.x0 + margin]
        for column in columns:
            x_positions.append(x_positions[-1] + table_width * column["ratio"])

        title_rect = fitz.Rect(
            page_rect.x0 + margin,
            page_rect.y0 + margin,
            page_rect.x1 - margin,
            page_rect.y0 + margin + 22,
        )
        page.insert_textbox(
            title_rect,
            f"Spoken Content Summary - Page {page_number}",
            fontsize=14,
            fontname="helv",
            align=fitz.TEXT_ALIGN_LEFT,
        )

        legend_y = title_rect.y1 + 6
        if legend_map:
            for key, meta in legend_map.items():
                color_hex = meta.get("color") or "#DDDDDD"
                rgb = _hex_to_rgb_tuple(color_hex)
                swatch = fitz.Rect(
                    page_rect.x0 + margin,
                    legend_y,
                    page_rect.x0 + margin + 12,
                    legend_y + 12,
                )
                page.draw_rect(swatch, color=rgb, fill=rgb)
                label_rect = fitz.Rect(
                    swatch.x1 + 6,
                    legend_y,
                    page_rect.x1 - margin,
                    legend_y + 12,
                )
                label = meta.get("label") or key.title()
                page.insert_textbox(
                    label_rect,
                    label,
                    fontsize=9,
                    fontname="helv",
                    align=fitz.TEXT_ALIGN_LEFT,
                )
                legend_y = label_rect.y1 + 4
        header_top = max(legend_y, title_rect.y1 + 6)

        header_height = 18
        header_bottom = header_top + header_height
        for idx, column in enumerate(columns):
            cell_rect = fitz.Rect(x_positions[idx], header_top, x_positions[idx + 1], header_bottom)
            page.draw_rect(cell_rect, color=(0.7, 0.7, 0.7), fill=(0.9, 0.9, 0.9))
            page.insert_textbox(
                cell_rect,
                column["header"],
                fontsize=10,
                fontname="helv",
                align=fitz.TEXT_ALIGN_CENTER,
            )

        return header_bottom, x_positions

    def _draw_summary_rows(
        doc: "fitz.Document",
        base_insert_index: int,
        base_page_rect: "fitz.Rect",
        page_number: int,
        segments: List[Dict],
        legend_map: Dict[str, Dict[str, str]],
        margin: float = 36.0,
    ) -> int:
        columns = _summary_columns()
        insert_position = base_insert_index + 1
        page = doc.new_page(insert_position, width=base_page_rect.width, height=base_page_rect.height)
        pages_created = 1
        insert_position += 1

        current_y, x_positions = _draw_summary_header(page, page_number, legend_map, columns, margin=margin)
        table_bottom = page.rect.y1 - margin
        body_fontsize = 9

        for segment in segments:
            row_values = _prepare_row(segment, legend_map)
            wrapped_texts: Dict[str, str] = {}
            max_lines = 1
            for idx, column in enumerate(columns):
                col_width = max((x_positions[idx + 1] - x_positions[idx]) - 4, 4)
                wrapped_lines = _wrap_text(row_values[column["key"]], col_width, body_fontsize)
                max_lines = max(max_lines, len(wrapped_lines))
                wrapped_texts[column["key"]] = "\n".join(wrapped_lines)

            line_height = body_fontsize + 2
            row_height = max(18, max_lines * line_height + 4)

            if current_y + row_height > table_bottom:
                page = doc.new_page(insert_position, width=base_page_rect.width, height=base_page_rect.height)
                pages_created += 1
                insert_position += 1
                current_y, x_positions = _draw_summary_header(page, page_number, legend_map, columns, margin=margin)
                table_bottom = page.rect.y1 - margin

                wrapped_texts = {}
                max_lines = 1
                for idx, column in enumerate(columns):
                    col_width = max((x_positions[idx + 1] - x_positions[idx]) - 4, 4)
                    wrapped_lines = _wrap_text(row_values[column["key"]], col_width, body_fontsize)
                    max_lines = max(max_lines, len(wrapped_lines))
                    wrapped_texts[column["key"]] = "\n".join(wrapped_lines)
                row_height = max(18, max_lines * line_height + 4)

            for idx, column in enumerate(columns):
                cell_rect = fitz.Rect(x_positions[idx], current_y, x_positions[idx + 1], current_y + row_height)
                page.draw_rect(cell_rect, color=(0.8, 0.8, 0.8))
                text_rect = fitz.Rect(cell_rect.x0 + 2, cell_rect.y0 + 2, cell_rect.x1 - 2, cell_rect.y1 - 2)
                page.insert_textbox(
                    text_rect,
                    wrapped_texts[column["key"]],
                    fontsize=body_fontsize,
                    fontname="helv",
                    align=fitz.TEXT_ALIGN_LEFT,
                )

            current_y += row_height

        return pages_created

    summary_pages = summary_payload.get("pages") or []
    legend_map = summary_payload.get("legend") or {}

    if summary_pages:
        summary_pages = sorted(
            [p for p in summary_pages if p.get("page_number")],
            key=lambda item: int(item.get("page_number")),
        )
        inserted_count = 0
        for page_info in summary_pages:
            page_number = int(page_info.get("page_number"))
            base_index = max(0, min(page_number - 1 + inserted_count, len(target_doc) - 1))
            base_page = target_doc.load_page(base_index)
            segments = page_info.get("segments") or []
            if not segments:
                continue
            created = _draw_summary_rows(
                target_doc,
                base_index,
                base_page.rect,
                page_number,
                segments,
                legend_map,
            )
            inserted_count += created

    # --- 4. Save the Output ---
    try:
        target_doc.save(output_path, garbage=4, deflate=True)
    finally:
        target_doc.close()
        if doc_path.lower().endswith('.docx') and os.path.exists(pdf_path):
            os.unlink(pdf_path)

    return output_path

def create_highlighted_pdf_document(
    text_s3_url,
    transcript,
    require_transcript_match=True,
    spoken_summary: Optional[Dict] = None,
):
    """
    Orchestrates the generation of a highlighted PDF using the new logic.
    """
    output_filename = f"processed/{uuid.uuid4()}_highlighted_report.pdf"
    
    # Download the reference document
    s3_key = get_s3_key_from_url(text_s3_url)
    temp_input_path = download_file_from_s3(s3_key)
    
    temp_output_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_output_path = temp_file.name

        generate_highlighted_pdf(
            temp_input_path,
            transcript,
            temp_output_path,
            require_transcript_match=require_transcript_match,
            spoken_summary=spoken_summary,
        )

        with open(temp_output_path, 'rb') as f_out:
            output_s3_url = upload_file_to_s3(f_out, output_filename)

        return output_s3_url

    finally:
        for path in (temp_input_path, temp_output_path):
            if not path:
                continue
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except PermissionError:
                logging.warning(f"Temp file in use, skipping deletion: {path}")


def create_highlighted_docx_from_s3(
    text_s3_url,
    transcript,
    high_threshold=0.6,
    low_threshold=0.3,
    spoken_summary: Optional[Dict] = None,
):
    """
    Generates a highlighted DOCX report from a reference document (PDF or DOCX)
    and a transcript, then uploads it to S3.
    """
    # Use dummy S3 functions for local paths if not using actual S3
    # Detect if the input is an S3 URL (either s3:// or https://...amazonaws.com/)
    is_s3_url = text_s3_url.startswith("s3://") or (
        text_s3_url.startswith("https://") and ".amazonaws.com/" in text_s3_url
    )
    s3_key = get_s3_key_from_url(text_s3_url) if is_s3_url else text_s3_url
    
    # Always download if it's an S3 URL (s3:// or https://...amazonaws.com/)
    temp_input_path = download_file_from_s3(s3_key) if is_s3_url else s3_key
    
    docx_in_path = None
    output_path = None
    
    try:
        norm_trans = [normalize_line(ln) for ln in transcript.splitlines() if ln.strip()]
        ext = s3_key.rsplit('.', 1)[-1].lower()
        
        # --- CONVERT INPUT FILE TO DOCX IF NECESSARY ---
        if ext == 'docx':
            print("Input is a DOCX file. No conversion needed.")
            docx_in_path = temp_input_path
        elif ext == 'pdf':
            print("Input is a PDF file. Converting to DOCX...")
            docx_in_path = tempfile.NamedTemporaryFile(delete=False, suffix='.docx').name
            cv = Converter(temp_input_path)
            cv.convert(docx_in_path, start=0, end=None)
            cv.close()
            print(f"Conversion complete. Temporary DOCX at: {docx_in_path}")
        else:
            raise ValueError(f"Unsupported file type for reference document: '{ext}'")

        # --- HIGHLIGHT THE DOCX AND CREATE THE FINAL REPORT ---
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.docx').name
        
        highlight_docx_three_color(
            docx_path=docx_in_path,
            norm_trans=norm_trans,
            output_path=output_path,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            spoken_summary=spoken_summary,
        )
        
        # --- UPLOAD THE FINAL REPORT TO S3 ---
        output_filename = os.path.basename(s3_key).rsplit('.', 1)[0]
        output_s3_key = f"processed/{uuid.uuid4()}_{output_filename}.docx"
        
        with open(output_path, 'rb') as f:
            output_s3_url = upload_file_to_s3(f, output_s3_key) if is_s3_url else output_path
        
        return output_s3_url
        
    finally:
        # --- Clean up all temporary files ---
        if is_s3_url and temp_input_path and os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if docx_in_path and docx_in_path != temp_input_path and os.path.exists(docx_in_path):
            os.unlink(docx_in_path)
        if output_path and os.path.exists(output_path) and is_s3_url: # Keep local file if not using S3
            os.unlink(output_path)

def diarization_from_audio(audio_url, transcript_segments, transcript_words=None):
    import contextlib
    import requests
    import subprocess
    import wave

    threshold = getattr(settings, "SPEAKER_MATCH_THRESHOLD", 0.8)
    min_embed_duration = getattr(settings, "SPEAKER_EMBEDDING_MIN_DURATION", 0.8)

    # Download audio locally for processing
    local_audio_path = os.path.join(tempfile.gettempdir(), f"diar_{os.path.basename(audio_url)}")
    with requests.get(audio_url, stream=True) as r:
        r.raise_for_status()
        with open(local_audio_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Convert to mono 16k wav for consistency
    wav_path = local_audio_path
    if not local_audio_path.lower().endswith('.wav'):
        wav_path = local_audio_path.rsplit('.', 1)[0] + '.wav'
        command = [
            'ffmpeg', '-y', '-i', local_audio_path,
            '-ar', '16000', '-ac', '1', wav_path
        ]
        subprocess.run(command, check=True)
        os.unlink(local_audio_path)

    diarization = _get_diarization_pipeline()(wav_path)
    embedding_inference = _get_embedding_inference()

    audio_duration = None
    try:
        with contextlib.closing(wave.open(wav_path, "rb")) as wav_file:
            frame_rate = wav_file.getframerate() or 1
            audio_duration = wav_file.getnframes() / float(frame_rate)
    except Exception:
        logging.exception("Unable to determine audio duration for %s", wav_path)
        audio_duration = None
    
    def get_segment_text_from_words(words, seg_start, seg_end, overlap_threshold=0.1):
        """
        Extract text from words that overlap with the diarization segment.
        Uses overlap threshold to handle timing misalignments.
        """
        segment_words = []
        
        for word in words:
            word_start = word.get('start')
            word_end = word.get('end')
            
            if word_start is None or word_end is None:
                continue
            
            # Calculate overlap between word and diarization segment
            overlap_start = max(word_start, seg_start)
            overlap_end = min(word_end, seg_end)
            
            if overlap_end > overlap_start:
                # Calculate overlap percentage relative to word duration
                word_duration = word_end - word_start
                overlap_duration = overlap_end - overlap_start
                
                if word_duration > 0:
                    overlap_percentage = overlap_duration / word_duration
                    
                    # Include word if it has significant overlap
                    if overlap_percentage >= overlap_threshold:
                        segment_words.append({
                            'word': word.get('word', '').strip(),
                            'start': word_start,
                            'confidence': word.get('confidence', 1.0)
                        })
        
        # Sort by start time and join
        segment_words.sort(key=lambda x: x['start'])
        return " ".join(w['word'] for w in segment_words if w['word'])
    
    def get_segment_text_from_segments(segments, seg_start, seg_end, overlap_threshold=0.3):
        """
        Extract text from transcript segments that overlap with the diarization segment.
        """
        segment_texts = []
        
        for segment in segments:
            s_start = segment.get('start')
            s_end = segment.get('end')
            
            if s_start is None or s_end is None:
                continue
            
            # Calculate overlap
            overlap_start = max(s_start, seg_start)
            overlap_end = min(s_end, seg_end)
            
            if overlap_end > overlap_start:
                segment_duration = s_end - s_start
                overlap_duration = overlap_end - overlap_start
                
                if segment_duration > 0:
                    overlap_percentage = overlap_duration / segment_duration
                    
                    # Include segment if it has significant overlap
                    if overlap_percentage >= overlap_threshold:
                        segment_texts.append({
                            'text': segment.get('text', '').strip(),
                            'start': s_start
                        })
        
        # Sort by start time and join
        segment_texts.sort(key=lambda x: x['start'])
        return " ".join(t['text'] for t in segment_texts if t['text'])
    
    def merge_consecutive_same_speaker_segments(segments, max_gap=1.0):
        """Merge consecutive segments from the same speaker if they're close together."""
        if not segments:
            return segments

        merged = []

        def _init_segment(segment):
            seg = segment.copy()
            vectors = []
            vec = seg.get("speaker_vector")
            if vec:
                vectors.append(vec)
            seg["_vectors"] = vectors
            return seg

        current_segment = _init_segment(segments[0])

        for i in range(1, len(segments)):
            next_segment = _init_segment(segments[i])

            # Check if same speaker and segments are close
            if (
                current_segment['speaker'] == next_segment['speaker'] and
                next_segment['start'] - current_segment['end'] <= max_gap
            ):
                # Merge segments
                current_segment['end'] = next_segment['end']
                if next_segment['text'].strip():
                    if current_segment['text'].strip():
                        current_segment['text'] += " " + next_segment['text']
                    else:
                        current_segment['text'] = next_segment['text']
                current_segment["_vectors"].extend(next_segment.get("_vectors", []))
                if next_segment.get("duration"):
                    current_segment['duration'] = round(
                        current_segment.get('end', next_segment['end']) - current_segment.get('start', next_segment['start']), 2
                    )
            else:
                merged.append(current_segment)
                current_segment = next_segment

        merged.append(current_segment)

        # Compute averaged vectors and clean helper keys
        for seg in merged:
            vectors = seg.pop("_vectors", [])
            if vectors:
                try:
                    seg['speaker_vector'] = (
                        np.mean(np.array(vectors, dtype=float), axis=0).tolist()
                    )
                except Exception:
                    seg['speaker_vector'] = vectors[0]

        return merged
    
    # Process diarization results
    diarization_segments = []
    label_map: Dict[str, str] = {}
    label_index = 0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        seg_start = float(turn.start)
        seg_end = float(turn.end)

        # Extract text using the appropriate method
        if transcript_words:
            segment_text = get_segment_text_from_words(transcript_words, seg_start, seg_end)
        else:
            segment_text = get_segment_text_from_segments(transcript_segments, seg_start, seg_end)

        # Only add segments with actual content or significant duration
        if segment_text.strip() or (seg_end - seg_start) > 0.5:
            if speaker not in label_map:
                label_map[speaker] = f"SPEAKER_{label_index}"
                label_index += 1

            requested_segment = Segment(seg_start, seg_end)
            vector_list = None

            try:
                duration = seg_end - seg_start
                target_segment = requested_segment

                if duration < min_embed_duration and audio_duration:
                    center = seg_start + (duration / 2.0)
                    padded_start = max(0.0, center - min_embed_duration / 2.0)
                    padded_end = min(audio_duration, padded_start + min_embed_duration)

                    if padded_end - padded_start >= min_embed_duration * 0.5:
                        target_segment = Segment(padded_start, padded_end)

                if target_segment.duration > 0:
                    vector = embedding_inference.crop(wav_path, target_segment)
                    if vector is not None:
                        vector_list = vector.tolist()
            except Exception as exc:
                logging.warning(
                    "Skipping embedding for segment %.3f-%.3f: %s",
                    seg_start,
                    seg_end,
                    exc,
                )

            diarization_segments.append({
                "speaker": label_map[speaker],
                "speaker_label": label_map[speaker],
                "start": seg_start,
                "end": seg_end,
                "text": segment_text.strip(),
                "duration": round(seg_end - seg_start, 2),
                "speaker_vector": vector_list,
                "speaker_profile_id": None,
            })

    # Merge consecutive segments from the same speaker
    diarization_segments = merge_consecutive_same_speaker_segments(diarization_segments)

    # Filter out very short segments with no text
    diarization_segments = [
        seg for seg in diarization_segments 
        if seg['text'].strip() or seg['duration'] > 1.0
    ]
    
    # Attempt to match diarized speakers with stored profiles
    label_vectors: Dict[str, List[List[float]]] = {}
    for segment in diarization_segments:
        label = segment.get("speaker_label") or segment.get("speaker")
        vec = segment.get("speaker_vector")
        if label and vec:
            label_vectors.setdefault(label, []).append(vec)

    matched_profiles: Dict[str, SpeakerProfile] = {}
    label_means: Dict[str, List[float]] = {}
    for label, vectors in label_vectors.items():
        try:
            arr = np.array(vectors, dtype=float)
        except Exception:
            continue
        if arr.size == 0:
            continue
        if np.isnan(arr).any() or np.isinf(arr).any():
            continue
        mean_vec = np.mean(arr, axis=0)
        if mean_vec is None or np.isnan(mean_vec).any() or np.isinf(mean_vec).any():
            continue
        mean_vec_list = mean_vec.tolist()
        label_means[label] = mean_vec_list

        profile = match_speaker_embedding(mean_vec_list, threshold=threshold)
        if profile:
            matched_profiles[label] = profile

    # Persist new speaker profiles for unmatched speakers so they can be named later.
    for label, mean_vec in label_means.items():
        if label in matched_profiles:
            continue
        try:
            profile = SpeakerProfile.objects.create(
                embedding=mean_vec,
                name=label,
            )
        except Exception:
            continue
        matched_profiles[label] = profile

    for segment in diarization_segments:
        label = segment.get("speaker_label") or segment.get("speaker")
        profile = matched_profiles.get(label)
        if profile:
            segment["speaker"] = profile.name or label
            segment["speaker_name"] = profile.name
            segment["speaker_profile_id"] = profile.id
        else:
            segment.setdefault("speaker_name", None)

    # Clean up
    if os.path.exists(wav_path):
        os.unlink(wav_path)

    return diarization_segments


# Diarization helpers

def build_speaker_summary(segments: Optional[List[Dict]]) -> List[Dict]:
    summary: Dict[str, Dict] = {}
    for seg in segments or []:
        label = seg.get('speaker_label') or seg.get('speaker')
        if not label:
            continue
        entry = summary.setdefault(
            label,
            {
                'speaker_label': label,
                'speaker_name': None,
                'speaker_profile_id': None,
                'segment_count': 0,
                'total_duration': 0.0,
            },
        )
        entry['segment_count'] += 1
        entry['total_duration'] += float(seg.get('duration') or 0.0)
        if seg.get('speaker_profile_id'):
            entry['speaker_profile_id'] = seg['speaker_profile_id']
            entry['speaker_name'] = seg.get('speaker_name') or seg.get('speaker')
        elif entry['speaker_name'] is None:
            entry['speaker_name'] = seg.get('speaker_name') or seg.get('speaker')

    for entry in summary.values():
        entry['total_duration'] = round(entry['total_duration'], 2)

    return list(summary.values())


# --- CORE THREE-COLOR HIGHLIGHTING LOGIC ---

HIGHLIGHT_STATUS_LEGEND = {
    "correct": {"label": "Correct Match", "color": "#C6EFCE"},
    "mismatch": {"label": "Mismatch", "color": "#FFC7CE"},
    "unspoken": {"label": "Unspoken", "color": "#FFEB9C"},
}


def _extract_docx_pages(docx_path: str) -> List[Dict]:
    document = docx.Document(docx_path)
    pages: List[Dict] = []
    current_lines: List[str] = []
    page_number = 1

    def _flush():
        nonlocal current_lines, page_number
        if current_lines:
            pages.append({
                "page_number": page_number,
                "text": "\n".join(current_lines),
            })
            current_lines = []

    for para in document.paragraphs:
        text = para.text.strip()
        if text:
            current_lines.append(text)

        has_page_break = False
        for run in para.runs:
            br_elems = run._element.findall(qn('w:br'))
            for br in br_elems:
                br_type = br.get(qn('w:type'))
                if br_type == 'page':
                    has_page_break = True
                    break
            if has_page_break:
                break

        if has_page_break:
            _flush()
            page_number += 1

    _flush()

    if not pages:
        combined = "\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())
        if combined:
            pages.append({"page_number": 1, "text": combined})

    return pages


def _load_document_pages(text_source: str) -> List[Dict]:
    is_s3_url = text_source.startswith("s3://") or (
        text_source.startswith("https://") and ".amazonaws.com/" in text_source
    )
    s3_key = get_s3_key_from_url(text_source) if is_s3_url else text_source
    local_path = download_file_from_s3(s3_key) if is_s3_url else text_source

    pages: List[Dict] = []
    ext = s3_key.rsplit('.', 1)[-1].lower() if '.' in s3_key else ''

    try:
        if ext == 'pdf':
            doc = fitz.open(local_path)
            try:
                for idx in range(len(doc)):
                    page = doc.load_page(idx)
                    pages.append({
                        "page_number": idx + 1,
                        "text": page.get_text("text") or "",
                    })
            finally:
                doc.close()
        elif ext == 'docx':
            pages = _extract_docx_pages(local_path)
        elif ext == 'txt':
            pages = [{"page_number": 1, "text": extract_text_txt(local_path)}]
        else:
            # Fallback: treat entire extracted text as a single page when possible
            extracted = ''
            try:
                if ext in {'doc', 'docm'}:
                    extracted = extract_text_docx(local_path)
                elif ext in {'rtf', 'text', ''}:
                    extracted = extract_text_txt(local_path)
            except Exception as exc:
                logging.warning("Failed to extract text for summary from %s: %s", s3_key, exc)
            if extracted:
                pages = [{"page_number": 1, "text": extracted}]
    finally:
        if is_s3_url and local_path and os.path.exists(local_path):
            os.unlink(local_path)

    return pages


def _prepare_transcript_segments(
    transcript_segments: Optional[List[Dict]],
    diarization_segments: Optional[List[Dict]],
) -> List[Dict]:
    diarization_segments = diarization_segments or []
    prepared: List[Dict] = []

    for segment in transcript_segments or []:
        text = (segment.get('text') or '').strip()
        if not text:
            continue

        start = segment.get('start')
        end = segment.get('end') if segment.get('end') is not None else start
        start = float(start) if start is not None else None
        end = float(end) if end is not None else start

        best_match = None
        best_overlap = 0.0
        for diar_seg in diarization_segments:
            d_start = float(diar_seg.get('start') or 0.0)
            d_end = float(diar_seg.get('end') or d_start)
            overlap = min(end or d_end, d_end) - max(start or d_start, d_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = diar_seg

        speaker_label = None
        speaker_name = None
        if best_match:
            speaker_label = best_match.get('speaker_label') or best_match.get('speaker')
            speaker_name = best_match.get('speaker_name') or speaker_label

        prepared.append({
            'text': text,
            'norm_text': normalize_line(text),
            'start': round(start, 3) if isinstance(start, float) else None,
            'end': round(end, 3) if isinstance(end, float) else None,
            'speaker_label': speaker_label,
            'speaker_name': speaker_name,
            'matched': False,
        })

    return prepared


def _find_best_transcript_segment(norm_text: str, segments: List[Dict]):
    best_segment = None
    best_score = 0.0
    for segment in segments:
        comp = segment.get('norm_text')
        if not comp:
            continue
        score = fuzz.token_set_ratio(norm_text, comp) / 100.0
        if score > best_score:
            best_score = score
            best_segment = segment
    return best_segment, best_score


def generate_spoken_content_summary(
    text_source: str,
    transcript_segments: Optional[List[Dict]],
    diarization: Optional[Dict] = None,
    high_threshold: float = 0.6,
    low_threshold: float = 0.3,
) -> Dict:
    legend_copy = {key: value.copy() for key, value in HIGHLIGHT_STATUS_LEGEND.items()}
    diar_segments = []
    if diarization:
        if isinstance(diarization, dict):
            diar_segments = diarization.get('segments') or []
        elif isinstance(diarization, list):
            diar_segments = diarization

    prepared_segments = _prepare_transcript_segments(transcript_segments, diar_segments)
    pages = _load_document_pages(text_source) if text_source else []

    summary_pages: List[Dict] = []

    splitter = re.compile(r'(?<=[.!?])\s+')

    for page in pages:
        page_number = page.get('page_number')
        page_text = page.get('text') or ''
        sentences = [s.strip() for s in splitter.split(page_text) if len(s.strip()) > 2]
        if not sentences and page_text.strip():
            sentences = [page_text.strip()]

        page_entries: List[Dict] = []

        for sentence in sentences:
            norm_sentence = normalize_line(sentence)
            if not norm_sentence:
                continue

            best_segment, score = _find_best_transcript_segment(norm_sentence, prepared_segments)
            if score >= high_threshold:
                status = 'correct'
            elif score >= low_threshold:
                status = 'mismatch'
            else:
                status = 'unspoken'

            entry: Dict = {
                'document_text': sentence,
                'status': status,
                'confidence': round(score, 3),
            }

            if best_segment and status != 'unspoken':
                best_segment['matched'] = True
                entry.update({
                    'spoken_text': best_segment['text'],
                    'speaker': best_segment.get('speaker_name') or best_segment.get('speaker_label'),
                    'start': best_segment.get('start'),
                    'end': best_segment.get('end'),
                })

            page_entries.append(entry)

        if page_entries:
            summary_pages.append({'page_number': page_number, 'segments': page_entries})

    unmatched_spoken = [
        {
            'spoken_text': segment['text'],
            'speaker': segment.get('speaker_name') or segment.get('speaker_label'),
            'start': segment.get('start'),
            'end': segment.get('end'),
        }
        for segment in prepared_segments
        if not segment.get('matched')
    ]

    return {
        'legend': legend_copy,
        'pages': summary_pages,
        'unmatched_spoken_segments': unmatched_spoken,
    }


def _apply_color_to_paragraph_runs(p_element, color_hex_str):
    """Applies a color to all text runs within a paragraph's XML element."""
    for r_element in p_element.xpath('.//w:r'):
        rPr = r_element.find(qn('w:rPr'))
        if rPr is None:
            rPr = docx.oxml.OxmlElement('w:rPr')
            r_element.insert(0, rPr)
        
        color_element = rPr.find(qn('w:color'))
        if color_element is None:
            color_element = docx.oxml.OxmlElement('w:color')
            rPr.append(color_element)
        color_element.set(qn('w:val'), color_hex_str)

def _process_element_three_color(element, norm_trans, thresholds, colors):
    """Finds all paragraphs in an XML element and applies color based on the two-threshold system."""
    if element is None:
        return
        
    for p_element in element.xpath('.//w:p'):
        full_text = "".join(p_element.xpath('.//w:t/text()')).strip()
        if not full_text:
            continue
            
        norm_para_text = normalize_line(full_text)
        best_score = max((fuzz.token_set_ratio(norm_para_text, t) for t in norm_trans), default=0) / 100.0
        
        if best_score >= thresholds['high']:
            color_to_apply = colors['GREEN']
        elif best_score >= thresholds['low']:
            color_to_apply = colors['RED']
        else:
            color_to_apply = colors['ORANGE']

        _apply_color_to_paragraph_runs(p_element, color_to_apply)

def highlight_docx_three_color(
    docx_path,
    norm_trans,
    output_path,
    high_threshold=0.6,
    low_threshold=0.3,
    spoken_summary: Optional[Dict] = None,
):
    """Highlights text in a DOCX using the Green/Red/Black system."""
    document = docx.Document(docx_path)
    colors = {
        "GREEN": "C6EFCE",   # light green for correct matches
        "RED": "FFC7CE",     # light red for mismatches
        "ORANGE": "FFEB9C",  # light orange for unspoken content
    }
    thresholds = {'high': high_threshold, 'low': low_threshold}

    # Process main body, headers, and footers for complete coverage
    _process_element_three_color(document.element.body, norm_trans, thresholds, colors)
    for section in document.sections:
        for part in [section.header, section.footer, section.first_page_header,
                     section.first_page_footer, section.even_page_header, section.even_page_footer]:
            _process_element_three_color(part._element, norm_trans, thresholds, colors)

    if spoken_summary:
        _append_spoken_summary_to_docx(document, spoken_summary)

    document.save(output_path)


def _add_docx_bookmark(paragraph, bookmark_name: str) -> None:
    bookmark_id = str(uuid.uuid4().int % 1000000)
    start = OxmlElement('w:bookmarkStart')
    start.set(qn('w:id'), bookmark_id)
    start.set(qn('w:name'), bookmark_name)
    end = OxmlElement('w:bookmarkEnd')
    end.set(qn('w:id'), bookmark_id)
    paragraph._p.insert(0, start)
    paragraph._p.append(end)


def _add_docx_internal_link(paragraph, text: str, anchor: str) -> Run:
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('w:anchor'), anchor)
    new_run = OxmlElement('w:r')
    r_pr = OxmlElement('w:rPr')
    new_run.append(r_pr)
    text_element = OxmlElement('w:t')
    text_element.text = text
    new_run.append(text_element)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)
    return Run(new_run, paragraph)


def _apply_button_style(run: Run, fill_hex: str = "#4472C4") -> None:
    fill = fill_hex.lstrip('#').upper() or "4472C4"
    run.font.color.rgb = RGBColor(255, 255, 255)
    run.font.bold = True
    run.font.size = Pt(10)
    r_pr = run._r.get_or_add_rPr()
    shading = OxmlElement('w:shd')
    shading.set(qn('w:val'), 'clear')
    shading.set(qn('w:color'), 'auto')
    shading.set(qn('w:fill'), fill)
    r_pr.append(shading)


def _docx_cell_set_shading(cell, fill_hex: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd')
        tc_pr.append(shd)
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex.lstrip('#').upper())


def _set_table_style(table, style_name: str, fallback: str = 'Table Grid') -> None:
    try:
        table.style = style_name
    except (KeyError, ValueError):
        table.style = fallback


def _add_docx_summary_header_button(document: docx.Document, anchor_name: str) -> None:
    label = "Show Spoken Summary"
    for section in document.sections:
        header = section.header
        if any(anchor_name in paragraph._p.xml for paragraph in header.paragraphs):
            continue
        paragraph = header.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = _add_docx_internal_link(paragraph, label, anchor_name)
        _apply_button_style(run)


def _append_spoken_summary_to_docx(document: docx.Document, spoken_summary: Dict) -> None:
    pages = spoken_summary.get('pages') or []
    unmatched = spoken_summary.get('unmatched_spoken_segments') or []
    legend = spoken_summary.get('legend') or {}

    if not pages and not unmatched:
        return

    document.add_page_break()
    heading = document.add_heading("Spoken Content Summary", level=1)
    anchor_name = "SpokenContentSummary"
    _add_docx_bookmark(heading, anchor_name)
    _add_docx_summary_header_button(document, anchor_name)

    if legend:
        legend_table = document.add_table(rows=1, cols=2)
        _set_table_style(legend_table, 'Light List Accent 1')
        header_cells = legend_table.rows[0].cells
        header_cells[0].text = "Status"
        header_cells[1].text = "Description"
        for key, meta in legend.items():
            row_cells = legend_table.add_row().cells
            label = meta.get('label') or key.title()
            description = meta.get('description') or label
            row_cells[0].text = label
            row_cells[1].text = description
            color = meta.get('color') or '#FFFFFF'
            _docx_cell_set_shading(row_cells[0], color)
        document.add_paragraph()

    for page_data in pages:
        page_number = page_data.get('page_number')
        document.add_heading(f"Page {page_number}", level=2)
        segments = page_data.get('segments') or []
        if not segments:
            document.add_paragraph("No spoken segments were matched for this page.")
            continue

        table = document.add_table(rows=1, cols=4)
        _set_table_style(table, 'Light List Accent 2')
        headers = table.rows[0].cells
        headers[0].text = "Document Text"
        headers[1].text = "Spoken Details"
        headers[2].text = "Status"
        headers[3].text = "Confidence"

        for segment in segments:
            row_cells = table.add_row().cells
            row_cells[0].text = segment.get('document_text') or '-'

            spoken_lines = []
            speaker = segment.get('speaker') or 'Unknown speaker'
            start = _format_timestamp(segment.get('start'))
            end = _format_timestamp(segment.get('end'))
            spoken_text = segment.get('spoken_text') or '-'
            spoken_lines.append(f"{speaker} ({start} – {end})")
            spoken_lines.append(spoken_text)
            row_cells[1].text = "\n".join(spoken_lines)

            status_label = (segment.get('status') or '').title() or 'Unknown'
            row_cells[2].text = status_label
            color = legend.get(segment.get('status') or '', {}).get('color')
            if color:
                _docx_cell_set_shading(row_cells[2], color)

            confidence = round(float(segment.get('confidence') or 0.0), 2)
            row_cells[3].text = str(confidence)

        document.add_paragraph()

    if unmatched:
        document.add_heading("Unmatched Spoken Segments", level=2)
        table = document.add_table(rows=1, cols=3)
        _set_table_style(table, 'Light List Accent 3')
        headers = table.rows[0].cells
        headers[0].text = "Speaker"
        headers[1].text = "Timestamps"
        headers[2].text = "Spoken Text"

        for segment in unmatched:
            row_cells = table.add_row().cells
            row_cells[0].text = segment.get('speaker') or 'Unknown speaker'
            start = _format_timestamp(segment.get('start'))
            end = _format_timestamp(segment.get('end'))
            row_cells[1].text = f"{start} – {end}"
            row_cells[2].text = segment.get('spoken_text') or '-'
