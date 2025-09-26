import os
import re
import string
import tempfile
import csv
import logging
import threading
from typing import Dict, List, Optional

import boto3
import numpy as np
from django.conf import settings as django_settings
from peercheck import settings
from pdf2docx import Converter
import whisper
import PyPDF2
import docx
from docx.shared import RGBColor
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from io import BytesIO
from docx.oxml.ns import qn
import docx.oxml
import uuid
from collections import Counter
from sentence_transformers import SentenceTransformer, util
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

def generate_highlighted_pdf(doc_path, query_text, output_path, require_transcript_match=True):
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

        transcript_tokens = [token for token in re.findall(r"\w+", query_text.lower()) if token]

        for w in words_on_page:
            word_text = w[4].strip()
            if not word_text:
                continue
            word_text_lower = word_text.lower()
            rect = fitz.Rect(w[:4])

            is_numeric_match = any(
                word_text_lower == t or word_text_lower in t or t in word_text_lower
                for t in filtered_query_tokens
            )
            is_semantic_match = any(word_text_lower in chunk for chunk in matched_page_chunks)

            light_green = (0.6, 1, 0.6)
            light_red = (1, 0.6, 0.6)
            light_orange = (1, 0.83, 0.65)

            best_ratio = 0
            if transcript_tokens:
                best_ratio = max((fuzz.partial_ratio(word_text_lower, token) for token in transcript_tokens), default=0)

            if is_numeric_match or is_semantic_match:
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors(stroke=light_green)
                highlight.set_opacity(0.3)
                highlight.update()
                green_rects.append(rect)
            else:
                color = light_orange if best_ratio < 60 else light_red
                if not any(rect.intersects(g) for g in green_rects):
                    highlight = page.add_highlight_annot(rect)
                    highlight.set_colors(stroke=color)
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
    # --- 4. Save the Output ---
    try:
        target_doc.save(output_path, garbage=4, deflate=True)
    finally:
        target_doc.close()
        if doc_path.lower().endswith('.docx') and os.path.exists(pdf_path):
            os.unlink(pdf_path)

    return output_path

def create_highlighted_pdf_document(text_s3_url, transcript, require_transcript_match=True):
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
    speaker_segments=None,
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
            speaker_segments=speaker_segments,
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

def _normalize_color_to_hex(color_value: str) -> str:
    """Normalize a color input into a DOCX-friendly hex string."""
    if not color_value:
        return "000000"
    if isinstance(color_value, tuple):
        r, g, b = color_value
        return f"{int(r):02X}{int(g):02X}{int(b):02X}"
    if hasattr(color_value, "rgb"):
        # RGBColor from python-docx exposes .rgb which is already a hex string
        return str(color_value.rgb)
    color_str = str(color_value).replace("#", "").strip()
    if color_str.startswith("RGBColor"):
        parts = [p.strip().replace("0x", "") for p in color_str[color_str.find("(") + 1:color_str.find(")")].split(",")]
        if len(parts) == 3:
            return "".join(p.zfill(2) for p in parts)
    if len(color_str) == 6:
        return color_str.upper()
    return "000000"


def _apply_color_to_paragraph_runs(p_element, color_hex_str):
    """Applies a color to all text runs within a paragraph's XML element."""
    hex_value = _normalize_color_to_hex(color_hex_str)
    for r_element in p_element.xpath('.//w:r'):
        rPr = r_element.find(qn('w:rPr'))
        if rPr is None:
            rPr = docx.oxml.OxmlElement('w:rPr')
            r_element.insert(0, rPr)

        color_element = rPr.find(qn('w:color'))
        if color_element is None:
            color_element = docx.oxml.OxmlElement('w:color')
            rPr.append(color_element)
        color_element.set(qn('w:val'), hex_value)


def _process_element_three_color(element, norm_trans, thresholds, colors, paragraph_details=None):
    """Finds all paragraphs in an XML element and applies color based on the multi-threshold system."""
    if element is None:
        return

    for p_element in element.xpath('.//w:p'):
        full_text = "".join(p_element.xpath('.//w:t/text()')).strip()
        if not full_text:
            continue

        norm_para_text = normalize_line(full_text)
        best_score = max((fuzz.token_set_ratio(norm_para_text, t) for t in norm_trans), default=0) / 100.0

        if best_score >= thresholds['high']:
            color_key = 'MATCH'
            status = 'correct'
        elif best_score >= thresholds['low']:
            color_key = 'MISMATCH'
            status = 'mismatch'
        else:
            color_key = 'UNSPOKEN'
            status = 'unspoken'

        _apply_color_to_paragraph_runs(p_element, colors[color_key])

        if paragraph_details is not None:
            paragraph_details.append({
                'text': full_text,
                'score': best_score,
                'status': status,
            })


def highlight_docx_three_color(
    docx_path,
    norm_trans,
    output_path,
    high_threshold=0.6,
    low_threshold=0.3,
    speaker_segments=None,
):
    """Highlights text in a DOCX using the enhanced color system and appends a spoken summary."""
    document = docx.Document(docx_path)
    colors = {
        "MATCH": "4CAF50",      # Light green for correct matches
        "MISMATCH": "F28B82",   # Light red for mismatches
        "UNSPOKEN": "FDCB9E",   # Light orange for unspoken content
    }
    thresholds = {'high': high_threshold, 'low': low_threshold}

    paragraph_details = []

    # Process main body, headers, and footers for complete coverage
    _process_element_three_color(document.element.body, norm_trans, thresholds, colors, paragraph_details)
    for section in document.sections:
        for part in [section.header, section.footer, section.first_page_header,
                     section.first_page_footer, section.even_page_header, section.even_page_footer]:
            _process_element_three_color(part._element, norm_trans, thresholds, colors)

    if speaker_segments:
        _append_spoken_content_summary(document, paragraph_details, speaker_segments, thresholds)

    document.save(output_path)


CONFIRMATION_PHRASES = {
    "that's correct",
    "confirmed",
    "absolutely",
    "yes that's right",
    "indeed",
    "yes correct",
    "correct",
}


def _format_timestamp(time_value):
    if time_value is None:
        return None
    try:
        total_seconds = max(float(time_value), 0.0)
    except (TypeError, ValueError):
        return None
    minutes, seconds = divmod(int(total_seconds), 60)
    return f"{int(minutes):02d}:{int(seconds):02d}"


def _format_time_range(start, end):
    start_fmt = _format_timestamp(start)
    end_fmt = _format_timestamp(end)
    if start_fmt and end_fmt and start_fmt != end_fmt:
        return f"{start_fmt} - {end_fmt}"
    return start_fmt or end_fmt or "—"


def _segment_contains_confirmation(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in CONFIRMATION_PHRASES)


def _lookup_confirmation_label(index, segments):
    current = segments[index]
    if _segment_contains_confirmation(current.get('text', '')):
        return "Confirmed"

    current_end = current.get('end')
    current_speaker = current.get('speaker') or current.get('speaker_name')

    for offset in range(1, 4):
        if index + offset >= len(segments):
            break
        candidate = segments[index + offset]
        candidate_speaker = candidate.get('speaker') or candidate.get('speaker_name')
        if candidate_speaker == current_speaker:
            continue

        if current_end is not None and candidate.get('start') is not None:
            if float(candidate['start']) - float(current_end or 0) > 30:
                break

        if _segment_contains_confirmation(candidate.get('text', '')):
            display_name = candidate.get('speaker_name') or candidate.get('speaker') or "Speaker"
            return f"Confirmed by {display_name}"

    return ""


def _append_spoken_content_summary(document, paragraph_details, speaker_segments, thresholds):
    body_paragraphs = [entry['text'] for entry in paragraph_details if entry.get('text')]
    body_norms = [normalize_line(text) for text in body_paragraphs]

    segments = [seg for seg in speaker_segments if isinstance(seg, dict)]
    segments.sort(key=lambda seg: seg.get('start') or 0)

    summary_rows = []

    for idx, segment in enumerate(segments):
        spoken_text = (segment.get('text') or '').strip()
        if not spoken_text:
            continue

        norm_spoken = normalize_line(spoken_text)
        best_score = 0.0
        best_idx = None

        for doc_idx, doc_norm in enumerate(body_norms):
            if not doc_norm:
                continue
            score = fuzz.token_set_ratio(norm_spoken, doc_norm) / 100.0
            if score > best_score:
                best_score = score
                best_idx = doc_idx

        if best_score >= thresholds['high']:
            status = "Correct"
        elif best_score >= thresholds['low']:
            status = "Mismatch"
        else:
            status = "Mismatch"

        related_phrase = body_paragraphs[best_idx] if best_idx is not None else ""
        speaker_name = segment.get('speaker_name') or segment.get('speaker') or "Speaker"
        timestamp = _format_time_range(segment.get('start'), segment.get('end'))
        confirmation = _lookup_confirmation_label(idx, segments) if segments else ""

        summary_rows.append({
            'speaker': speaker_name,
            'timestamp': timestamp,
            'spoken_text': spoken_text,
            'status': status,
            'document_phrase': related_phrase,
            'confirmation': confirmation,
        })

    for detail in paragraph_details:
        if detail.get('status') != 'unspoken' or not detail.get('text'):
            continue
        summary_rows.append({
            'speaker': '—',
            'timestamp': '—',
            'spoken_text': '',
            'status': 'Unspoken',
            'document_phrase': detail['text'],
            'confirmation': '',
        })

    if not summary_rows:
        return

    document.add_page_break()
    document.add_heading("Spoken Content Summary", level=1)

    headers = [
        "Speaker",
        "Time",
        "Spoken Content",
        "Status",
        "Document Phrase(s)",
        "Confirmation",
    ]

    table = document.add_table(rows=1, cols=len(headers))
    header_cells = table.rows[0].cells
    for cell, title in zip(header_cells, headers):
        paragraph = cell.paragraphs[0]
        run = paragraph.add_run(title)
        run.bold = True

    for row in summary_rows:
        cells = table.add_row().cells
        cells[0].text = str(row['speaker'])
        cells[1].text = row['timestamp']
        cells[2].text = row['spoken_text']
        cells[3].text = row['status']
        cells[4].text = row['document_phrase']
        cells[5].text = row['confirmation']
