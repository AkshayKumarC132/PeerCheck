import os
import re
import string
import tempfile
import csv
import logging
import threading
import textwrap
import importlib.util
from contextlib import contextmanager, suppress
from typing import Any, Dict, List, Optional, Set, Tuple

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
from docx.shared import RGBColor
from docx.oxml.ns import qn
import docx.oxml
import uuid
from collections import Counter
from sentence_transformers import SentenceTransformer, util
# Add at top (after existing ML imports like whisper/pyannote)
from sentence_transformers import SentenceTransformer
import torch  # If not present, add to requirements

# Lazy global model (thread-safe)
_SENTENCE_MODEL = None
_MODEL_LOCK = threading.Lock()

def _get_sentence_model():
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        with _MODEL_LOCK:
            if _SENTENCE_MODEL is None:
                _SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _SENTENCE_MODEL

from docx.enum.text import WD_COLOR_INDEX
from docx.shared import RGBColor
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

        _, ext = os.path.splitext(s3_key)
        suffix = ext if ext else None

        # Create temporary file while preserving the original extension so
        # downstream consumers can perform extension-based handling (for
        # example, DOCX conversion before PDF highlighting).
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
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
        if not result or not result.get("text"):
            raise ValueError("Transcription result is empty or invalid.")
        return result  # Return the full result dict
    except Exception as e:
        logging.error(f"Failed to transcribe audio from S3: {e}")
        raise
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

@contextmanager
def _word_com_apartment() -> Any:
    """Initialise a COM apartment suitable for Microsoft Word automation."""

    spec = importlib.util.find_spec("pythoncom")
    if spec is None:
        raise ValueError(
            "Converting Office documents to PDF requires the 'pywin32' package "
            "and a Microsoft Word installation."
        )

    import pythoncom  # type: ignore

    com_initialized = False
    try:
        try:
            pythoncom.CoInitialize()
            com_initialized = True
        except Exception as exc:  # pragma: no cover - best effort initialisation
            # RPC_E_CHANGED_MODE occurs when the thread is already initialised
            # in a different COM apartment. In that case we fall back to
            # explicitly initialising an STA apartment which Word automation
            # requires.
            if getattr(exc, "hresult", None) == -2147417850:
                pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
                com_initialized = True
            else:
                raise ValueError(
                    "Failed to initialise COM before launching Word: {0}".format(exc)
                ) from exc

        yield pythoncom
    finally:
        if com_initialized:
            pythoncom.CoUninitialize()


def _convert_office_document_to_pdf(source_path: str, output_path: str) -> None:
    """Convert DOCX or DOC files to PDF using the best available backend."""

    extension = os.path.splitext(source_path)[1].lower()
    if extension == ".docx":
        try:
            with _word_com_apartment():
                convert(source_path, output_path)
        except Exception as exc:
            raise ValueError(
                f"Failed to convert DOCX file '{source_path}' to PDF using Word: {exc}"
            ) from exc
        return

    if extension == ".doc":
        if importlib.util.find_spec("win32com.client") is None:
            raise ValueError(
                "Converting legacy .doc files to PDF requires the 'pywin32' package "
                "and a Microsoft Word installation."
            )

        from win32com.client import DispatchEx  # type: ignore

        with _word_com_apartment():
            try:
                word = DispatchEx("Word.Application")
            except Exception as exc:
                raise ValueError(
                    f"Failed to launch Microsoft Word for DOC conversion: {exc}"
                ) from exc

            word.Visible = False
            try:
                try:
                    document = word.Documents.Open(os.path.abspath(source_path))
                except Exception as exc:
                    raise ValueError(
                        f"Failed to open DOC file '{source_path}' with Word: {exc}"
                    ) from exc

                try:
                    document.SaveAs(os.path.abspath(output_path), FileFormat=17)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to export DOC file '{source_path}' to PDF using Word: {exc}"
                    ) from exc
                finally:
                    document.Close(False)
            finally:
                word.Quit()

        return

    raise ValueError(
        f"Unsupported Office document extension '{extension}' for PDF conversion."
    )


def generate_highlighted_pdf(doc_path, query_text, output_path, require_transcript_match=True):
    """
    Opens a document, identifies relevant pages, highlights text on those pages based on semantic and numeric matching,
    and saves the result to a new PDF file.
    Adds robust validation for PDF input.
    """
    import logging
    pdf_path = doc_path
    source_was_office_doc = doc_path.lower().endswith(('.docx', '.doc'))
    if source_was_office_doc:
        temp_pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        try:
            _convert_office_document_to_pdf(doc_path, temp_pdf_path)
        except Exception as exc:
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
            raise ValueError(
                f"Failed to convert Office document '{doc_path}' to PDF: {exc}"
            )
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

            if is_numeric_match or is_semantic_match:
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors(stroke=light_green)
                highlight.set_opacity(0.3)
                highlight.update()
                green_rects.append(rect)
            else:
                if not any(rect.intersects(g) for g in green_rects):
                    highlight = page.add_highlight_annot(rect)
                    highlight.set_colors(stroke=light_red)
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
        if source_was_office_doc and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except PermissionError:
                logging.warning(
                    "Temp file in use after Office conversion, skipping deletion: %s",
                    pdf_path,
                )
            except OSError as exc:
                logging.warning(
                    "Failed to delete temporary Office conversion file %s: %s",
                    pdf_path,
                    exc,
                )

    return output_path

def create_highlighted_pdf_document(
    text_s3_url,
    transcript,
    require_transcript_match=True,
    three_pc_entries: Optional[List[Dict]] = None,
):
    """
    Orchestrates the generation of a highlighted PDF using the new logic and optionally
    appends Three-Part Communication (3PC) summary pages.
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

        if three_pc_entries is not None:
            append_three_pc_summary_to_pdf(temp_output_path, three_pc_entries)

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
    three_pc_entries: Optional[List[Dict]] = None,
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
            low_threshold=low_threshold
        )

        if three_pc_entries is not None:
            append_three_pc_summary_to_docx(output_path, three_pc_entries)
        
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



VERIFICATION_PHRASES: Set[str] = {
    "10-4",
    "absolutely",
    "acknowledged",
    "affirm",
    "affirmative",
    "agreed",
    "all good",
    "aye",
    "check",
    "confirmed",
    "concur",
    "copy that",
    "correct",
    "exactly",
    "got it",
    "indeed",
    "i agree",
    "i confirm",
    "makes sense",
    "precisely",
    "right on",
    "roger",
    "roger that",
    "sounds good",
    "that's accurate",
    "that's correct",
    "that's right",
    "that is correct",
    "that is right",
    "understand",
    "understood",
    "verified",
    "will do",
    "wilco",
    "yes correct",
    "yes ma'am",
    "yes sir",
    "yes that's right",
    "yep",
    "yup",
    "you got it",
}

READBACK_PHRASES: Set[str] = {
    "and you",
    "and you're",
    "confirm that",
    "confirming",
    "just to clarify",
    "double checking",
    "i understand",
    "let me make sure",
    "let me repeat",
    "moving to",
    "next step",
    "proceed to",
    "read back",
    "reading back",
    "so you",
    "so you're",
    "recapping",
    "summarizing",
    "to confirm",
    "understand you",
    "you are",
    "you completed",
    "you have",
    "you're done",
    "you're ready",
    "you've",
}


def _segment_speaker(segment: Dict[str, Any]) -> str:
    return (
        segment.get("speaker_name")
        or segment.get("speaker_label")
        or segment.get("speaker")
        or "Unknown"
    )


def _segment_text(segment: Dict[str, Any]) -> str:
    text = segment.get("text") or segment.get("content") or ""
    return " ".join(text.strip().split())


def _is_verification_phrase(text: Optional[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in VERIFICATION_PHRASES)


def _is_readback_phrase(text: Optional[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in READBACK_PHRASES)


def _merge_incomplete_segments(
    segments: List[Dict[str, Any]],
    gap_seconds: float = 3.0,
) -> List[Dict[str, Any]]:
    """Conservatively merge consecutive segments from the same speaker."""

    if not segments:
        return []

    merged: List[Dict[str, Any]] = []
    for segment in sorted(segments, key=lambda seg: seg.get("start") or 0.0):
        text = _segment_text(segment)
        if not text and segment.get("is_unspoken"):
            # Respect the user's request to omit unspoken entries entirely.
            continue

        if not merged:
            new_segment = dict(segment)
            new_segment["text"] = text
            merged.append(new_segment)
            continue

        previous = merged[-1]
        same_speaker = _segment_speaker(previous) == _segment_speaker(segment)
        prev_end = previous.get("end")
        curr_start = segment.get("start")
        try:
            prev_end_f = float(prev_end) if prev_end is not None else None
            curr_start_f = float(curr_start) if curr_start is not None else None
        except (TypeError, ValueError):
            prev_end_f = curr_start_f = None

        gap = None
        if prev_end_f is not None and curr_start_f is not None:
            gap = max(0.0, curr_start_f - prev_end_f)

        should_chain = same_speaker and (
            (gap is not None and gap <= gap_seconds)
            or _is_verification_phrase(text)
            or (not text and not segment.get("is_unspoken"))
        )

        if should_chain:
            previous_text = previous.get("text") or ""
            combined = " ".join(part for part in [previous_text, text] if part).strip()
            previous["text"] = combined
            previous["end"] = segment.get("end", previous.get("end"))
            if segment.get("duration"):
                previous["duration"] = (
                    previous.get("duration", 0.0) + float(segment.get("duration"))
                )
            continue

        new_segment = dict(segment)
        new_segment["text"] = text
        merged.append(new_segment)

    return merged


def _detect_3pc_patterns(
    segments: List[Dict[str, Any]],
    gap_tolerance: float = 5.0,
) -> List[Dict[str, Any]]:
    """Identify sender/receiver/confirmation patterns across flexible spans."""

    patterns: List[Dict[str, Any]] = []
    if not segments:
        return patterns

    pattern_counter = 1
    idx = 0
    total_segments = len(segments)

    def _gap(left: Dict[str, Any], right: Dict[str, Any]) -> float:
        try:
            return max(
                0.0,
                float(right.get("start") or 0.0) - float(left.get("end") or 0.0),
            )
        except (TypeError, ValueError):
            return gap_tolerance

    while idx < total_segments - 1:
        first = segments[idx]
        sender_speaker = _segment_speaker(first)
        if not sender_speaker:
            idx += 1
            continue

        sender_text = _segment_text(first)
        receiver_indices: List[int] = []
        sender_followups: List[int] = []
        intermediate_indices: List[int] = []
        receiver_fragments: List[str] = []
        confirmation_index: Optional[int] = None

        previous = first
        j = idx + 1
        while j < total_segments:
            current = segments[j]
            if _gap(previous, current) > gap_tolerance:
                break

            current_speaker = _segment_speaker(current)
            current_text = _segment_text(current)

            if not current_text:
                intermediate_indices.append(j)
                previous = current
                j += 1
                continue

            if current_speaker == sender_speaker:
                is_confirmation = _is_verification_phrase(current_text)
                if not is_confirmation and receiver_fragments:
                    aggregated_receiver = " ".join(receiver_fragments)
                    if aggregated_receiver:
                        similarity_score = (
                            fuzz.partial_ratio(aggregated_receiver.lower(), current_text.lower())
                            / 100.0
                        )
                        is_confirmation = similarity_score >= 0.8

                if is_confirmation:
                    confirmation_index = j
                    break

                sender_followups.append(j)
            else:
                receiver_indices.append(j)
                receiver_fragments.append(current_text)

            intermediate_indices.append(j)
            previous = current
            j += 1

        if confirmation_index is not None and receiver_indices:
            aggregated_receiver_text = " ".join(receiver_fragments)
            receiver_similarity = 0.0
            if sender_text and aggregated_receiver_text:
                receiver_similarity = (
                    fuzz.partial_ratio(sender_text.lower(), aggregated_receiver_text.lower())
                    / 100.0
                )

            roles: Dict[int, str] = {
                idx: "sender",
                confirmation_index: "confirmation",
            }
            for offset, receiver_idx in enumerate(receiver_indices):
                roles[receiver_idx] = "receiver" if offset == 0 else "receiver_followup"
            for sender_idx in sender_followups:
                roles.setdefault(sender_idx, "sender_followup")

            pattern_id = f"3pc-{pattern_counter}"
            pattern_counter += 1

            patterns.append(
                {
                    "id": pattern_id,
                    "sender_index": idx,
                    "receiver_indices": receiver_indices,
                    "sender_followup_indices": sender_followups,
                    "confirmation_index": confirmation_index,
                    "intermediate_indices": intermediate_indices,
                    "receiver_similarity": receiver_similarity,
                    "roles": roles,
                }
            )

            idx = confirmation_index + 1
        else:
            idx += 1

    return patterns


def _create_document_buckets(
    reference_text: Optional[str],
    model: SentenceTransformer,
) -> List[Dict[str, Any]]:
    """Group reference text into semantic buckets for comparison."""

    if not reference_text:
        return []

    paragraphs: List[List[str]] = []
    current: List[str] = []
    for line in reference_text.splitlines():
        stripped = line.strip()
        if stripped:
            current.append(stripped)
        elif current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)

    buckets: List[Dict[str, Any]] = []
    for idx, lines in enumerate(paragraphs):
        text_block = " ".join(lines)
        if not text_block:
            continue
        try:
            embedding = model.encode([text_block], convert_to_tensor=True)[0]
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to embed reference bucket %s: %s", idx, exc)
            embedding = None
        buckets.append(
            {
                "id": idx,
                "text": text_block,
                "lines": lines,
                "normalized": normalize_line(text_block),
                "embedding": embedding,
            }
        )
    return buckets


def _find_best_bucket_match(
    content: str,
    buckets: List[Dict[str, Any]],
    model: SentenceTransformer,
) -> Tuple[Optional[Dict[str, Any]], float, float, float]:
    """Return the strongest bucket match using semantic + fuzzy scoring."""

    if not content or not buckets:
        return None, 0.0, 0.0, 0.0

    try:
        query_embedding = model.encode([content], convert_to_tensor=True)[0]
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Failed to embed segment content: %s", exc)
        query_embedding = None

    best_bucket: Optional[Dict[str, Any]] = None
    best_score = 0.0
    best_semantic = 0.0
    best_fuzzy = 0.0
    for bucket in buckets:
        semantic_score = 0.0
        if query_embedding is not None and bucket.get("embedding") is not None:
            semantic_score = float(util.cos_sim(query_embedding, bucket["embedding"]).item())
        fuzzy_score = fuzz.token_set_ratio(content.lower(), bucket["normalized"]) / 100.0
        combined = (semantic_score * 0.7) + (fuzzy_score * 0.3)
        if combined > best_score:
            best_score = combined
            best_bucket = bucket
            best_semantic = semantic_score
            best_fuzzy = fuzzy_score

    return best_bucket, best_score, best_semantic, best_fuzzy


def _format_timestamp(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    try:
        total_seconds = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "-"

    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    remainder = secs % 1
    secs = int(secs)
    if hours:
        return f"{int(hours):02d}:{int(minutes):02d}:{secs:02d}"
    if remainder:
        return f"{int(minutes):02d}:{secs:02d}.{int(remainder * 10):01d}"
    return f"{int(minutes):02d}:{secs:02d}"


def build_three_part_communication_summary(
    reference_text: Optional[str],
    diarization_segments: Optional[List[Dict]],
    match_threshold: float = 0.65,
    max_entries: int = 1000,
) -> List[Dict[str, Any]]:
    """Build a Three-Part Communication (3PC) summary with layered comparison logic."""

    merged_segments = _merge_incomplete_segments(diarization_segments or [])
    if not merged_segments:
        return []

    model = _get_sentence_model()
    buckets = _create_document_buckets(reference_text, model)
    partial_threshold = max(match_threshold * 0.8, 0.45)

    patterns = _detect_3pc_patterns(merged_segments)
    pattern_lookup: Dict[int, Dict[str, Any]] = {}
    role_lookup: Dict[int, str] = {}
    for order, pattern in enumerate(patterns):
        bundle_id = pattern.get("id") or f"3pc-{order + 1}"
        pattern["id"] = bundle_id
        for index, role in pattern.get("roles", {}).items():
            pattern_lookup[index] = pattern
            role_lookup[index] = role

    summary_entries: List[Dict[str, Any]] = []
    index_to_entry: Dict[int, Dict[str, Any]] = {}

    for idx, segment in enumerate(merged_segments):
        speaker = _segment_speaker(segment)
        text = _segment_text(segment)
        if not text:
            continue

        start = segment.get("start")
        end = segment.get("end")

        previous_entry = summary_entries[-1] if summary_entries else None

        bucket, combined_score, semantic_score, fuzzy_score = _find_best_bucket_match(
            text, buckets, model
        )
        matched_reference = bucket["text"] if bucket else None

        status = "general conversation"
        similarity = combined_score
        context_links: Dict[str, Any] = {}

        pattern = pattern_lookup.get(idx)
        role = role_lookup.get(idx)

        if role == "confirmation" or _is_verification_phrase(text):
            status = "acknowledged"
            if role == "confirmation" and pattern:
                sender_entry = index_to_entry.get(pattern["sender_index"])
                receiver_entries = [
                    index_to_entry.get(r_idx)
                    for r_idx in pattern.get("receiver_indices", [])
                ]
                if sender_entry:
                    matched_reference = sender_entry.get("reference") or matched_reference
                    similarity = max(similarity, sender_entry.get("similarity") or 0.0)
                    context_links["sender"] = sender_entry.get("speaker")
                receiver_names = [
                    entry.get("speaker")
                    for entry in receiver_entries
                    if entry and entry.get("speaker")
                ]
                if receiver_names:
                    context_links["receivers"] = receiver_names
                for receiver_entry in receiver_entries:
                    if not receiver_entry:
                        continue
                    if not matched_reference:
                        matched_reference = receiver_entry.get("reference")
                    similarity = max(
                        similarity,
                        receiver_entry.get("similarity") or 0.0,
                    )
                context_links["bundle_id"] = pattern["id"]
                context_links["role"] = "confirmation"
            elif previous_entry and previous_entry.get("status") in {"match", "partial match"}:
                matched_reference = previous_entry.get("reference") or matched_reference
                similarity = max(similarity, previous_entry.get("similarity") or 0.0)
            else:
                context_links.setdefault("role", "confirmation")
        else:
            long_enough = len(text.split()) >= 4
            if bucket:
                if combined_score >= match_threshold:
                    status = "match"
                elif combined_score >= partial_threshold:
                    status = "partial match"
                elif long_enough:
                    status = "mismatch"
            else:
                if long_enough:
                    status = "mismatch"

            if pattern and role in {"receiver", "receiver_followup"}:
                if status == "general conversation":
                    status = "partial match" if combined_score >= 0.4 else "general conversation"
                context_links["role"] = role
                context_links["bundle_id"] = pattern["id"]
                context_links.setdefault(
                    "receiver_similarity", round(pattern.get("receiver_similarity", 0.0), 2)
                )
            elif pattern and role in {"sender", "sender_followup"}:
                context_links["role"] = role
                context_links["bundle_id"] = pattern["id"]
                if status == "general conversation" and combined_score >= partial_threshold:
                    status = "partial match"

            if _is_readback_phrase(text) and status != "acknowledged":
                status = "partial match" if bucket else "general conversation"

        entry = {
            "speaker": speaker,
            "start": start,
            "end": end,
            "content": text,
            "status": status,
            "reference": matched_reference,
            "similarity": round(float(similarity), 2),
            "semantic_score": round(float(semantic_score), 2),
            "fuzzy_score": round(float(fuzzy_score), 2),
        }
        if bucket:
            entry["bucket_id"] = bucket["id"]
        if pattern:
            entry["bundle_id"] = pattern["id"]
        if context_links:
            entry["context"] = context_links
        if pattern:
            entry.setdefault("context", {})
            entry["context"].setdefault("bundle_id", pattern["id"])
            if role:
                entry["context"].setdefault("role", role)
            if role in {"receiver", "receiver_followup"} and "receiver_similarity" not in entry["context"]:
                entry["context"]["receiver_similarity"] = round(
                    pattern.get("receiver_similarity", 0.0), 2
                )

        summary_entries.append(entry)
        index_to_entry[idx] = entry
        if len(summary_entries) >= max_entries:
            break

    summary_entries.sort(
        key=lambda entry: (
            0 if entry.get("start") is not None else 1,
            float(entry.get("start") or 0.0),
            entry.get("speaker") or "",
        )
    )
    return summary_entries[:max_entries]


def append_three_pc_summary_to_docx(docx_path: str, entries: Optional[List[Dict]]) -> None:
    document = docx.Document(docx_path)
    document.add_page_break()
    document.add_heading("Three-Part Communication (3PC) Summary", level=1)

    headers = ["Speaker", "Start", "End", "Status", "Content"]
    table = document.add_table(rows=1, cols=len(headers))
    try:
        table.style = "Light Grid Accent 1"
    except KeyError:
        table.style = "Light Grid"

    for cell, title in zip(table.rows[0].cells, headers):
        cell.text = title

    status_shading = {
        "match": "c6efce",
        "acknowledged": "d9ead3",
        "partial match": "fff2cc",
        "general conversation": "dae8fc",
        "mismatch": "f4cccc",
    }

    def _apply_shading(cell, hex_color: str) -> None:
        if not hex_color:
            return
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = docx.oxml.OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), hex_color)
        tc_pr.append(shd)

    def _hex_to_rgb(hex_color: str) -> RGBColor:
        hex_color = hex_color.lstrip('#')
        return RGBColor(
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )

    if not entries:
        document.add_paragraph("No speaker diarization data was available to summarise.")
    else:
        for entry in entries:
            row = table.add_row()
            values = [
                entry.get("speaker") or "Unknown",
                _format_timestamp(entry.get("start")),
                _format_timestamp(entry.get("end")),
                (entry.get("status") or "").capitalize() or "-",
                entry.get("content") or "",
            ]
            for cell, value in zip(row.cells, values):
                cell.text = value

            status_key = (entry.get("status") or "").lower()
            shading = status_shading.get(status_key)
            if shading:
                status_cell = row.cells[3]
                _apply_shading(status_cell, shading)
                rgb = _hex_to_rgb(shading)
                for paragraph in status_cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.color.rgb = rgb

    document.save(docx_path)


def append_three_pc_summary_to_pdf(pdf_path: str, entries: Optional[List[Dict]]) -> None:
    doc = fitz.open(pdf_path)
    fd, temp_pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    saved = False

    try:
        margin = 36
        title_font_size = 16
        body_font_size = 10
        row_padding = 4
        content_width_ratio = [0.12, 0.1, 0.1, 0.13, 0.55]
        headers = ["Speaker", "Start", "End", "Status", "Content"]

        status_shading = {
            "match": (0.78, 0.94, 0.79),
            "acknowledged": (0.85, 0.92, 0.83),
            "partial match": (1.0, 0.95, 0.8),
            "general conversation": (0.85, 0.9, 0.97),
            "mismatch": (0.96, 0.8, 0.8),
        }

        status_font = {
            "match": (0, 0.4, 0),
            "acknowledged": (0.1, 0.4, 0.1),
            "partial match": (0.6, 0.4, 0),
            "general conversation": (0.1, 0.2, 0.6),
            "mismatch": (0.6, 0, 0),
        }

        def _new_page(include_header: bool = True):
            page = doc.new_page()
            width = page.rect.width
            y_pos = margin
            if include_header:
                page.insert_text(
                    (margin, y_pos),
                    "Three-Part Communication (3PC) Summary",
                    fontsize=title_font_size,
                    fontname="helv",
                )
                y_pos += title_font_size + 8

                content_width = width - 2 * margin
                x = margin
                for header, ratio in zip(headers, content_width_ratio):
                    col_width = content_width * ratio
                    rect = fitz.Rect(x, y_pos, x + col_width, y_pos + body_font_size + 8)
                    page.draw_rect(rect, color=(0.6, 0.6, 0.6), width=0.5)
                    page.insert_textbox(
                        rect,
                        header,
                        fontsize=body_font_size,
                        fontname="helv",
                        align=fitz.TEXT_ALIGN_CENTER,
                    )
                    x += col_width
                y_pos += body_font_size + 8
            return page, y_pos

        def _wrap_cell(text: str, max_width: float) -> List[str]:
            if not text:
                return [""]
            approx_char_width = max(int(max_width / (body_font_size * 0.6)), 1)
            return textwrap.wrap(text, width=approx_char_width) or [""]

        def _draw_row(page, y_pos, values: List[str], status_key: str):
            if page is None:
                page, y_pos = _new_page()

            width = page.rect.width
            height = page.rect.height
            available_height = height - margin
            content_width = width - 2 * margin

            columns = []
            for value, ratio in zip(values, content_width_ratio):
                col_width = content_width * ratio
                lines = _wrap_cell(value, col_width - (2 * row_padding))
                columns.append((col_width, lines))

            shading = status_shading.get(status_key, (0.95, 0.95, 0.95))
            font_color = status_font.get(status_key, (0, 0, 0))
            line_height = body_font_size + 2

            remaining_lines = [list(lines) for _, lines in columns]
            column_widths = [col_width for col_width, _ in columns]

            while True:
                if all(not lines for lines in remaining_lines):
                    break

                if y_pos + (2 * row_padding) + line_height > available_height:
                    page, y_pos = _new_page()
                    continue

                usable_height = available_height - y_pos - (2 * row_padding)
                max_lines_fit = max(int(usable_height / line_height), 0)

                if max_lines_fit <= 0:
                    page, y_pos = _new_page()
                    continue

                chunk_lines: List[List[str]] = []
                max_lines = 0
                more_remaining = False
                for lines in remaining_lines:
                    chunk = lines[:max_lines_fit] if lines else []
                    chunk_lines.append(chunk)
                    max_lines = max(max_lines, len(chunk))
                    if len(lines) > len(chunk):
                        more_remaining = True

                if max_lines == 0:
                    # No content to draw on this page section; move to next page.
                    page, y_pos = _new_page()
                    continue

                row_height = max_lines * line_height + (2 * row_padding)
                x = margin
                for idx, (col_width, lines_chunk) in enumerate(zip(column_widths, chunk_lines)):
                    rect = fitz.Rect(x, y_pos, x + col_width, y_pos + row_height)
                    page.draw_rect(rect, color=(0.7, 0.7, 0.7), width=0.2, fill=shading)

                    text_y = y_pos + row_padding + body_font_size
                    for line in lines_chunk:
                        page.insert_text(
                            (x + row_padding, text_y),
                            line,
                            fontsize=body_font_size,
                            fontname="helv",
                            fill=font_color if headers[idx] == "Status" else (0, 0, 0),
                        )
                        text_y += line_height
                    x += col_width

                for idx, chunk in enumerate(chunk_lines):
                    if chunk:
                        remaining_lines[idx] = remaining_lines[idx][len(chunk) :]

                y_pos += row_height

                if not more_remaining:
                    break

            return page, y_pos

        page, cursor = _new_page()

        if not entries:
            page.insert_text(
                (margin, cursor),
                "No speaker diarization data was available to summarise.",
                fontsize=body_font_size,
                fontname="helv",
            )
        else:
            for entry in entries:
                status_key = (entry.get("status") or "").lower()
                content_text = entry.get("content") or ""
                row_values = [
                    entry.get("speaker") or "Unknown",
                    _format_timestamp(entry.get("start")),
                    _format_timestamp(entry.get("end")),
                    (entry.get("status") or "").capitalize() or "-",
                    content_text,
                ]
                page, cursor = _draw_row(page, cursor, row_values, status_key)

        doc.save(temp_pdf_path, deflate=True)
        saved = True
    finally:
        doc.close()
        try:
            if saved:
                os.replace(temp_pdf_path, pdf_path)
        finally:
            if os.path.exists(temp_pdf_path):
                try:
                    os.unlink(temp_pdf_path)
                except OSError:
                    logging.warning(
                        "Temp file in use, skipping deletion: %s", temp_pdf_path
                    )


# --- CORE THREE-COLOR HIGHLIGHTING LOGIC ---

def _apply_color_to_paragraph_runs(p_element, color_hex_str):
    """Applies a color to all text runs within a paragraph's XML element."""
    for r_element in p_element.xpath('.//w:r'):
        rPr = r_element.find(qn('w:rPr'))
        if rPr is None:
            rPr = docx.oxml.OxmlElement('w:rPr')
            r_element.insert(0, rPr)
        
        color_element = docx.oxml.OxmlElement('w:color')
        color_element.set(qn('w:val'), color_hex_str)
        rPr.append(color_element)

def _collect_non_empty_paragraphs(element):
    if element is None:
        return []

    paragraphs = []
    for p_element in element.xpath('.//w:p'):
        full_text = "".join(p_element.xpath('.//w:t/text()')).strip()
        if not full_text:
            continue
        paragraphs.append((p_element, full_text, normalize_line(full_text)))
    return paragraphs


def _paragraph_index_for_offset(paragraphs, offset):
    """Map a character offset within the concatenated paragraph text to its index."""
    running = 0
    for idx, (_element, _text, normalized) in enumerate(paragraphs):
        end = running + len(normalized)
        if offset <= end:
            return idx
        running = end + 1  # account for the joining newline
    return max(len(paragraphs) - 1, 0)


def _detect_highlight_anchor_sliding(paragraphs, norm_trans, threshold):
    if not paragraphs or not norm_trans:
        return 0

    K = min(len(norm_trans), len(paragraphs), 3)
    if K == 0:
        return 0

    best_avg, best_idx = 0.0, 0
    for i in range(len(paragraphs) - K + 1):
        avg = sum(
            SequenceMatcher(None, paragraphs[i + j][2], norm_trans[j]).ratio()
            for j in range(K)
        ) / K
        if avg > best_avg:
            best_avg = avg
            best_idx = i

    return best_idx if best_avg >= threshold else 0


def _detect_highlight_anchor(paragraphs, norm_trans, threshold):
    if not paragraphs or not norm_trans:
        return 0

    doc_blob = "\n".join(p[2] for p in paragraphs if p[2])
    transcript_blob = "\n".join(t for t in norm_trans if t)

    if doc_blob and transcript_blob:
        matcher = SequenceMatcher(None, doc_blob, transcript_blob, autojunk=False)
        best_block = None
        best_quality = 0.0
        for block in matcher.get_matching_blocks():
            if not block.size:
                continue

            transcript_ratio = block.size / max(len(transcript_blob), 1)
            doc_ratio = block.size / max(len(doc_blob), 1)
            quality = max(transcript_ratio, doc_ratio)

            if block.size < 30 and transcript_ratio < threshold and doc_ratio < threshold:
                # Skip tiny overlaps unless they represent a sufficiently strong match.
                continue

            # Prefer longer, higher-quality matches to anchor the highlights.
            if quality > best_quality or (
                quality == best_quality and best_block and block.size > best_block.size
            ):
                best_quality = quality
                best_block = block

        if best_block and (best_quality >= threshold or best_block.size >= 80):
            return _paragraph_index_for_offset(paragraphs, best_block.a)

    # Fall back to the sliding window heuristic when we cannot locate a strong block
    return _detect_highlight_anchor_sliding(paragraphs, norm_trans, threshold)


def _apply_highlighting_to_paragraphs(paragraphs, norm_trans, thresholds, colors, start_index=0):
    if not paragraphs:
        return

    for idx, (p_element, _full_text, normalized_text) in enumerate(paragraphs):
        if idx < start_index:
            continue

        best_score = max((fuzz.token_set_ratio(normalized_text, t) for t in norm_trans), default=0) / 100.0

        color_to_apply = None
        if best_score >= thresholds['high']:
            color_to_apply = colors['GREEN']
        elif best_score >= thresholds['low']:
            color_to_apply = colors['RED']

        if color_to_apply:
            _apply_color_to_paragraph_runs(p_element, str(color_to_apply))


def _process_element_three_color(element, norm_trans, thresholds, colors, start_index=0):
    paragraphs = _collect_non_empty_paragraphs(element)
    _apply_highlighting_to_paragraphs(paragraphs, norm_trans, thresholds, colors, start_index)


def highlight_docx_three_color(docx_path, norm_trans, output_path, high_threshold=0.6, low_threshold=0.3):
    """Highlights text in a DOCX using the Green/Red/Black system."""
    document = docx.Document(docx_path)
    colors = {"GREEN": RGBColor(0, 176, 80), "RED": RGBColor(255, 0, 0)}
    thresholds = {'high': high_threshold, 'low': low_threshold}

    # Process main body, headers, and footers for complete coverage
    body_paragraphs = _collect_non_empty_paragraphs(document.element.body)
    body_start_index = _detect_highlight_anchor(body_paragraphs, norm_trans, thresholds['low'])
    _apply_highlighting_to_paragraphs(body_paragraphs, norm_trans, thresholds, colors, body_start_index)
    for section in document.sections:
        for part in [section.header, section.footer, section.first_page_header,
                     section.first_page_footer, section.even_page_header, section.even_page_footer]:
            _process_element_three_color(part._element, norm_trans, thresholds, colors)

    document.save(output_path)

def get_media_duration(file_obj):
    """
    Returns duration in seconds for an audio or video file object.
    Works on Windows by closing temp file before ffmpeg/moviepy reads it.
    """
    from moviepy import VideoFileClip, AudioFileClip

    # pick a reasonable suffix (extension) so ffmpeg can infer format
    suffix = os.path.splitext(getattr(file_obj, "name", ""))[-1] or ".media"

    tmp_path = None
    try:
        # Write to a *closed* temp file path (delete=False) so ffmpeg can open it
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)  # VERY IMPORTANT on Windows
        file_obj.seek(0)
        with open(tmp_path, "wb") as out:
            # If file_obj is large, copy in chunks
            chunk = file_obj.read(1024 * 1024)
            while chunk:
                out.write(chunk)
                chunk = file_obj.read(1024 * 1024)

        # Try as video first
        with suppress(Exception):
            with VideoFileClip(tmp_path) as clip:
                if clip.duration:
                    return float(clip.duration)

        # Then as audio
        with suppress(Exception):
            with AudioFileClip(tmp_path) as clip:
                if clip.duration:
                    return float(clip.duration)

        return 0.0
    except Exception:
        return 0.0
    finally:
        if tmp_path and os.path.exists(tmp_path):
            with suppress(Exception):
                os.remove(tmp_path)
