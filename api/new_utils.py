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
import torch  # If not present, add to requirements

# Encourage deterministic behaviour for repeatable scoring where possible.
torch.manual_seed(0)
np.random.seed(0)

# Lazy global SentenceTransformer model (thread-safe)
_SENTENCE_MODEL = None
_SENTENCE_MODEL_LOCK = threading.Lock()

def _get_sentence_model():
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        with _SENTENCE_MODEL_LOCK:
            if _SENTENCE_MODEL is None:
                _SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _SENTENCE_MODEL

from docx.enum.text import WD_COLOR_INDEX
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
_PIPELINE_LOCK = threading.Lock()


def _get_hf_token() -> Optional[str]:
    """Return the Hugging Face token if available."""
    token = getattr(settings, "HF_TOKEN", None) or getattr(django_settings, "HF_TOKEN", None)
    return token


def _get_diarization_pipeline() -> Pipeline:
    """Lazily initialise the diarization pipeline."""
    global _DIARIZATION_PIPELINE
    if _DIARIZATION_PIPELINE is None:
        with _PIPELINE_LOCK:
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
        with _PIPELINE_LOCK:
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


_PROCEDURE_HEADING_RE = re.compile(
    r"^(?P<number>(?:\d+\.)*\d+[A-Za-z]?)\s*(?:\((?P<paren>[^\)]+)\))?\s*(?P<title>.*)$"
)
_ENTITY_PATTERN = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9\-/]*)\b")
_ALPHANUM_ENTITY_PATTERN = re.compile(r"\b(?:[A-Z]+\d+[A-Z0-9\-]*)\b")
_ACTION_KEYWORDS = {
    "OPEN",
    "CLOSE",
    "VERIFY",
    "RECORD",
    "SET",
    "CHECK",
    "ALIGN",
    "TURN",
    "PRESS",
    "SWITCH",
    "START",
    "STOP",
    "MONITOR",
    "REPORT",
    "MEASURE",
    "ISOLATE",
    "VENT",
    "DRAIN",
    "LOCK",
    "TAG",
}
_STEP_MARKERS = ("NOTE", "VERIFY", "RECORD", "IF", "THEN")


def _extract_required_entities(text: str) -> Set[str]:
    entities: Set[str] = set()
    if not text:
        return entities

    for match in _ENTITY_PATTERN.findall(text):
        entities.add(match)

    for match in _ALPHANUM_ENTITY_PATTERN.findall(text):
        entities.add(match)

    return entities


def _extract_required_actions(text: str) -> Set[str]:
    actions: Set[str] = set()
    if not text:
        return actions

    for keyword in _ACTION_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text, flags=re.IGNORECASE):
            actions.add(keyword)
    return actions


def _detect_step_markers(text: str) -> Set[str]:
    markers: Set[str] = set()
    if not text:
        return markers

    for marker in _STEP_MARKERS:
        if re.search(rf"\b{marker}\b", text, flags=re.IGNORECASE):
            markers.add(marker.lower())
    return markers


def _parse_procedure_heading(text: str) -> Optional[Dict[str, Optional[str]]]:
    if not text:
        return None

    match = _PROCEDURE_HEADING_RE.match(text.strip())
    if not match:
        return None

    number = match.group("number") or ""
    parenthetical = match.group("paren")
    title = (match.group("title") or "").strip()

    parts = [p for p in number.split(".") if p]
    if not parts:
        return None

    section = parts[0]
    step = parts[1] if len(parts) > 1 else None
    substep = None
    if len(parts) > 2:
        substep = ".".join(parts[2:])

    if parenthetical:
        substep = f"{substep + '-' if substep else ''}{parenthetical}"

    return {
        "section": section,
        "step": step,
        "substep": substep,
        "title": title or number,
    }


def _normalize_word_token(token: str) -> str:
    return re.sub(r"[^\w']+", "", token or "").lower()


def _tokenize_phrase(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[\w']+", text.lower()) if tok]

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
    model = _get_sentence_model()
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
        """Extract overlapping words with precise timing."""

        segment_words: List[Dict[str, Any]] = []

        for word in words:
            word_start = word.get('start')
            word_end = word.get('end')

            if word_start is None or word_end is None:
                continue

            overlap_start = max(word_start, seg_start)
            overlap_end = min(word_end, seg_end)

            if overlap_end > overlap_start:
                word_duration = word_end - word_start
                overlap_duration = overlap_end - overlap_start

                if word_duration > 0:
                    overlap_percentage = overlap_duration / word_duration

                    if overlap_percentage >= overlap_threshold:
                        segment_words.append({
                            'word': word.get('word', ''),
                            'start': word_start,
                            'end': word_end,
                            'confidence': word.get('confidence', 1.0)
                        })

        segment_words.sort(key=lambda x: x['start'])
        text_tokens = [w.get('word', '') for w in segment_words]
        segment_text = "".join(text_tokens).strip()
        if not segment_text:
            segment_text = " ".join(token.strip() for token in text_tokens if token.strip())
        return segment_text, segment_words
    
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
            if seg.get("words"):
                seg["words"] = [w.copy() for w in seg.get("words") or []]
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
                if current_segment.get("words") or next_segment.get("words"):
                    current_words = current_segment.get("words") or []
                    next_words = next_segment.get("words") or []
                    combined_words = current_words + [w.copy() for w in next_words]
                    combined_words.sort(key=lambda w: w.get("start", 0.0))
                    current_segment["words"] = combined_words
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
            segment_text, segment_word_items = get_segment_text_from_words(
                transcript_words, seg_start, seg_end
            )
        else:
            segment_text = get_segment_text_from_segments(transcript_segments, seg_start, seg_end)
            segment_word_items = []

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
                "words": segment_word_items,
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


def _format_timestamp(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    try:
        total_seconds = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "-"

    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

# Verification/Acknowledgment phrases (case-insensitive)
VERIFICATION_PHRASES = {
    "that's correct", "that's right", "correct", "affirmative", "yes that's right",
    "that is correct", "roger that", "confirmed", "acknowledged", "understand",
    "got it", "copy that", "10-4", "i confirm", "verified", "check", "okay correct",
    "yes correct", "exactly", "precisely", "absolutely", "that's accurate",
    "agreed", "concur", "affirm", "i agree", "sounds good", "will do", "understood",
    "roger", "wilco", "yes sir", "yes ma'am", "aye", "yep", "yup", "okay"
}

_VERIFICATION_TOKEN_SEQUENCES = sorted(
    [(phrase, _tokenize_phrase(phrase)) for phrase in VERIFICATION_PHRASES],
    key=lambda item: len(item[1]),
    reverse=True,
)

READBACK_PHRASES = {
    "and you're", "you are", "so you", "you're ready", "proceed to", "moving to",
    "next step", "understand you", "i understand", "copy that", "you're done",
    "and you", "so you're", "you have", "you've", "you completed", "you are ready"
}


def _is_verification_phrase(text: str, threshold: float = 0.80) -> bool:
    """Detect if text is a verification/acknowledgment phrase."""
    if not text:
        return False
    
    text_lower = text.lower().strip()
    word_count = len(text_lower.split())
    
    # Be more lenient with word count (up to 10 words)
    if word_count > 10:
        return False
    
    # Exact substring match (most reliable)
    for phrase in VERIFICATION_PHRASES:
        if phrase in text_lower:
            logging.debug(f"✓ Verification phrase detected: '{text_lower}' contains '{phrase}'")
            return True
    
    # Fuzzy match for short texts
    if word_count <= 6:
        for phrase in VERIFICATION_PHRASES:
            if fuzz.ratio(text_lower, phrase) >= threshold * 100:
                logging.debug(f"✓ Verification phrase detected (fuzzy): '{text_lower}' ≈ '{phrase}'")
                return True
    
    return False


def _is_readback_phrase(text: str) -> bool:
    """Detect if text contains readback language."""
    if not text:
        return False
    
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in READBACK_PHRASES)


def _merge_incomplete_segments(segments: List[Dict], max_gap: float = 1.5) -> List[Dict]:
    """
    CONSERVATIVE merging - only merge truly fragmented segments.
    NEVER merge verification phrases or complete statements.
    """
    if not segments:
        return segments

    merged = []
    i = 0
    
    while i < len(segments):
        current = segments[i].copy()
        
        # Look ahead for potential merge
        while i + 1 < len(segments):
            next_seg = segments[i + 1]
            
            same_speaker = current.get('speaker') == next_seg.get('speaker')
            time_gap = next_seg['start'] - current['end']
            
            current_text = (current.get('text') or '').strip()
            next_text = (next_seg.get('text') or '').strip()
            
            # CRITICAL: Never merge verification phrases
            if _is_verification_phrase(current_text) or _is_verification_phrase(next_text):
                break
            
            # Check if incomplete (no period, short, or trailing connector)
            is_incomplete = (
                not current_text.endswith(('.', '!', '?')) or
                len(current_text.split()) < 3 or
                current_text.endswith((',', 'and', 'or', 'so', 'the', 'a', 'an', 'I', 'my'))
            )
            
            # Merge only if: same speaker, short gap, incomplete
            if same_speaker and time_gap <= max_gap and is_incomplete:
                current['end'] = next_seg['end']
                if current_text and next_text:
                    current['text'] = f"{current_text} {next_text}"
                elif next_text:
                    current['text'] = next_text
                current['duration'] = round(current['end'] - current['start'], 2)
                i += 1
            else:
                break
        
        merged.append(current)
        i += 1
    
    logging.info(f"Segment merging: {len(segments)} → {len(merged)}")
    return merged


def _reset_segment_annotations(segment: Dict[str, Any]) -> None:
    for key in (
        "_3pc_role",
        "_3pc_confirms",
        "_communication_type",
        "_role",
        "_pair_id",
        "_pair_role",
    ):
        segment.pop(key, None)
    segment.pop("_segment_id", None)


def _match_verification_word_spans(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not words:
        return []

    normalized_tokens: List[str] = []
    token_index_map: List[int] = []
    for idx, word in enumerate(words):
        token = _normalize_word_token(word.get("word", ""))
        if not token:
            continue
        normalized_tokens.append(token)
        token_index_map.append(idx)

    matches: List[Dict[str, Any]] = []
    i = 0
    while i < len(normalized_tokens):
        matched = False
        for phrase, tokens in _VERIFICATION_TOKEN_SEQUENCES:
            length = len(tokens)
            if not length or i + length > len(normalized_tokens):
                continue
            window = normalized_tokens[i : i + length]
            if window == tokens:
                start_idx = token_index_map[i]
                end_idx = token_index_map[i + length - 1]
                matches.append({
                    "phrase": phrase,
                    "start": start_idx,
                    "end": end_idx,
                })
                i += length
                matched = True
                break
        if not matched:
            i += 1

    return matches


def _materialize_segment_parts(segment: Dict[str, Any], parts_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    new_segments: List[Dict[str, Any]] = []

    segment_start = segment.get("start")
    segment_end = segment.get("end")
    try:
        total_duration = (
            max(float(segment_end) - float(segment_start), 0.0)
            if segment_start is not None and segment_end is not None
            else None
        )
    except (TypeError, ValueError):
        total_duration = None

    total_weight = sum(max(len(part.get("text") or ""), 1) for part in parts_meta) or 1
    current_time = (
        float(segment_start)
        if total_duration is not None and segment_start is not None
        else None
    )

    for idx, part in enumerate(parts_meta):
        new_segment = segment.copy()
        _reset_segment_annotations(new_segment)
        text_value = part.get("text", "").strip()
        new_segment["text"] = text_value

        part_words = part.get("words")
        if part_words is not None:
            new_segment["words"] = [w.copy() for w in part_words]
        elif "words" in new_segment:
            new_segment.pop("words")

        if total_duration is not None:
            start_candidate = None
            end_candidate = None
            if part_words:
                first_word = part_words[0]
                last_word = part_words[-1]
                if first_word.get("start") is not None:
                    start_candidate = float(first_word["start"])
                if last_word.get("end") is not None:
                    end_candidate = float(last_word["end"])

            if (
                start_candidate is not None
                and end_candidate is not None
                and end_candidate >= start_candidate
            ):
                new_start = start_candidate
                new_end = end_candidate
                current_time = new_end
            else:
                portion = total_duration * (max(len(text_value), 1) / total_weight)
                if current_time is None:
                    current_time = float(segment_start) if segment_start is not None else 0.0
                new_start = current_time
                if idx == len(parts_meta) - 1 and segment_end is not None:
                    new_end = float(segment_end)
                else:
                    new_end = new_start + portion
                current_time = new_end

            new_segment["start"] = round(new_start, 2)
            new_segment["end"] = round(new_end, 2)
            new_segment["duration"] = round(max(new_end - new_start, 0.0), 2)
        else:
            if "start" in segment:
                new_segment["start"] = segment.get("start")
            if "end" in segment:
                new_segment["end"] = segment.get("end")
            if isinstance(segment.get("duration"), (int, float)):
                new_segment["duration"] = segment.get("duration")
            elif "duration" in new_segment:
                new_segment.pop("duration")

        new_segments.append(new_segment)

    return new_segments


def _split_segments_on_verification_phrases(segments: List[Dict]) -> List[Dict]:
    """Split segments so that verification phrases are isolated."""

    if not segments:
        return []

    # Build a single regex so that longer phrases are matched first and we can
    # capture every occurrence, even if multiple appear within the same
    # diarization segment.
    phrases = sorted(VERIFICATION_PHRASES, key=len, reverse=True)
    pattern = re.compile(r"(?i)\b(" + "|".join(re.escape(p) for p in phrases) + r")\b")
    trailing_punctuation = ",.;:!?)}]\"'”’"

    split_segments: List[Dict] = []

    for segment in segments:
        original_text = (segment.get("text") or "").strip()
        if not original_text:
            split_segments.append(segment)
            continue

        regex_matches = list(pattern.finditer(original_text))
        word_matches = _match_verification_word_spans(segment.get("words") or [])

        if regex_matches and word_matches and len(regex_matches) == len(word_matches):
            parts_meta: List[Dict[str, Any]] = []
            char_cursor = 0
            word_cursor = 0
            words = segment.get("words") or []

            for regex_match, word_match in zip(regex_matches, word_matches):
                start_idx, end_idx = regex_match.span()
                trailing_end = end_idx
                while (
                    trailing_end < len(original_text)
                    and original_text[trailing_end] in trailing_punctuation
                ):
                    trailing_end += 1

                leading_text = original_text[char_cursor:start_idx].strip()
                leading_words = words[word_cursor:word_match["start"]]
                if leading_text:
                    parts_meta.append({
                        "text": leading_text,
                        "words": leading_words,
                        "is_verification": False,
                    })

                verification_words = words[word_match["start"] : word_match["end"] + 1]
                verification_text = original_text[start_idx:trailing_end].strip()
                parts_meta.append({
                    "text": verification_text,
                    "words": verification_words,
                    "is_verification": True,
                })

                char_cursor = trailing_end
                word_cursor = word_match["end"] + 1

            trailing_text = original_text[char_cursor:].strip()
            trailing_words = (segment.get("words") or [])[word_cursor:]
            if trailing_text:
                parts_meta.append({
                    "text": trailing_text,
                    "words": trailing_words,
                    "is_verification": False,
                })

            if parts_meta:
                split_segments.extend(_materialize_segment_parts(segment, parts_meta))
            else:
                split_segments.append(segment)
            continue

        if not regex_matches:
            split_segments.append(segment)
            continue

        parts_meta = []
        cursor = 0
        for match in regex_matches:
            start_idx, end_idx = match.span()

            leading = original_text[cursor:start_idx].strip()
            if leading:
                parts_meta.append({
                    "text": leading,
                    "words": None,
                    "is_verification": False,
                })

            trailing_end = end_idx
            while (
                trailing_end < len(original_text)
                and original_text[trailing_end] in trailing_punctuation
            ):
                trailing_end += 1

            verification_text = original_text[start_idx:trailing_end].strip()
            parts_meta.append({
                "text": verification_text,
                "words": None,
                "is_verification": True,
            })
            cursor = trailing_end

        trailing_text = original_text[cursor:].strip()
        if trailing_text:
            parts_meta.append({
                "text": trailing_text,
                "words": None,
                "is_verification": False,
            })

        split_segments.extend(_materialize_segment_parts(segment, parts_meta))

    logging.info(
        "Segment verification split: %d → %d", len(segments), len(split_segments)
    )
    return split_segments


def _combine_non_verification_runs(segments: List[Dict]) -> List[Dict]:
    """Combine adjacent non-verification segments spoken by the same speaker."""

    if not segments:
        return []

    combined: List[Dict] = []

    def _speaker_identity(segment: Dict) -> Optional[str]:
        return (
            segment.get("speaker")
            or segment.get("speaker_name")
            or segment.get("speaker_label")
        )

    for segment in segments:
        current_segment = segment.copy()
        if segment.get("words"):
            current_segment["words"] = [w.copy() for w in segment.get("words") or []]

        text = (current_segment.get("text") or "").strip()
        if not text:
            combined.append(current_segment)
            continue

        speaker_id = _speaker_identity(current_segment)
        if not combined:
            combined.append(current_segment)
            continue

        previous = combined[-1]
        previous_text = (previous.get("text") or "").strip()
        previous_speaker = _speaker_identity(previous)

        if (
            speaker_id
            and speaker_id == previous_speaker
            and not _is_verification_phrase(previous_text)
            and not _is_verification_phrase(text)
        ):
            merged_text = " ".join(filter(None, [previous_text, text]))
            previous["text"] = merged_text.strip()

            # Update timing metadata when available.
            if current_segment.get("end") is not None:
                previous["end"] = current_segment.get("end")
            if previous.get("start") is not None and previous.get("end") is not None:
                try:
                    previous["duration"] = round(
                        float(previous["end"]) - float(previous["start"]), 2
                    )
                except (TypeError, ValueError):
                    previous.pop("duration", None)

            if previous.get("words") or current_segment.get("words"):
                prev_words = previous.get("words") or []
                new_words = current_segment.get("words") or []
                merged_words = prev_words + [w.copy() for w in new_words]
                merged_words.sort(key=lambda w: w.get("start", 0.0))
                previous["words"] = merged_words

            continue

        combined.append(current_segment)

    logging.info(
        "Combined non-verification runs: %d → %d", len(segments), len(combined)
    )
    return combined


_STEP_IDENTIFIER_RE = re.compile(
    r"^\s*(?P<identifier>(?:\d+\.)*\d+(?:\([^)]+\))*)(?:\s*[-:]\s*|\s+)?(?P<title>.*)$"
)
_STEP_KEYWORD_RE = re.compile(
    r"^\s*(?P<keyword>(?:NOTE|VERIFY|RECORD|IF|THEN|CAUTION|WARNING|OBSERVE)\b.*)",
    re.IGNORECASE,
)


def _parse_step_header(line: str) -> Optional[Dict[str, Any]]:
    """Extract structured metadata from a potential procedure heading."""

    if not line:
        return None

    match = _STEP_IDENTIFIER_RE.match(line)
    if match:
        identifier = match.group("identifier")
        title = match.group("title").strip()
        parts = identifier.split(".") if identifier else []
        section = parts[0] if parts else None
        step = ".".join(parts[:2]) if len(parts) >= 2 else identifier
        substep = identifier
        return {
            "identifier": identifier,
            "section": section,
            "step": step,
            "substep": substep,
            "title": title or line.strip(),
        }

    keyword_match = _STEP_KEYWORD_RE.match(line)
    if keyword_match:
        keyword = keyword_match.group("keyword").strip()
        keyword_upper = keyword.split()[0].upper()
        return {
            "identifier": keyword_upper,
            "section": None,
            "step": keyword_upper,
            "substep": None,
            "title": line.strip(),
        }

    return None


def _create_document_buckets(
    reference_lines: List[str],
    model: SentenceTransformer,
) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
    """Parse the reference document into structured buckets for alignment."""

    if not reference_lines:
        return [], None

    buckets: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def _finalize_bucket(bucket: Dict[str, Any]) -> None:
        text = " ".join(bucket.get("lines") or []).strip()
        if not text:
            return

        bucket["text"] = text
        bucket["normalized"] = normalize_line(text)
        bucket["entities"] = sorted(_extract_required_entities(text))
        bucket["actions"] = sorted(_extract_required_actions(text))
        display_lines = bucket.get("lines") or []
        display_text = " ".join(display_lines[:2]).strip()
        if not display_text:
            display_text = text[:160]
        bucket["display"] = display_text
        bucket.setdefault("identifier", f"bucket-{len(buckets) + 1}")
        buckets.append(bucket)

    for line_index, line in enumerate(reference_lines):
        stripped = line.strip()
        if not stripped:
            if current:
                current.setdefault("lines", []).append("")
            continue

        header = _parse_step_header(stripped)
        if header:
            if current:
                _finalize_bucket(current)
            current = {
                "identifier": header.get("identifier"),
                "section": header.get("section"),
                "step": header.get("step"),
                "substep": header.get("substep"),
                "title": header.get("title"),
                "lines": [stripped],
                "start_line": line_index,
            }
            continue

        if current is None:
            current = {
                "identifier": None,
                "section": None,
                "step": None,
                "substep": None,
                "title": reference_lines[0].strip() if reference_lines else "",
                "lines": [],
                "start_line": line_index,
            }

        current.setdefault("lines", []).append(stripped)

    if current:
        _finalize_bucket(current)

    if not buckets:
        return [], None

    try:
        bucket_texts = [bucket["text"] for bucket in buckets]
        embeddings = model.encode(bucket_texts)
        return buckets, np.array(embeddings)
    except Exception as exc:
        logging.warning("Failed to encode document buckets: %s", exc)
        return buckets, None


def _score_bucket_alignment(
    text: str,
    segment_embedding: Optional[np.ndarray],
    buckets: List[Dict[str, Any]],
    bucket_embeddings: Optional[np.ndarray],
) -> Tuple[Optional[int], float, float, float]:
    """Return the best matching bucket index and similarity scores."""

    if not text or not buckets:
        return None, 0.0, 0.0, 0.0

    normalized = normalize_line(text)
    best_idx: Optional[int] = None
    best_combined = 0.0
    best_semantic = 0.0
    best_lexical = 0.0

    semantic_scores = None
    if bucket_embeddings is not None and segment_embedding is not None:
        try:
            segment_array = np.array([segment_embedding])
            semantic_scores = util.cos_sim(segment_array, bucket_embeddings)[0]
        except Exception as exc:
            logging.debug("Failed semantic bucket comparison: %s", exc)
            semantic_scores = None

    for idx, bucket in enumerate(buckets):
        semantic_score = 0.0
        if semantic_scores is not None:
            try:
                semantic_score = float(semantic_scores[idx])
            except Exception:
                semantic_score = 0.0

        lexical_score = 0.0
        try:
            lexical_score = (
                fuzz.token_set_ratio(normalized, bucket.get("normalized", "")) / 100.0
            )
        except Exception:
            lexical_score = 0.0

        combined = (0.7 * semantic_score) + (0.3 * lexical_score)
        if combined > best_combined:
            best_idx = idx
            best_combined = combined
            best_semantic = semantic_score
            best_lexical = lexical_score

    return best_idx, max(0.0, min(1.0, best_combined)), max(0.0, min(1.0, best_semantic)), max(0.0, min(1.0, best_lexical))


def _assign_document_context(
    segments: List[Dict[str, Any]],
    buckets: List[Dict[str, Any]],
    bucket_embeddings: Optional[np.ndarray],
) -> None:
    """Attach document alignment metadata to each segment."""

    if not segments or not buckets:
        for segment in segments:
            segment["_doc_bucket_idx"] = None
            segment["_doc_similarity"] = 0.0
            segment["_doc_status"] = None
        return

    for segment in segments:
        text = (segment.get("text") or "").strip()
        embedding = segment.get("_embedding")
        idx, combined, semantic_score, lexical_score = _score_bucket_alignment(
            text,
            embedding,
            buckets,
            bucket_embeddings,
        )

        if idx is None:
            segment["_doc_bucket_idx"] = None
            segment["_doc_similarity"] = 0.0
            segment["_doc_semantic"] = 0.0
            segment["_doc_lexical"] = 0.0
            segment["_doc_reference"] = None
            segment["_doc_identifier"] = None
            segment["_doc_required_entities"] = []
            segment["_doc_required_actions"] = []
            segment["_doc_coverage"] = 0.0
            segment["_doc_status"] = "general conversation"
            continue

        bucket = buckets[idx]
        segment_entities = set(segment.get("_entities") or [])
        segment_actions = set(segment.get("_actions") or [])
        bucket_entities = set(bucket.get("entities") or [])
        bucket_actions = set(bucket.get("actions") or [])

        entity_coverage = (
            len(segment_entities & bucket_entities) / len(bucket_entities)
            if bucket_entities
            else (1.0 if segment_entities else 0.0)
        )
        action_coverage = (
            len(segment_actions & bucket_actions) / len(bucket_actions)
            if bucket_actions
            else (1.0 if segment_actions else 0.0)
        )
        coverage = 0.5 * entity_coverage + 0.5 * action_coverage

        segment["_doc_bucket_idx"] = idx
        segment["_doc_similarity"] = combined
        segment["_doc_semantic"] = semantic_score
        segment["_doc_lexical"] = lexical_score
        segment["_doc_reference"] = bucket.get("display")
        segment["_doc_identifier"] = bucket.get("identifier")
        segment["_doc_required_entities"] = sorted(bucket_entities)
        segment["_doc_required_actions"] = sorted(bucket_actions)
        segment["_doc_coverage"] = max(0.0, min(1.0, coverage))

        if combined >= 0.6 and coverage >= 0.6:
            doc_status = "match"
        elif combined >= 0.3 and coverage >= 0.3:
            doc_status = "partial match"
        else:
            doc_status = "general conversation"

        segment["_doc_status"] = doc_status


def _detect_3pc_patterns(segments: List[Dict]) -> List[Dict]:
    """
    Detect 3PC patterns: Statement → Readback → Confirmation
    IMPROVED: More aggressive verification detection
    """
    if len(segments) < 2:
        return segments
    
    for i in range(len(segments)):
        current = segments[i]
        current_text = (current.get('text') or '').strip()
        current_speaker = current.get('speaker')
        
        # PRIORITY: Look for verification phrases
        if _is_verification_phrase(current_text):
            if i > 0:
                prev = segments[i - 1]
                prev_speaker = prev.get('speaker')
                prev_text = (prev.get('text') or '').strip()
                
                # 3PC Confirmation: Different speaker + verification phrase
                if prev_speaker and prev_speaker != current_speaker:
                    current['_3pc_role'] = 'confirmation'
                    current['_3pc_confirms'] = i - 1
                    
                    # Mark previous as readback
                    prev['_3pc_role'] = 'readback'
                    
                    logging.info(f"✓ 3PC Confirmation: {current_speaker} confirms {prev_speaker}: '{current_text}'")
                    
                    # Look for original statement
                    if i > 1:
                        prev_prev = segments[i - 2]
                        if prev_prev.get('speaker') == current_speaker:
                            prev_prev['_3pc_role'] = 'statement'
                            prev['_3pc_confirms'] = i - 2
        
        # Look for readback language
        elif _is_readback_phrase(current_text) and i > 0:
            prev = segments[i - 1]
            if prev.get('speaker') != current_speaker:
                current['_3pc_role'] = 'readback'
                prev['_3pc_role'] = 'statement'
    
    return segments


def _assign_communication_groups(segments: List[Dict]) -> None:
    """Label segments as part of two-part or three-part communications."""

    group_id = 0
    processed_indices: Set[int] = set()

    for idx, segment in enumerate(segments):
        if segment.get("_3pc_role") != "confirmation" or idx in processed_indices:
            continue

        group_members: Set[int] = {idx}
        roles_present: Set[str] = {"confirmation"}

        readback_idx = segment.get("_3pc_confirms")
        statement_idx = None

        if isinstance(readback_idx, int) and 0 <= readback_idx < len(segments):
            readback_segment = segments[readback_idx]
            if readback_segment.get("_3pc_role") == "readback":
                group_members.add(readback_idx)
                roles_present.add("readback")
                statement_idx = readback_segment.get("_3pc_confirms")

        if isinstance(statement_idx, int) and 0 <= statement_idx < len(segments):
            statement_segment = segments[statement_idx]
            if statement_segment.get("_3pc_role") == "statement":
                group_members.add(statement_idx)
                roles_present.add("statement")

        # Fallback: confirmations may directly point to statements without readbacks.
        if ("readback" not in roles_present) and isinstance(readback_idx, int):
            statement_candidate = segments[readback_idx]
            if statement_candidate.get("_3pc_role") == "statement":
                group_members.add(readback_idx)
                roles_present.add("statement")

        if len(group_members) <= 1:
            continue

        group_id += 1
        communication_type = "3pc" if {"statement", "readback", "confirmation"} <= roles_present else "2pc"

        for member_idx in group_members:
            segments[member_idx]["_communication_group"] = group_id
            segments[member_idx]["_communication_type"] = communication_type
            processed_indices.add(member_idx)


_SENDER_ROLE_HINTS = (
    "i will",
    "i'll",
    "we will",
    "we'll",
    "let me",
    "i am going to",
    "we are going to",
)
_RECEIVER_ROLE_HINTS = (
    "you said",
    "you want",
    "confirming",
    "repeating",
    "let me confirm",
    "copying",
    "so you're",
)
_VERIFIER_ROLE_HINTS = (
    "confirm",
    "acknowledge",
    "roger",
    "affirm",
)


def _classify_segment_roles(segments: List[Dict[str, Any]]) -> None:
    for segment in segments:
        text = (segment.get("text") or "").strip()
        lower_text = text.lower()

        role = None
        three_pc_role = segment.get("_3pc_role")
        if three_pc_role == "statement":
            role = "sender"
        elif three_pc_role == "readback":
            role = "receiver"
        elif three_pc_role == "confirmation":
            role = "verifier"

        if not role and _is_verification_phrase(text):
            role = "verifier"

        if not role and any(cue in lower_text for cue in _VERIFIER_ROLE_HINTS):
            role = "verifier"

        if not role and _extract_required_actions(text):
            role = "sender"

        if not role and any(cue in lower_text for cue in _SENDER_ROLE_HINTS):
            role = "sender"

        if not role and any(cue in lower_text for cue in _RECEIVER_ROLE_HINTS):
            role = "receiver"

        if not role and lower_text.startswith("?"):
            role = "receiver"

        segment["_role"] = role


def _prepare_segment_features(segments: List[Dict[str, Any]], model: SentenceTransformer) -> None:
    """Attach reusable features to each segment for downstream scoring."""

    texts: List[str] = []
    text_indices: List[int] = []

    for idx, segment in enumerate(segments):
        text = (segment.get("text") or "").strip()
        segment["_entities"] = sorted(_extract_required_entities(text))
        segment["_actions"] = sorted(_extract_required_actions(text))

        if text:
            texts.append(text)
            text_indices.append(idx)
        else:
            segment["_embedding"] = None

    if not texts:
        return

    try:
        embeddings = model.encode(texts)
    except Exception as exc:
        logging.warning("Failed to encode segment texts: %s", exc)
        for idx in text_indices:
            segments[idx]["_embedding"] = None
        return

    for idx, embedding in zip(text_indices, embeddings):
        segments[idx]["_embedding"] = embedding


def _segment_speaker(segment: Dict[str, Any]) -> str:
    return (
        segment.get("speaker")
        or segment.get("speaker_name")
        or segment.get("speaker_label")
        or "Unknown"
    )


def _evaluate_pair(
    pair_id: str,
    sender: Dict[str, Any],
    receiver: Optional[Dict[str, Any]],
    verifier: Optional[Dict[str, Any]],
    match_threshold: float,
    partial_threshold: float,
) -> Dict[str, Any]:
    sender_text = (sender.get("text") or "").strip()
    receiver_text = (receiver.get("text") or "").strip() if receiver else ""

    sender_entities = set(sender.get("_entities") or [])
    receiver_entities = set(receiver.get("_entities") or []) if receiver else set()
    sender_actions = set(sender.get("_actions") or [])
    receiver_actions = set(receiver.get("_actions") or []) if receiver else set()

    doc_similarity = float(sender.get("_doc_similarity") or 0.0)
    doc_similarity = max(0.0, min(1.0, doc_similarity))
    doc_coverage = float(sender.get("_doc_coverage") or 0.0)
    doc_coverage = max(0.0, min(1.0, doc_coverage))
    doc_reference = sender.get("_doc_reference")
    doc_identifier = sender.get("_doc_identifier")

    strong_doc_alignment = doc_similarity >= match_threshold and doc_coverage >= 0.6
    doc_alignment = doc_similarity >= partial_threshold and doc_coverage >= 0.3

    if sender_entities:
        entity_overlap = len(sender_entities & receiver_entities) / len(sender_entities)
    else:
        entity_overlap = 1.0 if receiver_entities else 0.0

    if sender_actions:
        action_overlap = len(sender_actions & receiver_actions) / len(sender_actions)
    else:
        action_overlap = 1.0 if receiver_actions else 0.0

    sender_embedding = sender.get("_embedding")
    receiver_embedding = receiver.get("_embedding") if receiver else None
    semantic_similarity = 0.0
    if sender_embedding is not None and receiver_embedding is not None:
        try:
            semantic_similarity = float(
                util.cos_sim(
                    np.array([sender_embedding]),
                    np.array([receiver_embedding]),
                )[0][0]
            )
        except Exception as exc:
            logging.debug("Failed semantic similarity computation: %s", exc)
            semantic_similarity = 0.0

    semantic_similarity = max(0.0, min(1.0, semantic_similarity))

    lexical_similarity = 0.0
    sequence_similarity = 0.0
    if receiver_text:
        try:
            lexical_similarity = (
                fuzz.token_set_ratio(
                    normalize_line(sender_text),
                    normalize_line(receiver_text),
                )
                / 100.0
            )
        except Exception:
            lexical_similarity = 0.0
        sequence_similarity = SequenceMatcher(
            None,
            sender_text.lower(),
            receiver_text.lower(),
        ).ratio()

    lexical_similarity = max(0.0, min(1.0, lexical_similarity))
    sequence_similarity = max(0.0, min(1.0, sequence_similarity))
    combined_lexical = 0.5 * lexical_similarity + 0.5 * sequence_similarity

    similarity = (0.7 * semantic_similarity) + (0.3 * combined_lexical)
    coverage = (0.5 * entity_overlap) + (0.5 * action_overlap)

    timing_components: List[float] = []
    if sender.get("end") is not None and receiver and receiver.get("start") is not None:
        try:
            gap = max(0.0, float(receiver["start"]) - float(sender["end"]))
            timing_components.append(max(0.0, 1.0 - min(gap / 10.0, 1.0)))
        except (TypeError, ValueError):
            pass
    if receiver and receiver.get("end") is not None and verifier and verifier.get("start") is not None:
        try:
            gap = max(0.0, float(verifier["start"]) - float(receiver["end"]))
            timing_components.append(max(0.0, 1.0 - min(gap / 10.0, 1.0)))
        except (TypeError, ValueError):
            pass
    timing_score = (
        sum(timing_components) / len(timing_components)
        if timing_components
        else (0.7 if verifier else 0.5)
    )

    ack_bonus = 0.05 if verifier else 0.0
    confidence = min(
        1.0,
        max(
            0.0,
            (0.45 * doc_similarity)
            + (0.35 * similarity)
            + (0.1 * doc_coverage)
            + (0.1 * timing_score)
            + ack_bonus,
        ),
    )

    reasons: List[str] = []
    status = "general conversation"

    doc_status = "general conversation"
    if strong_doc_alignment:
        doc_status = "match"
    elif doc_alignment:
        doc_status = "partial match"

    if receiver:
        if not doc_alignment:
            status = "mismatch"
            reasons.append("Instruction not aligned with reference document")
        else:
            if similarity >= match_threshold and strong_doc_alignment:
                status = "match"
            elif similarity >= partial_threshold:
                status = "partial match"
            else:
                status = "mismatch"

        if similarity < match_threshold:
            reasons.append("Low similarity between sender and receiver")

        if sender_entities and entity_overlap < 0.5:
            reasons.append("Missing entities in acknowledgement")

        if sender_actions and action_overlap < 0.5:
            reasons.append("Missing actions in acknowledgement")

        if timing_score < 0.4:
            reasons.append("Long delay between sender and receiver")

        if not verifier:
            reasons.append("No confirmation")
    elif verifier:
        status = "acknowledged" if strong_doc_alignment else "general conversation"
        if not strong_doc_alignment:
            reasons.append("Verification without clear document alignment")
    else:
        status = doc_status
        if not doc_alignment:
            reasons.append("Instruction not aligned with reference document")
        reasons.append("No receiver acknowledgement")

    if doc_status == "partial match" and status == "match":
        status = "partial match"
        reasons.append("Instruction partially aligned with reference document")

    if doc_status == "general conversation" and status == "match":
        status = "partial match"
        reasons.append("Limited evidence from reference document")

    if doc_alignment and doc_coverage < 0.5:
        reasons.append("Partial coverage of required entities/actions")

    a_score = round(doc_similarity, 3)
    b_score = round(similarity, 3)
    confidence = round(confidence, 3)
    timing_score = round(timing_score, 3)

    sender["_pair_id"] = pair_id
    sender["_pair_role"] = "sender"
    sender["_pair_status"] = status
    sender["_pair_confidence"] = confidence
    sender["_a_score"] = a_score
    sender["_b_score"] = b_score
    sender["_pair_doc_status"] = doc_status
    sender["_pair_doc_reference"] = doc_reference
    sender["_pair_doc_identifier"] = doc_identifier

    if receiver:
        receiver["_pair_id"] = pair_id
        receiver["_pair_role"] = "receiver"
        receiver["_pair_status"] = status
        receiver["_pair_confidence"] = confidence
        receiver["_a_score"] = a_score
        receiver["_b_score"] = b_score
        receiver["_pair_doc_status"] = doc_status
        receiver["_pair_doc_reference"] = doc_reference
        receiver["_pair_doc_identifier"] = doc_identifier

    if verifier:
        verifier["_pair_id"] = pair_id
        verifier["_pair_role"] = "verifier"
        verifier["_pair_status"] = status
        verifier["_pair_confidence"] = confidence
        verifier["_a_score"] = a_score
        verifier["_b_score"] = b_score
        verifier["_pair_doc_status"] = doc_status
        verifier["_pair_doc_reference"] = doc_reference
        verifier["_pair_doc_identifier"] = doc_identifier

    pair_display = None
    if receiver:
        pair_display = f"{_segment_speaker(sender)} → {_segment_speaker(receiver)}"
        if verifier:
            pair_display += f" → {_segment_speaker(verifier)}"
    elif verifier:
        pair_display = f"{_segment_speaker(sender)} → {_segment_speaker(verifier)}"

    return {
        "id": pair_id,
        "sender_segment_id": sender.get("_segment_id"),
        "receiver_segment_id": receiver.get("_segment_id") if receiver else None,
        "verifier_segment_id": verifier.get("_segment_id") if verifier else None,
        "a_score": a_score,
        "b_score": b_score,
        "confidence": confidence,
        "status": status,
        "reasons": [reason for reason in reasons if reason],
        "similarity": b_score,
        "coverage": round(coverage, 3),
        "timing_score": timing_score,
        "pair_display": pair_display,
        "doc_similarity": a_score,
        "doc_coverage": round(doc_coverage, 3),
        "doc_status": doc_status,
        "doc_reference": doc_reference,
        "doc_identifier": doc_identifier,
    }


def _build_pairs(
    segments: List[Dict[str, Any]],
    match_threshold: float,
    partial_threshold: float,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    used_segments: Set[str] = set()

    group_members: Dict[int, List[Dict[str, Any]]] = {}
    for segment in segments:
        group_id = segment.get("_communication_group")
        if group_id:
            group_members.setdefault(group_id, []).append(segment)

    for group_id, segs in group_members.items():
        segs.sort(key=lambda s: float(s.get("start") or 0.0))
        sender = next((s for s in segs if s.get("_role") == "sender"), None)
        receiver = next((s for s in segs if s.get("_role") == "receiver"), None)
        verifier = next((s for s in segs if s.get("_role") == "verifier"), None)

        if not sender and receiver:
            sender = receiver
            receiver = None

        if not sender:
            continue

        pair_id = str(uuid.uuid4())
        pair = _evaluate_pair(pair_id, sender, receiver, verifier, match_threshold, partial_threshold)
        pairs.append(pair)

        for participant in (sender, receiver, verifier):
            if participant and participant.get("_segment_id"):
                used_segments.add(participant["_segment_id"])

    segments_by_start = sorted(
        [seg for seg in segments if seg.get("_segment_id") not in used_segments],
        key=lambda s: float(s.get("start") or 0.0),
    )

    pending_receivers: List[Dict[str, Any]] = []

    for segment in segments_by_start:
        role = segment.get("_role")
        segment_id = segment.get("_segment_id")
        if not segment_id:
            continue

        if role == "sender":
            receiver_candidate = None
            verifier_candidate = None

            for candidate in pending_receivers:
                if candidate.get("_segment_id") in used_segments:
                    continue
                receiver_candidate = candidate
                break

            if not receiver_candidate:
                for candidate in segments_by_start:
                    if candidate.get("_segment_id") in used_segments:
                        continue
                    if candidate.get("_role") != "receiver":
                        continue
                    if candidate.get("start") is not None and segment.get("start") is not None:
                        if float(candidate.get("start")) < float(segment.get("start")):
                            continue
                    receiver_candidate = candidate
                    break

            if receiver_candidate:
                used_segments.add(receiver_candidate.get("_segment_id"))

            if receiver_candidate:
                for candidate in segments_by_start:
                    if candidate.get("_segment_id") in used_segments:
                        continue
                    if candidate.get("_role") != "verifier":
                        continue
                    if (
                        receiver_candidate.get("end") is not None
                        and candidate.get("start") is not None
                        and float(candidate.get("start")) < float(receiver_candidate.get("end"))
                    ):
                        continue
                    verifier_candidate = candidate
                    break

            pair_id = str(uuid.uuid4())
            pair = _evaluate_pair(
                pair_id,
                segment,
                receiver_candidate,
                verifier_candidate,
                match_threshold,
                partial_threshold,
            )
            pairs.append(pair)

            used_segments.add(segment_id)
            if verifier_candidate and verifier_candidate.get("_segment_id"):
                used_segments.add(verifier_candidate.get("_segment_id"))

        elif role == "receiver":
            pending_receivers.append(segment)

        elif role == "verifier":
            if segment.get("_segment_id") in used_segments:
                continue
            pair_id = str(uuid.uuid4())
            pair = _evaluate_pair(
                pair_id,
                segment,
                None,
                None,
                match_threshold,
                partial_threshold,
            )
            pairs.append(pair)
            used_segments.add(segment_id)

    return pairs


def _build_summary_entries(
    segments: List[Dict[str, Any]],
    pairs: List[Dict[str, Any]],
    document_buckets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pair_lookup = {pair["id"]: pair for pair in pairs}
    bucket_lookup = {idx: bucket for idx, bucket in enumerate(document_buckets or [])}
    entries: List[Dict[str, Any]] = []

    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue

        pair_id = segment.get("_pair_id")
        pair_info = pair_lookup.get(pair_id)

        status = segment.get("_pair_status") or (pair_info.get("status") if pair_info else None)
        if not status:
            role = segment.get("_role")
            if role == "verifier":
                status = "acknowledged"
            elif role == "receiver":
                has_content = bool(segment.get("_entities") or segment.get("_actions"))
                status = "partial match" if has_content else "general conversation"
            else:
                status = "general conversation"

        confidence = pair_info.get("confidence") if pair_info else None
        a_score = pair_info.get("a_score") if pair_info else None
        b_score = pair_info.get("b_score") if pair_info else None
        reasons = pair_info.get("reasons") if pair_info else []
        pair_display = pair_info.get("pair_display") if pair_info else None

        doc_similarity = segment.get("_doc_similarity")
        if doc_similarity is None and pair_info:
            doc_similarity = pair_info.get("doc_similarity")
        doc_coverage = segment.get("_doc_coverage")
        doc_status = (
            segment.get("_pair_doc_status")
            or (pair_info.get("doc_status") if pair_info else None)
            or segment.get("_doc_status")
        )
        doc_reference = segment.get("_pair_doc_reference") or segment.get("_doc_reference")
        doc_identifier = segment.get("_pair_doc_identifier") or segment.get("_doc_identifier")

        bucket_index = segment.get("_doc_bucket_idx")
        if doc_reference is None and bucket_index is not None:
            bucket = bucket_lookup.get(bucket_index)
            if bucket:
                doc_reference = bucket.get("display")
                doc_identifier = doc_identifier or bucket.get("identifier")

        role = segment.get("_role")
        if role == "verifier":
            entry_type = "Verification"
        elif role == "receiver":
            entry_type = "Acknowledgement"
        else:
            entry_type = "Instruction"

        entry = {
            "id": segment.get("_segment_id"),
            "speaker": _segment_speaker(segment),
            "start": segment.get("start"),
            "end": segment.get("end"),
            "content": text,
            "status": status,
            "role": role,
            "type": entry_type,
            "pair": pair_id,
            "pair_display": pair_display,
            "pair_role": segment.get("_pair_role"),
            "confidence": confidence,
            "a_score": a_score,
            "b_score": b_score,
            "reasons": reasons,
            "three_pc_role": segment.get("_3pc_role"),
            "pair_status": pair_info.get("status") if pair_info else None,
            "communication_type": segment.get("_communication_type"),
            "reference": doc_reference,
            "doc_similarity": round(doc_similarity, 3) if isinstance(doc_similarity, (int, float)) else None,
            "doc_status": doc_status,
            "doc_identifier": doc_identifier,
            "doc_coverage": round(doc_coverage, 3) if isinstance(doc_coverage, (int, float)) else None,
        }

        entries.append(entry)

    return entries


def build_three_part_communication_summary(
    reference_text: Optional[str],
    diarization_segments: Optional[List[Dict]],
    match_threshold: float = 0.6,
    partial_threshold: float = 0.3,
    max_entries: int = 1000,
) -> List[Dict[str, Any]]:
    """
    CORRECTED Three-Part Communication (3PC) summary.
    
    FIXES:
    - "That's correct" now properly detected as "acknowledged"
    - 3PC pattern detection improved
    - All segments included (no truncation)
    """
    segments = diarization_segments or []
    
    if not segments:
        return []
    
    logging.info(f"Starting 3PC with {len(segments)} segments")
    
    # STEP 1: Conservative merging
    segments = _merge_incomplete_segments(segments, max_gap=1.5)

    # STEP 2: Split verification phrases that are bundled with other speech
    segments = _split_segments_on_verification_phrases(segments)
    # STEP 2a: Recombine the remaining speech for the same speaker
    segments = _combine_non_verification_runs(segments)

    # STEP 3: Detect 3PC patterns
    segments = _detect_3pc_patterns(segments)
    _assign_communication_groups(segments)

    # Get semantic model
    try:
        model = _get_sentence_model()
    except:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    for segment in segments:
        segment["_segment_id"] = segment.get("_segment_id") or str(uuid.uuid4())

    reference_lines: List[str] = []
    if reference_text:
        reference_lines = [ln.strip() for ln in reference_text.splitlines() if ln.strip()]

    _classify_segment_roles(segments)
    _prepare_segment_features(segments, model)

    document_buckets, bucket_embeddings = _create_document_buckets(reference_lines, model)
    _assign_document_context(segments, document_buckets, bucket_embeddings)

    pairs = _build_pairs(segments, match_threshold, partial_threshold)
    summary_entries = _build_summary_entries(segments, pairs, document_buckets)

    summary_entries.sort(
        key=lambda e: (
            0 if e.get("start") is not None else 1,
            float(e.get("start") or 0),
        )
    )

    if len(summary_entries) > max_entries:
        logging.warning(f"3PC entries exceed {max_entries}, consider increasing limit")

    confirmations = sum(1 for e in summary_entries if e.get("three_pc_role") == "confirmation")
    logging.info(f"✓ 3PC Complete: {len(summary_entries)} entries, {confirmations} confirmations")

    return summary_entries
def append_three_pc_summary_to_pdf(pdf_path: str, entries: Optional[List[Dict]]) -> None:
    """
    Append 3PC summary to PDF with FIXED PAGINATION.
    Shows ALL entries across multiple pages.
    """
    import fitz
    import tempfile
    import textwrap
    import os
    
    doc = fitz.open(pdf_path)
    fd, temp_pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    saved = False

    try:
        margin = 36
        title_font_size = 16
        body_font_size = 10
        row_padding = 4
        content_width_ratio = [
            0.18,
            0.12,
            0.12,
            0.43,
            0.15,
        ]
        headers = [
            "Speaker",
            "Start",
            "End",
            "Content",
            "Status",
        ]
        
        status_colors = {
            "match": (0, 0.5, 0),
            "acknowledged": (0, 0.4, 0.8),
            "partial match": (1, 0.65, 0),
            "general conversation": (0.5, 0.5, 0.5),
            "mismatch": (0.75, 0, 0),
        }

        def _new_page(include_header: bool = True):
            page = doc.new_page()
            width = page.rect.width
            height = page.rect.height
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
                    page.draw_rect(rect, color=(0, 0, 0))
                    page.insert_textbox(
                        rect,
                        header,
                        fontsize=body_font_size,
                        fontname="helv",
                        align=fitz.TEXT_ALIGN_CENTER,
                    )
                    x += col_width
                y_pos += body_font_size + 8
            
            return page, y_pos, height

        def _wrap_cell(text: str, max_width: float) -> List[str]:
            if not text:
                return [""]

            approx_char_width = max(int(max_width / (body_font_size * 0.6)), 1)
            wrapped: List[str] = []
            for block in text.splitlines() or [""]:
                if not block:
                    wrapped.append("")
                    continue
                wrapped.extend(textwrap.wrap(block, width=approx_char_width) or [""])
            return wrapped or [""]

        def _draw_row(page, y_pos, page_height, values: List[str]):
            """Draw a row, creating new page if needed."""
            width = page.rect.width
            available_height = page_height - margin
            content_width = width - 2 * margin

            # Calculate row height
            columns = []
            max_lines = 1
            for value, ratio in zip(values, content_width_ratio):
                col_width = content_width * ratio
                lines = _wrap_cell(value, col_width - (2 * row_padding))
                columns.append((col_width, lines))
                max_lines = max(max_lines, len(lines))

            row_height = max_lines * (body_font_size + 2) + (2 * row_padding)

            # CHECK: Do we need a new page?
            if y_pos + row_height > available_height:
                page, y_pos, page_height = _new_page(include_header=True)
                width = page.rect.width
                content_width = width - 2 * margin
                
                # Recalculate columns for new page
                columns = []
                max_lines = 1
                for value, ratio in zip(values, content_width_ratio):
                    col_width = content_width * ratio
                    lines = _wrap_cell(value, col_width - (2 * row_padding))
                    columns.append((col_width, lines))
                    max_lines = max(max_lines, len(lines))
                row_height = max_lines * (body_font_size + 2) + (2 * row_padding)

            # Draw the row
            x = margin
            for idx, (col_width, lines) in enumerate(columns):
                rect = fitz.Rect(x, y_pos, x + col_width, y_pos + row_height)
                page.draw_rect(rect, color=(0.7, 0.7, 0.7))

                text_y = y_pos + row_padding + body_font_size
                color = (0, 0, 0)
                
                if headers[idx] == "Status":
                    status_key = values[idx].strip().lower()
                    base_status = status_key.split('(')[0].strip()
                    color = status_colors.get(base_status, color)

                for line in lines:
                    page.insert_text(
                        (x + row_padding, text_y),
                        line,
                        fontsize=body_font_size,
                        fontname="helv",
                        fill=color,
                    )
                    text_y += body_font_size + 2
                
                x += col_width

            return page, y_pos + row_height, page_height

        # Start first page
        page, y_cursor, page_height = _new_page()

        if not entries:
            page.insert_text(
                (margin, y_cursor),
                "No speaker diarization data available.",
                fontsize=body_font_size,
                fontname="helv",
            )
        else:
            # Draw ALL entries (fixed pagination)
            for idx, entry in enumerate(entries):
                start_ts = _format_timestamp(entry.get("start")) or "-"
                end_ts = _format_timestamp(entry.get("end")) or "-"

                status_text = (entry.get("status") or "").capitalize() or "-"
                if entry.get("three_pc_role"):
                    role = entry["three_pc_role"]
                    status_text += f" (3PC: {role})"

                confidence_value = entry.get("confidence")
                confidence_display = (
                    f"{confidence_value:.2f}"
                    if isinstance(confidence_value, (int, float))
                    else None
                )

                reasons = entry.get("reasons") or []

                content_parts: List[str] = []
                content_value = entry.get("content") or ""
                if content_value:
                    content_parts.append(content_value)

                reference_text = entry.get("reference")
                if reference_text:
                    content_parts.append(f"Doc: {reference_text}")

                entry_type = entry.get("type")
                if entry_type and entry_type != "Instruction":
                    content_parts.append(f"Type: {entry_type}")

                pair_display = entry.get("pair_display") or entry.get("pair")
                if pair_display:
                    role_display = entry.get("pair_role")
                    if role_display:
                        content_parts.append(f"Pair: {pair_display} ({role_display})")
                    else:
                        content_parts.append(f"Pair: {pair_display}")

                doc_status = entry.get("doc_status")
                if doc_status:
                    content_parts.append(f"Doc status: {doc_status}")

                doc_similarity = entry.get("doc_similarity")
                if isinstance(doc_similarity, (int, float)):
                    content_parts.append(f"Doc similarity: {doc_similarity:.2f}")

                doc_identifier = entry.get("doc_identifier")
                if doc_identifier:
                    content_parts.append(f"Step: {doc_identifier}")

                three_pc_role = entry.get("three_pc_role")
                if three_pc_role:
                    content_parts.append(f"3PC role: {three_pc_role}")

                if confidence_display:
                    content_parts.append(f"Confidence: {confidence_display}")

                if reasons:
                    content_parts.append("Reasons: " + ", ".join(reasons))

                comm_type = entry.get("communication_type")
                if comm_type:
                    content_parts.append(f"Communication: {comm_type.upper()}")

                content_text = "\n".join(content_parts) if content_parts else "-"

                row_values = [
                    entry.get("speaker") or "Unknown",
                    start_ts,
                    end_ts,
                    content_text,
                    status_text,
                ]

                page, y_cursor, page_height = _draw_row(page, y_cursor, page_height, row_values)
                
                # Log progress every 50 entries
                if (idx + 1) % 50 == 0:
                    logging.info(f"PDF: Added {idx + 1}/{len(entries)} entries")

        doc.save(temp_pdf_path, deflate=True)
        saved = True
        logging.info(f"✓ 3PC PDF saved: {len(entries)} entries across {len(doc)} pages")
        
    finally:
        doc.close()
        if saved:
            os.replace(temp_pdf_path, pdf_path)
        if os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
            except OSError:
                pass


def append_three_pc_summary_to_docx(docx_path: str, entries: Optional[List[Dict]]) -> None:
    """Append 3PC summary to DOCX with proper colors."""
    import docx
    from docx.shared import RGBColor
    
    document = docx.Document(docx_path)
    document.add_page_break()
    document.add_heading("Three-Part Communication (3PC) Summary", level=1)

    if not entries:
        document.add_paragraph("No speaker diarization data available.")
        document.save(docx_path)
        return

    headers = [
        "Speaker",
        "Start",
        "End",
        "Content",
        "Status",
    ]
    table = document.add_table(rows=1, cols=len(headers))
    try:
        table.style = "Light Grid Accent 1"
    except KeyError:
        table.style = "Light Grid"
    
    for cell, title in zip(table.rows[0].cells, headers):
        cell.text = title

    status_colors = {
        "match": RGBColor(0, 128, 0),
        "acknowledged": RGBColor(0, 100, 200),
        "partial match": RGBColor(255, 165, 0),
        "general conversation": RGBColor(128, 128, 128),
        "mismatch": RGBColor(192, 0, 0),
    }

    for entry in entries:
        row = table.add_row()
        cells = row.cells
        
        confidence_value = entry.get("confidence")
        confidence_display = (
            f"{confidence_value:.2f}" if isinstance(confidence_value, (int, float)) else None
        )

        reasons = entry.get("reasons") or []

        content_parts: List[str] = []
        content_value = entry.get("content") or ""
        if content_value:
            content_parts.append(content_value)

        reference_text = entry.get("reference")
        if reference_text:
            content_parts.append(f"Doc: {reference_text}")

        entry_type = entry.get("type")
        if entry_type and entry_type != "Instruction":
            content_parts.append(f"Type: {entry_type}")

        pair_display = entry.get("pair_display") or entry.get("pair")
        if pair_display:
            pair_role = entry.get("pair_role")
            if pair_role:
                content_parts.append(f"Pair: {pair_display} ({pair_role})")
            else:
                content_parts.append(f"Pair: {pair_display}")

        doc_status = entry.get("doc_status")
        if doc_status:
            content_parts.append(f"Doc status: {doc_status}")

        doc_similarity = entry.get("doc_similarity")
        if isinstance(doc_similarity, (int, float)):
            content_parts.append(f"Doc similarity: {doc_similarity:.2f}")

        doc_identifier = entry.get("doc_identifier")
        if doc_identifier:
            content_parts.append(f"Step: {doc_identifier}")

        three_pc_role = entry.get("three_pc_role")
        if three_pc_role:
            content_parts.append(f"3PC role: {three_pc_role}")

        if confidence_display:
            content_parts.append(f"Confidence: {confidence_display}")

        if entry.get("pair_status"):
            content_parts.append(f"Pair status: {entry.get('pair_status')}")

        comm_type = entry.get("communication_type")
        if comm_type:
            content_parts.append(f"Communication: {comm_type.upper()}")

        if reasons:
            content_parts.append("Reasons: " + ", ".join(reasons))

        details_text = "\n".join(content_parts) if content_parts else "-"

        start_ts = _format_timestamp(entry.get("start")) or "-"
        end_ts = _format_timestamp(entry.get("end")) or "-"

        status_value = entry.get("status") or "-"

        values = [
            entry.get("speaker") or "Unknown",
            start_ts,
            end_ts,
            details_text,
            status_value,
        ]

        for idx, value in enumerate(values):
            cell = cells[idx]
            text_value = value if isinstance(value, str) else str(value)

            if headers[idx] == "Content" and "\n" in text_value:
                cell.text = ""
                lines = text_value.splitlines()
                if not lines:
                    cell.text = ""
                else:
                    cell.text = lines[0]
                    for line in lines[1:]:
                        cell.add_paragraph(line)
            else:
                cell.text = text_value

        status = (entry.get("status") or "").lower()
        status_idx = headers.index("Status")
        status_cell = cells[status_idx]

        status_text = status.capitalize() if status else "-"
        if entry.get("three_pc_role"):
            status_text += f" (3PC: {entry.get('three_pc_role')})"

        status_cell.text = status_text

        if status in status_colors:
            for paragraph in status_cell.paragraphs:
                for run in paragraph.runs:
                    run.font.color.rgb = status_colors[status]

    document.save(docx_path)
    logging.info(f"✓ 3PC DOCX saved: {len(entries)} entries")

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
