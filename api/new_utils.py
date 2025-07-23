import os
import re
import string
import tempfile
import pythoncom
import boto3
from peercheck import settings
from docx import Document as DocxDocument
from pdf2docx import Converter
import win32com.client
import whisper
import PyPDF2
import docx
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from io import BytesIO

# Load Whisper model once
model = whisper.load_model(getattr(settings, 'WHISPER_MODEL', 'base.en'))

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
    """Transcribe audio file stored in S3"""
    s3_key = get_s3_key_from_url(s3_url)
    temp_path = download_file_from_s3(s3_key)
    
    try:
        result = model.transcribe(temp_path)
        return result["text"]
    finally:
        # Clean up temp file
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

def create_highlighted_docx_from_s3(text_s3_url, transcript, threshold=0.6):
    """Create a DOCX with highlighted text and upload to S3"""
    s3_key = get_s3_key_from_url(text_s3_url)
    temp_input = download_file_from_s3(s3_key)
    
    try:
        norm_trans = [normalize_line(ln) for ln in transcript.splitlines() if ln.strip()]
        
        ext = s3_key.rsplit('.', 1)[1].lower()
        
        # Convert to DOCX if needed
        if ext == 'docx':
            docx_in = temp_input
        elif ext == 'pdf':
            docx_in = tempfile.NamedTemporaryFile(delete=False, suffix='.docx').name
            Converter(temp_input).convert(docx_in, start=0, end=None)
        else:  # txt
            docx_in = tempfile.NamedTemporaryFile(delete=False, suffix='.docx').name
            d = DocxDocument()
            with open(temp_input, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip():
                        d.add_paragraph(line.strip())
            d.save(docx_in)
        
        # Create output file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.docx').name
        
        # Use Word COM to highlight text
        pythoncom.CoInitialize()
        try:
            word = win32com.client.Dispatch('Word.Application')
            word.Visible = False
            word.DisplayAlerts = False
            
            wdoc = word.Documents.Open(os.path.abspath(docx_in))
            
            RED = 255
            GREEN = 65280
            
            # Highlight paragraphs
            for para in wdoc.Paragraphs:
                txt = para.Range.Text.strip()
                if not txt:
                    continue
                norm = normalize_line(txt)
                best = max((fuzz.token_set_ratio(norm, t) for t in norm_trans), default=0) / 100.0
                para.Range.Font.Color = GREEN if best >= threshold else RED
            
            # Highlight shapes/text boxes
            for shp in wdoc.Shapes:
                if shp.TextFrame.HasText:
                    for p2 in shp.TextFrame.TextRange.Paragraphs:
                        txt = p2.Range.Text.strip()
                        if not txt:
                            continue
                        norm = normalize_line(txt)
                        best = max((fuzz.token_set_ratio(norm, t) for t in norm_trans), default=0) / 100.0
                        p2.Range.Font.Color = GREEN if best >= threshold else RED
            
            wdoc.SaveAs(os.path.abspath(output_path), FileFormat=12)
            wdoc.Close(False)
            word.Quit()
            
        finally:
            pythoncom.CoUninitialize()
        
        # Upload to S3
        import uuid
        output_s3_key = f"processed/{uuid.uuid4()}_{os.path.basename(output_path)}"
        with open(output_path, 'rb') as f:
            output_s3_url = upload_file_to_s3(f, output_s3_key)
        
        # Clean up temporary files
        os.unlink(temp_input)
        if docx_in != temp_input:
            os.unlink(docx_in)
        os.unlink(output_path)
        
        return output_s3_url
        
    except Exception as e:
        # Clean up on error
        os.unlink(temp_input)
        if 'docx_in' in locals() and docx_in != temp_input:
            os.unlink(docx_in)
        if 'output_path' in locals():
            os.unlink(output_path)
        raise e
