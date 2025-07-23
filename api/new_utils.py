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

def diarization_from_audio(audio_url, transcript_segments, transcript_words=None):
    import requests, tempfile, os, subprocess
    from pyannote.audio import Pipeline
    from django.conf import settings
    
    # Download audio
    local_audio_path = os.path.join(tempfile.gettempdir(), f"diar_{os.path.basename(audio_url)}")
    with requests.get(audio_url, stream=True) as r:
        r.raise_for_status()
        with open(local_audio_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # Convert to wav if needed
    wav_path = local_audio_path
    if not local_audio_path.lower().endswith('.wav'):
        wav_path = local_audio_path.rsplit('.', 1)[0] + '.wav'
        command = [
            'ffmpeg', '-y', '-i', local_audio_path,
            '-ar', '16000', '-ac', '1', wav_path
        ]
        subprocess.run(command, check=True)
        os.unlink(local_audio_path)
    
    # Run diarization
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=settings.HF_TOKEN
    )
    diarization = pipeline(wav_path)
    
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
        """
        Merge consecutive segments from the same speaker if they're close together.
        """
        if not segments:
            return segments
        
        merged = []
        current_segment = segments[0].copy()
        
        for i in range(1, len(segments)):
            next_segment = segments[i]
            
            # Check if same speaker and segments are close
            if (current_segment['speaker'] == next_segment['speaker'] and 
                next_segment['start'] - current_segment['end'] <= max_gap):
                
                # Merge segments
                current_segment['end'] = next_segment['end']
                if next_segment['text'].strip():
                    if current_segment['text'].strip():
                        current_segment['text'] += " " + next_segment['text']
                    else:
                        current_segment['text'] = next_segment['text']
            else:
                # Different speaker or gap too large, save current and start new
                merged.append(current_segment)
                current_segment = next_segment.copy()
        
        # Add the last segment
        merged.append(current_segment)
        return merged
    
    # Process diarization results
    diarization_segments = []
    
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
            diarization_segments.append({
                "speaker": speaker,
                "start": seg_start,
                "end": seg_end,
                "text": segment_text.strip(),
                "duration": round(seg_end - seg_start, 2)
            })
    
    # Merge consecutive segments from the same speaker
    diarization_segments = merge_consecutive_same_speaker_segments(diarization_segments)
    
    # Filter out very short segments with no text
    diarization_segments = [
        seg for seg in diarization_segments 
        if seg['text'].strip() or seg['duration'] > 1.0
    ]
    
    # Clean up
    os.unlink(wav_path)
    
    return diarization_segments
