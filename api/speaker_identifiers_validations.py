from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from fpdf import FPDF
import os

model = SentenceTransformer('all-mpnet-base-v2')

# Your transcript data here...
transcript = []

confirm_phrases = ["that's correct", "confirmed", "absolutely", "yes, that's right", "indeed"]

# Initialize PDF
pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=10)
pdf.set_font("Courier", size=8)

columns = [
    ("Type", 25),
    ("Time", 20),
    ("Speaker", 30),
    ("Phrase", 95),
    ("Section", 40),
    ("Conf", 10)
]

# Header
pdf.set_font("Courier", style="B", size=8)
col_names = [col[0] for col in columns]
col_widths = [col[1] for col in columns]

def print_header():
    for col_name, col_width in columns:
        pdf.cell(col_width, 8, col_name, border=1, align='C')
    pdf.ln()
    pdf.set_font("Courier", size=8)
    pdf.line(10, pdf.get_y(), 10 + sum(col_widths), pdf.get_y())
    pdf.ln(2)

print_header()

def extract_sections(text):
    # Enhanced regex to capture steps, pages, and specific identifiers
    return re.findall(r'(?:step\s*)?(\d+(?:\.\d+)*)|page\s*(\d+)|(\d{3,})', text, re.IGNORECASE)

def is_confirmation(text):
    # Strict check for confirmation phrases only
    return any(phrase in text.lower() for phrase in confirm_phrases)

def get_row_type(text, has_sections):
    if is_confirmation(text):
        return "Confirmation"
    return "Action" if has_sections else "Response"

def get_fill_color(row_type):
    return {
        "Confirmation": (230, 255, 230),
        "Action": (220, 240, 255),
        "Response": (255, 245, 200)
    }.get(row_type, (255, 220, 220))

def check_page_space(required_height):
    if pdf.get_y() + required_height > pdf.h - pdf.b_margin:
        pdf.add_page()
        pdf.set_font("Courier", size=8)
        print_header()

def print_row(row_data, row_type, col_widths):
    fill_color = get_fill_color(row_type)
    x_start = pdf.get_x()
    y_start = pdf.get_y()
    # Estimate number of lines needed for wrapping
    line_counts = []
    for i, text in enumerate(row_data):
        lines = pdf.multi_cell(col_widths[i], 8, str(text), border=0, align='L', split_only=True)
        line_counts.append(len(lines))
    row_height = max(line_counts) * 8
    check_page_space(row_height)
    pdf.set_fill_color(*fill_color)
    for i, text in enumerate(row_data):
        pdf.set_xy(x_start + sum(col_widths[:i]), y_start)
        pdf.multi_cell(col_widths[i], 8, str(text), border=1, fill=True)
    pdf.set_xy(x_start, y_start + row_height)

# Identify structured interactions
for idx in range(len(transcript)):
    current = transcript[idx]
    if is_confirmation(current['text']):
        speaker_confirm = current['speaker']
        timestamp = f"{current['start']:.2f}s"
        conf_value = "0.95"

        # Collect up to two previous utterances in chronological order
        responses = []
        sections = []
        if idx >= 2:  # Ensure there are enough prior utterances
            responses = [transcript[idx - 2], transcript[idx - 1]]
            for resp in responses:
                sections_found = extract_sections(resp['text'])
                sections.extend([s for s in sections_found if s])  # Filter out empty matches

        # Estimate height and force page break if needed
        group_items = responses + [current]
        group_heights = [len(pdf.multi_cell(col_widths[3], 8, item['text'], border=0, align='L', split_only=True)) for item in group_items]
        total_group_height = sum(h * 8 for h in group_heights) + len(group_items) * 10
        check_page_space(total_group_height)

        # Print responses in chronological order
        for resp in responses:
            row_type = get_row_type(resp['text'], bool(extract_sections(resp['text'])))
            section = ", ".join(filter(None, extract_sections(resp['text']))) if row_type == "Action" else ""
            print_row([
                row_type,
                f"{resp['start']:.2f}s",
                resp['speaker'],
                resp['text'],
                section,
                ""
            ], row_type, col_widths)

        # Print confirmation
        print_row([
            "Confirmation",
            timestamp,
            speaker_confirm,
            current['text'],
            "",
            conf_value
        ], "Confirmation", col_widths)

        pdf.ln(4)  # Space between interaction groups

# Save PDF
pdf_output_path = "08__aligned_speaker_confirmation_report.pdf"
pdf.output(pdf_output_path)

# Auto-open PDF
try:
    if os.name == 'nt':
        os.startfile(pdf_output_path)
    elif os.uname().sysname == 'Darwin':
        os.system(f'open "{pdf_output_path}"')
    else:
        os.system(f'xdg-open "{pdf_output_path}"')
except Exception as e:
    print(f"Could not open PDF automatically: {e}")