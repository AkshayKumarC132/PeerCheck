# PeerCheck

This project is an offline-capable, edge-deployed system designed to process audio prompts, analyze them, and respond in a multimodal manner through an API

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x installed on your machine.
- Git installed on your machine.
- A virtual environment manager (optional, but recommended).

## Installation

Follow these steps to get your development environment set up:

1. **Clone the repository:**

   Open your terminal and run:

   ```bash
   git clone https://github.com/AkshayKumarC132/PeerCheck
   cd PeerCheck

## Setting up a Virtual Environment (Optional but Recommended)

2. Creating a virtual environment isolates your project dependencies and prevents conflicts with other Python projects.
### Activating the Virtual Environment 

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

#### For macOS/Linux:
```bash
   source venv/bin/activate
```
## Installing Dependencies

After activating your virtual environment, install the required dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Running the Development Server

Start the Django development server:
```bash
python manage.py runserver
```

The application will be accessible at `http://127.0.0.1:8000/`.

## Improved Speaker Diarization

The transcription pipeline clusters speaker embeddings to reduce
over-segmentation and more accurately detect the true number of speakers.
If the expected speaker count is known, it uses Agglomerative Clustering to
return exactly that many groups. Otherwise, DBSCAN groups embeddings by
similarity. When DBSCAN either over-splits or collapses all segments into a
single cluster, a fallback Agglomerative step re-clusters the embeddings to
produce a more reasonable number of speakers. Clusters are no longer dropped
based on short duration, ensuring every detected speaker receives a label.
Detected speaker embeddings are matched to stored profiles so the same
real-world speaker receives a consistent label across recordings. These
profiles ensure that a familiar voice is labelled consistently across
uploads.

## Procedure Validation API

The `validate-procedure/` endpoint allows you to upload an audio file along with
the original procedure document (PDF, DOCX or plain text). After transcription,
the API extracts the instructions, aligns the conversation to the first matching
step and highlights any missed instructions. PDF and DOCX files are automatically
converted to text before comparison.

Example usage:

```bash
curl -X POST -F "file=@meeting.wav" \
     -F "procedure_document=@procedure.pdf" \
     http://localhost:8000/api/validate-procedure/<token>/
```

The response includes `procedure_comparison.results` for each step as well as
`highlighted_document_markdown` and `highlighted_document_html` showing missed
instructions in **bold** or in red.
