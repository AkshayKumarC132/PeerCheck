# PeerCheck

PeerCheck is an offline-capable system for uploading audio, transcribing it with speaker diarization, and managing Standard Operating Procedures (SOPs) and feedback through a REST API.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features
- Upload audio files which are stored on AWS S3
- Automatic transcription with optional speaker diarization
- SOP creation with step tracking and keyword matching
- Session management linking multiple audio files
- Feedback and review workflow with role-based permissions
- User and system settings endpoints

## Prerequisites
- Python 3.x installed
- Git installed
- Optional: a virtual environment manager

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/AkshayKumarC132/PeerCheck
cd PeerCheck
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Set the following environment variables before running the server:
- `DJANGO_SECRET_KEY`
- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT` (optional)
- `AWS_STORAGE_BUCKET_NAME`, `AWS_S3_REGION_NAME`, `AWS_S3_ACCESS_KEY_ID`, `AWS_S3_SECRET_ACCESS_KEY`


## Usage
Run the Django development server:
```bash
python manage.py runserver
```
The application will be available at `http://127.0.0.1:8000/`.

## Contributing
Pull requests are welcome. Please open an issue first to discuss any major changes.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

