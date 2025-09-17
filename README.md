## SmartHireAI

SmartHireAI is a Flask web app that helps with resume analysis and running mock interviews. It extracts resume content, checks for relevant sections and skills, and optionally integrates with IBM Watson Speech-to-Text for an interview transcription flow. The application is designed to run even without any external APIs enabled.

---

## Features

- Resume analysis with ATS-style scoring and improvement suggestions
- Role-based skill matching and course recommendations
- Mock interview questions by company and role
- Optional speech transcription via IBM Watson STT (can be disabled)
- Optional vector storage (Qdrant) and embedding generation (sentence-transformers)

---

## Project Structure

```
Smart-Hire-AI-Agent-/
├─ app/
│  ├─ app.py                  # Flask app and routes
│  ├─ templates/
│  │  ├─ index.html
│  │  ├─ interview.html
│  │  └─ result.html
│  └─ static/
│     ├─ style.css
│     ├─ script.js
│     └─ images/
├─ requirements.txt           # Dependencies file
├─ Procfile.txt               # Process file for PaaS (e.g., Heroku/Render)
├─ runtime.txt                # Runtime hint for PaaS
└─ README.md
```

---

## Prerequisites

- Python 3.11+ recommended (Windows/macOS/Linux)
- PowerShell (Windows) or bash/zsh (macOS/Linux)
- Internet access for first-time model/package downloads (optional if fully offline)

Optional services (only if you want these features enabled):
- IBM Cloud account and Speech to Text service
  - `IBM_API_KEY`
  - `IBM_URL` (e.g., `https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/XXXX`)
- Qdrant vector DB (local or cloud) and sentence-transformers model download
  - `QDRANT_URL`, `QDRANT_API_KEY` (if applicable)

---

## Installation Steps

### 1) Create and activate a virtual environment

Windows (PowerShell):
```powershell
cd C:\Users\Anjana\projects\Smart-Hire-AI-Agent-
python -m venv venv
./venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
```

macOS/Linux:
```bash
cd ~/projects/Smart-Hire-AI-Agent-
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

If you plan to run fully offline, you can still install the subset you need for core resume analysis. At minimum:
```bash
pip install Flask PyPDF2 python-docx
```

### 3) (Optional) Configure environment variables
If you want to enable external services, set the following before running:

Windows (PowerShell):
```powershell
$env:IBM_API_KEY="<your_key>"
$env:IBM_URL="<your_service_url>"
$env:QDRANT_URL="http://localhost:6333"   # if using local Qdrant
```

macOS/Linux:
```bash
export IBM_API_KEY="<your_key>"
export IBM_URL="<your_service_url>"
export QDRANT_URL="http://localhost:6333"
```

Offline mode (disables external APIs/components):
```powershell
$env:OFFLINE_MODE="1"
```

---

## Run (Local Development)

```bash
python app/app.py
```

Then open `http://127.0.0.1:5000/` in your browser.

Notes:
- If you see a message like "IBM STT disabled", the app is running without IBM STT (expected if no keys).
- On first run, if embeddings are enabled, `sentence-transformers` may download a model (requires internet).

---

## Build & Deployment Steps

This repository includes `Procfile.txt` and `runtime.txt` to ease deployment to PaaS platforms. Two example approaches are below.

### A) Render (recommended simple PaaS)
1. Push this repo to GitHub/GitLab.
2. Create a new Render Web Service → connect the repo.
3. Environment:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `python app/app.py`
   - Env vars: set `IBM_API_KEY`, `IBM_URL` if using STT. Optionally `OFFLINE_MODE=1` to disable all external calls.
4. Deploy.

### B) Heroku (if available in your environment)
1. Ensure the following files exist: `Procfile.txt` and `runtime.txt`.
   - `Procfile.txt` (example): `web: python app/app.py`
   - `runtime.txt` (example): `python-3.11.9`
2. Commands:
```bash
heroku create <your-app-name>
heroku buildpacks:set heroku/python
heroku config:set IBM_API_KEY=... IBM_URL=... OFFLINE_MODE=1
git push heroku main
```

### C) Bare VM / Server
1. SSH into the server.
2. Install Python 3.11+ and `pip`.
3. Follow the local Installation Steps above.
4. (Optional) Use a production WSGI server like `gunicorn`:
```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:8000 app.app:app
```

---

## Demo Instructions

### Home page
Visit `http://127.0.0.1:5000/` to access the homepage.

### Resume Analysis
1. Upload a resume (`.pdf` or `.docx`).
2. Select a role.
3. Submit to view:
   - ATS-style score out of 100
   - Detected vs missing skills
   - Recommended courses
   - General feedback for resume improvements

### Mock Interview
1. Select a company and role.
2. Answer the displayed questions.
3. Submit to see per-question evaluation and an overall score in the results page.

### Speech-to-Text (optional)
If IBM STT is configured, POST audio to `/transcribe` from frontend or a tool like `curl`.

Without STT set, you can test the route by posting `text` along with a dummy `audio` file; the server will echo your text (for offline testing):
```bash
curl -X POST http://127.0.0.1:5000/transcribe \
  -F "audio=@/bin/ls" \
  -F "text=This is my spoken answer used as a placeholder."
```

Expected outputs:
- JSON with `text` field containing the transcript (real or placeholder)
- Pages render with scores and feedback

---

## Dependencies File

All Python dependencies are listed in `requirements.txt` at the repository root. Install with:
```bash
pip install -r requirements.txt
```

Core runtime libraries used in the app:
- Flask (web framework)
- PyPDF2 (PDF text extraction)
- python-docx (DOCX parsing)
- (Optional) ibm-watson, ibm-cloud-sdk-core (IBM STT)
- (Optional) sentence-transformers (embeddings)
- (Optional) qdrant-client (vector DB)

---

## Troubleshooting

- ModuleNotFoundError (e.g., `No module named 'PyPDF2'`)
  - Activate your venv and run `pip install -r requirements.txt`.

- STT disabled message
  - Set `IBM_API_KEY` and `IBM_URL` or run with `OFFLINE_MODE=1` to intentionally disable.

- Port already in use
  - Change the port or kill the running process. Example:
    ```powershell
    $env:PORT="5001"; python app/app.py
    ```

- First run is slow
  - If embeddings are enabled, the model download can be large and take time.

---

## License

This project is provided as-is for demonstration and educational purposes.
