from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import uuid
import requests

# IBM Watson STT
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Qdrant Vector DB
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Sentence Embedding
from sentence_transformers import SentenceTransformer

# ------------------ Flask Setup ------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("app", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ------------------ IBM Watson STT ------------------
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# ✅ IBM Watson credentials (from your JSON)
ibm_api_key = "xNl72WpEUZKzwVfGXqBgHmmE1Kd6UE1HTKIN43m96OcH"
ibm_url = "https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/41109ef9-d288-4b00-839c-62c2674dd59b"

# ✅ Setup STT service
speech_to_text = None
try:
    authenticator = IAMAuthenticator(ibm_api_key)
    speech_to_text = SpeechToTextV1(authenticator=authenticator)
    speech_to_text.set_service_url(ibm_url)
    print("✅ IBM Watson STT initialized successfully")
except Exception as e:
    print("⚠️ IBM Watson STT setup failed:", e)

# ✅ Route for audio transcription
@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    print("🎤 Received audio size:", len(audio_bytes))  # Debugging audio length
    audio_file.seek(0)

    if len(audio_bytes) < 500:  # Prevent empty/short uploads
        return jsonify({"error": "Empty or too short audio"}), 400

    try:
        result = speech_to_text.recognize(
            audio=audio_file,
            content_type="audio/webm;codecs=opus",  # Make sure frontend matches this
            model="en-US_BroadbandModel"            # Change to "hi-IN_BroadbandModel" if Hindi
        ).get_result()

        print("📝 Watson raw result:", result)  # Debug Watson response
        text = result["results"][0]["alternatives"][0]["transcript"] if result.get("results") else ""
        return jsonify({"text": text})

    except Exception as e:
        print("❌ Watson STT error:", e)
        return jsonify({"error": str(e)}), 500

# ------------------ Granite LLM via API ------------------
GRANITE_API_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"
GRANITE_API_KEY = "T0uWLVdrR6uANl6MMaP0yamcRUTc_-EoxF5W6CjSJdXL"
GRANITE_PROJECT_ID = "e1daced7-aac0-44e6-a227-f672004a57e1"
GRANITE_MODEL_ID = "ibm/granite-3-3-8b-instruct"

def call_granite_llm(prompt_text, max_tokens=2000):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GRANITE_API_KEY}"
    }

    body = {
        "project_id": GRANITE_PROJECT_ID,
        "model_id": GRANITE_MODEL_ID,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "prompt": prompt_text
    }

    try:
        response = requests.post(GRANITE_API_URL, headers=headers, json=body)
        response.raise_for_status()
        return response.json().get("output", "")
    except Exception as e:
        print("Granite API error:", e)
        return "Error calling Granite LLM"

# ------------------ Qdrant Setup ------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings

# Create collection for resumes
try:
    qdrant.recreate_collection(
        collection_name="resumes",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("✅ Qdrant collection 'resumes' ready")
except Exception as e:
    print("⚠️ Qdrant setup error:", e)

# ------------------ Role & Skills ------------------
ROLE_SKILLS = {
    "Software Engineer": ["Python","Data Structures","Algorithms","Git","OOP","Problem Solving","REST APIs","Unit Testing"],
    "Data Scientist": ["Python","Pandas","NumPy","Machine Learning","Statistics","Data Visualization","SQL","Feature Engineering"],
    "AI/ML Engineer": ["Python","TensorFlow","PyTorch","ML","Deep Learning","NLP","Computer Vision","Model Optimization"],
    "Full Stack Developer": ["HTML","CSS","JavaScript","React","Node.js","Database Design","API Integration","Responsive Design"],
    "HR Manager": ["Recruitment","Employee Engagement","Communication","Onboarding","Conflict Resolution","Performance Appraisal","HR Analytics","Leadership"]
}

ROLE_COURSES = {
    "Software Engineer": ["Python Bootcamp","Advanced DS & Algorithms","Clean Code Practices","System Design Interview Prep"],
    "Data Scientist": ["Machine Learning A-Z","Statistics for Data Science","Data Visualization with Tableau","SQL for Data Science"],
    "AI/ML Engineer": ["Deep Learning Specialization","AI for Everyone","Computer Vision with TensorFlow","NLP with Hugging Face"],
    "Full Stack Developer": ["The Web Developer Bootcamp","React JS Masterclass","Node.js & Express","Database Design & SQL"],
    "HR Manager": ["HR Analytics Certification","Leadership & Management","Conflict Resolution Skills","Strategic HR Business Partner"]
}

GENERAL_SECTIONS = ["Professional Summary","Experience","Education","Contact Info","Skills"]
RELEVANT_SKILL_SECTIONS = ["experience","work experience","projects","skills","technical skills","technical","education","professional summary"]

# ------------------ Mock Interview Questions ------------------
interview_data = {
    "Infosys": {
        "Software Engineer": [
            {"q": "Why do you want to join Infosys?", "keywords": ["global","innovation","career growth","digital transformation"]},
            {"q": "Explain OOP pillars.", "keywords": ["encapsulation","inheritance","polymorphism","abstraction"]},
            {"q": "How do you optimize code?", "keywords": ["time complexity","space complexity","refactor","efficiency"]},
            {"q": "Explain Agile methodology.", "keywords": ["sprint","scrum","iteration","collaboration"]},
            {"q": "How do you debug errors?", "keywords": ["logging","breakpoints","unit testing","traceback"]}
        ]
    }
}

# ------------------ Resume Parsing ------------------
def extract_text_from_pdf(filepath):
    text = ""
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + " "
    except Exception as e:
        print("PDF parsing error:", e)
    return text.lower()

def extract_text_from_docx(filepath):
    text = ""
    try:
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text += para.text + " "
    except Exception as e:
        print("DOCX parsing error:", e)
    return text.lower()

def extract_resume_text(filepath):
    ext = filepath.split(".")[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(filepath)
    elif ext in ["docx","doc"]:
        return extract_text_from_docx(filepath)
    else:
        return ""

def extract_relevant_text(resume_text):
    lines = resume_text.split("\n")
    relevant_text = ""
    current_section = ""
    for line in lines:
        line_lower = line.lower()
        if any(sec in line_lower for sec in RELEVANT_SKILL_SECTIONS):
            current_section = line_lower
        if any(sec in current_section for sec in RELEVANT_SKILL_SECTIONS):
            relevant_text += " " + line_lower
    return relevant_text

# ------------------ Evaluation ------------------
def evaluate_answers(company, role, user_answers):
    questions = interview_data.get(company, {}).get(role, [])
    results = []
    total_score = 0
    for idx, ans in enumerate(user_answers):
        if idx >= len(questions): continue
        q = questions[idx]
        ans_lower = ans.lower()
        matched = [kw for kw in q["keywords"] if kw.lower() in ans_lower]
        score = int((len(matched)/len(q["keywords"]))*100) if q["keywords"] else 0
        missing_keywords = [kw for kw in q["keywords"] if kw.lower() not in ans_lower]
        total_score += score
        results.append({"question": q["q"], "user_answer": ans, "score": score, "missing_keywords": missing_keywords})
    overall_score = total_score // len(user_answers) if user_answers else 0
    return results, overall_score

# ------------------ Routes ------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze_resume", methods=["POST"])
def analyze_resume():
    if "resume" not in request.files:
        return "No resume uploaded", 400
    file = request.files["resume"]
    if file.filename == "":
        return "Empty filename", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    role = request.form.get("role","Software Engineer")
    resume_text = extract_resume_text(filepath)
    relevant_text = extract_relevant_text(resume_text)

    ideal_skills = ROLE_SKILLS.get(role, [])
    matched_skills = [skill for skill in ideal_skills if skill.lower() in relevant_text]
    missing_skills = list(set(ideal_skills)-set(matched_skills))
    present_sections = [sec for sec in GENERAL_SECTIONS if sec.lower() in resume_text]
    missing_sections = list(set(GENERAL_SECTIONS)-set(present_sections))
    skill_score = int((len(matched_skills)/len(ideal_skills))*70) if ideal_skills else 0
    section_score = int((len(present_sections)/len(GENERAL_SECTIONS))*30) if GENERAL_SECTIONS else 0
    ats_score = skill_score + section_score
    second_round_prob = min(100, ats_score+5)
    recommended_courses = ROLE_COURSES.get(role, [])
    general_feedback = []
    if "Professional Summary" not in present_sections: general_feedback.append("Add a clear professional summary at the top of your resume.")
    if "Experience" not in present_sections: general_feedback.append("Include your work experience with achievements.")
    if "Education" not in present_sections: general_feedback.append("Mention your educational qualifications.")
    if "Contact Info" not in present_sections: general_feedback.append("Add your contact information.")
    if "Skills" not in present_sections: general_feedback.append("Add a dedicated 'Skills' section.")
    general_feedback.append("Use action verbs like 'developed', 'led', 'implemented'.")
    general_feedback.append("Quantify results wherever possible.")
    general_feedback.append("Tailor keywords to match the job description.")

    # ------------------ Store in Qdrant ------------------
    resume_summary = call_granite_llm(f"Summarize this resume: {resume_text}")
    embedding_vector = embedding_model.encode(resume_text).tolist()

    qdrant.upsert(
        collection_name="resumes",
        points=[{
            "id": str(uuid.uuid4()),
            "vector": embedding_vector,
            "payload": {
                "text": resume_text,
                "summary": resume_summary,
                "filename": filename,
                "role": role,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "ats_score": ats_score
            }
        }]
    )

    return render_template("resume_dashboard.html",
                           role=role,
                           score=ats_score,
                           common_skills=matched_skills,
                           missing_skills=missing_skills,
                           courses=recommended_courses,
                           second_round_prob=second_round_prob,
                           general_feedback=general_feedback)

# ------------------ Interview Routes ------------------
@app.route("/mock_interview", methods=["POST"])
def mock_interview():
    company = request.form.get("company", "Generic Company")
    role = request.form.get("role", "Software Engineer")
    questions = interview_data.get(company, {}).get(role, [])
    company_images = {
        "Infosys": "infosys.jpg",
        "TCS": "tcs.jpg",
        "Wipro": "wipro.jpg",
        "HCL": "hcl.jpg",
        "Tech Mahindra": "techmahindra.jpg"
    }
    bg_image = url_for('static', filename=f'images/{company_images.get(company, "default.jpg")}')
    return render_template("interview.html",
                           company=company,
                           role=role,
                           questions=questions,
                           bg_url=bg_image)

@app.route("/submit_interview", methods=["POST"])
def submit_interview():
    company = request.form.get("company")
    role = request.form.get("role")
    user_answers = request.form.getlist("answers")
    results, overall_score = evaluate_answers(company, role, user_answers)
    return render_template("result.html", 
                           company=company, 
                           role=role, 
                           results=results, 
                           overall_score=overall_score,
                           bg_url="/static/images/interview-bg.jpg")

@app.route("/granite_helper", methods=["POST"])
def granite_route():
    user_text = request.form.get("text")
    suggestion = call_granite_llm(f"Improve this interview answer: {user_text}")
    return jsonify({"input": user_text, "suggestion": suggestion})

# ------------------ Transcribe Speech to Text ------------------
@app.route("/transcribe", methods=["POST"])
def transcribe_route():
    """Route to receive audio file and return transcript"""
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    transcript, error = transcribe_audio_file(audio_file)
    if error:
        return jsonify({"error": error}), 500
    return jsonify({"text": transcript})

# ------------------ Run ------------------
if __name__ == "__main__":
    app.run(debug=True)
