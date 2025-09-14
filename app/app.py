from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import PyPDF2
import docx
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# ------------------ Flask Setup ------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("app", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ------------------ IBM Watson Setup ------------------
ibm_api_key = os.getenv("IBM_API_KEY")
ibm_url = os.getenv("IBM_URL")

speech_to_text = None
if ibm_api_key and ibm_url:
    try:
        authenticator = IAMAuthenticator(ibm_api_key)
        speech_to_text = SpeechToTextV1(authenticator=authenticator)
        speech_to_text.set_service_url(ibm_url)
    except Exception as e:
        print("⚠️ IBM Watson setup failed:", e)

# ------------------ Role Skill Sets ------------------
ROLE_SKILLS = {
    "Software Engineer": [
        "Python", "Data Structures", "Algorithms", "Git", "OOP",
        "Problem Solving", "REST APIs", "Unit Testing"
    ],
    "Data Scientist": [
        "Python", "Pandas", "NumPy", "Machine Learning", "Statistics",
        "Data Visualization", "SQL", "Feature Engineering"
    ],
    "AI/ML Engineer": [
        "Python", "TensorFlow", "PyTorch", "ML", "Deep Learning",
        "NLP", "Computer Vision", "Model Optimization"
    ],
    "Full Stack Developer": [
        "HTML", "CSS", "JavaScript", "React", "Node.js",
        "Database Design", "API Integration", "Responsive Design"
    ],
    "HR Manager": [
        "Recruitment", "Employee Engagement", "Communication", "Onboarding",
        "Conflict Resolution", "Performance Appraisal", "HR Analytics", "Leadership"
    ]
}

# Expanded Recommended Courses
ROLE_COURSES = {
    "Software Engineer": [
        "Python Bootcamp",
        "Advanced DS & Algorithms",
        "Clean Code Practices",
        "System Design Interview Prep"
    ],
    "Data Scientist": [
        "Machine Learning A-Z",
        "Statistics for Data Science",
        "Data Visualization with Tableau",
        "SQL for Data Science"
    ],
    "AI/ML Engineer": [
        "Deep Learning Specialization",
        "AI for Everyone",
        "Computer Vision with TensorFlow",
        "NLP with Hugging Face"
    ],
    "Full Stack Developer": [
        "The Web Developer Bootcamp",
        "React JS Masterclass",
        "Node.js & Express",
        "Database Design & SQL"
    ],
    "HR Manager": [
        "HR Analytics Certification",
        "Leadership & Management",
        "Conflict Resolution Skills",
        "Strategic HR Business Partner"
    ]
}

# General resume sections to check
GENERAL_SECTIONS = ["Professional Summary", "Experience", "Education", "Contact Info", "Skills"]
RELEVANT_SKILL_SECTIONS = ["experience", "work experience", "projects", "skills", "technical skills", "technical"]

# ------------------ Helper Functions ------------------
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
    elif ext in ["docx", "doc"]:
        return extract_text_from_docx(filepath)
    else:
        return ""

def extract_relevant_text(resume_text):
    """Return only text under relevant skill sections."""
    lines = resume_text.split("\n")
    relevant_text = ""
    current_section = ""
    for line in lines:
        line_lower = line.lower()
        # Detect section headers
        if any(sec in line_lower for sec in RELEVANT_SKILL_SECTIONS):
            current_section = line_lower
        # Collect lines only if under relevant sections
        if any(sec in current_section for sec in RELEVANT_SKILL_SECTIONS):
            relevant_text += " " + line_lower
    return relevant_text

# ------------------ Routes ------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze_resume", methods=["POST"])
def analyze_resume():
    # Check resume upload
    if "resume" not in request.files:
        return "No resume uploaded", 400
    file = request.files["resume"]
    if file.filename == "":
        return "Empty filename", 400

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Get role selected
    role = request.form.get("role", "Software Engineer")

    # ------------------ Extract text from resume ------------------
    resume_text = extract_resume_text(filepath)
    relevant_text = extract_relevant_text(resume_text)

    # ------------------ Match skills ------------------
    ideal_skills = ROLE_SKILLS.get(role, [])
    matched_skills = [skill for skill in ideal_skills if skill.lower() in relevant_text]
    missing_skills = list(set(ideal_skills) - set(matched_skills))

    # ------------------ Check general sections ------------------
    present_sections = [sec for sec in GENERAL_SECTIONS if sec.lower() in resume_text]
    missing_sections = list(set(GENERAL_SECTIONS) - set(present_sections))

    # ATS Score: 70% skills + 30% sections
    skill_score = int((len(matched_skills) / len(ideal_skills)) * 70) if ideal_skills else 0
    section_score = int((len(present_sections) / len(GENERAL_SECTIONS)) * 30) if GENERAL_SECTIONS else 0
    ats_score = skill_score + section_score

    # Mock 2nd round probability
    second_round_prob = min(100, ats_score + 5)  # fixed small boost

    # Recommendations
    recommended_courses = ROLE_COURSES.get(role, [])
    general_feedback = []

    # Section-based feedback
    if "Professional Summary" not in present_sections:
        general_feedback.append("Add a clear professional summary at the top of your resume.")
    if "Experience" not in present_sections:
        general_feedback.append("Include your work experience with achievements and measurable impact.")
    if "Education" not in present_sections:
        general_feedback.append("Mention your educational qualifications.")
    if "Contact Info" not in present_sections:
        general_feedback.append("Add your contact information (email, phone).")
    if "Skills" not in present_sections:
        general_feedback.append("Make sure you have a dedicated 'Skills' section for clarity.")

    # General broader suggestions
    general_feedback.append("Keep your resume within 1–2 pages for better readability.")
    general_feedback.append("Use action verbs like 'developed', 'led', 'implemented' to describe achievements.")
    general_feedback.append("Quantify results wherever possible (e.g., 'Improved efficiency by 20%').")
    general_feedback.append("Tailor your resume keywords to match the specific job description.")

    # ------------------ Render Dashboard ------------------
    return render_template(
        "resume_dashboard.html",
        role=role,
        score=ats_score,
        common_skills=matched_skills,
        missing_skills=missing_skills,
        courses=recommended_courses,
        second_round_prob=second_round_prob,
        general_feedback=general_feedback
    )


@app.route("/mock_interview", methods=["POST"])
def mock_interview():
    company = request.form.get("company", "Generic Company")
    questions = [
        f"Why do you want to join {company}?",
        "Tell me about a recent project you worked on.",
        "How do you handle stress and deadlines?",
    ]
    return render_template("interview.html", company=company, questions=questions)


# ------------------ Main ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
