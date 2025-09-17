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
ibm_api_key = speech to text ibm api key 
ibm_url = speech to text url

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
GRANITE_API_URL = api from iam
GRANITE_API_KEY = api key 
GRANITE_PROJECT_ID = project id
GRANITE_MODEL_ID = granite model you choose

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

# ------------------ Mock Interview Question Bank ------------------
interview_data = {
    "Infosys": {
        "Software Engineer": [
            {"q": "Why do you want to join Infosys?", "keywords": ["global","innovation","career growth","digital transformation"]},
            {"q": "Explain OOP pillars.", "keywords": ["encapsulation","inheritance","polymorphism","abstraction"]},
            {"q": "How do you optimize code?", "keywords": ["time complexity","space complexity","refactor","efficiency"]},
            {"q": "Explain Agile methodology.", "keywords": ["sprint","scrum","iteration","collaboration"]},
            {"q": "How do you debug errors?", "keywords": ["logging","breakpoints","unit testing","traceback"]}
        ],
        "Data Scientist": [
            {"q": "Explain supervised vs unsupervised learning.", "keywords": ["labels","classification","clustering","regression"]},
            {"q": "What is overfitting?", "keywords": ["high variance","low bias","regularization","cross-validation"]},
            {"q": "Explain feature engineering.", "keywords": ["normalization","scaling","encoding","feature extraction"]},
            {"q": "What is PCA?", "keywords": ["dimensionality reduction","variance","components","eigenvectors"]},
            {"q": "How do you handle missing data?", "keywords": ["imputation","drop","mean","median"]}
        ],
        "AI/ML Engineer": [
            {"q": "What is the difference between CNN and RNN?", "keywords": ["image","sequential","convolution","recurrent"]},
            {"q": "Explain gradient descent.", "keywords": ["optimization","loss function","learning rate","iterations"]},
            {"q": "What is overfitting in deep learning?", "keywords": ["dropout","regularization","validation","generalization"]},
            {"q": "Explain transfer learning.", "keywords": ["pretrained","fine-tuning","layers","reuse"]},
            {"q": "What is reinforcement learning?", "keywords": ["agent","environment","reward","policy"]}
        ],
        "Full Stack Developer": [
            {"q": "What is REST API?", "keywords": ["stateless","client-server","json","endpoint"]},
            {"q": "Explain difference between SQL and NoSQL.", "keywords": ["relational","document","scalability","schema"]},
            {"q": "How do you ensure web security?", "keywords": ["encryption","authentication","csrf","xss"]},
            {"q": "Explain MVC pattern.", "keywords": ["model","view","controller","separation"]},
            {"q": "How do you handle responsive design?", "keywords": ["media query","flexbox","grid","mobile-first"]}
        ],
        "HR Manager": [
            {"q": "How do you resolve conflicts?", "keywords": ["mediation","communication","neutral","resolution"]},
            {"q": "How do you improve employee engagement?", "keywords": ["recognition","feedback","motivation","wellbeing"]},
            {"q": "What is performance appraisal?", "keywords": ["evaluation","review","feedback","goals"]},
            {"q": "How do you handle recruitment challenges?", "keywords": ["talent acquisition","screening","sourcing","retention"]},
            {"q": "What is HR analytics?", "keywords": ["data-driven","metrics","turnover","performance"]}
        ]
    },
    # ------------------ Add TCS ------------------
    "TCS": {
        "Software Engineer": [
            {"q": "Why do you want to join TCS?", "keywords": ["global leader","IT services","innovation","growth"]},
            {"q": "Explain difference between array and linked list.", "keywords": ["contiguous","dynamic","pointer","index"]},
            {"q": "What is version control?", "keywords": ["git","repository","commit","branch"]},
            {"q": "Explain SDLC models.", "keywords": ["waterfall","agile","spiral","iterative"]},
            {"q": "What is multithreading?", "keywords": ["parallel","concurrency","thread","synchronization"]}
        ],
        "Data Scientist": [
            {"q": "What is hypothesis testing?", "keywords": ["null","alternative","p-value","significance"]},
            {"q": "Explain linear regression assumptions.", "keywords": ["linearity","independence","normality","homoscedasticity"]},
            {"q": "What is confusion matrix?", "keywords": ["accuracy","precision","recall","f1-score"]},
            {"q": "What is clustering?", "keywords": ["unsupervised","groups","similarity","k-means"]},
            {"q": "How do you select features?", "keywords": ["correlation","importance","forward selection","backward elimination"]}
        ],
        "AI/ML Engineer": [
            {"q": "What is NLP?", "keywords": ["text","language","tokenization","semantic"]},
            {"q": "Explain backpropagation.", "keywords": ["gradient","error","weights","layers"]},
            {"q": "What is regularization?", "keywords": ["l1","l2","penalty","overfitting"]},
            {"q": "Explain GANs.", "keywords": ["generator","discriminator","adversarial","fake"]},
            {"q": "What is dropout?", "keywords": ["neurons","random","prevent overfitting","training"]}
        ],
        "Full Stack Developer": [
            {"q": "What is React?", "keywords": ["javascript","component","virtual dom","state"]},
            {"q": "Explain asynchronous programming.", "keywords": ["async","await","callback","promise"]},
            {"q": "How do you handle database migrations?", "keywords": ["schema","migration","update","rollback"]},
            {"q": "What is Docker?", "keywords": ["container","virtualization","isolation","deployment"]},
            {"q": "Explain CI/CD pipeline.", "keywords": ["integration","deployment","automation","testing"]}
        ],
        "HR Manager": [
            {"q": "What is employee retention?", "keywords": ["engagement","satisfaction","growth","reduce turnover"]},
            {"q": "How do you onboard new employees?", "keywords": ["orientation","training","welcome","integration"]},
            {"q": "What is competency mapping?", "keywords": ["skills","roles","assessment","gap analysis"]},
            {"q": "How do you handle layoffs?", "keywords": ["transparency","communication","support","fairness"]},
            {"q": "What is diversity and inclusion?", "keywords": ["equality","equity","inclusive","culture"]}
        ]
    },
    # ------------------ Add Wipro ------------------
    "Wipro": {
        "Software Engineer": [
            {"q": "Why do you want to join Wipro?", "keywords": ["IT services","innovation","growth","digital"]},
            {"q": "What is exception handling?", "keywords": ["try","catch","error","throw"]},
            {"q": "Explain database normalization.", "keywords": ["1nf","2nf","3nf","redundancy"]},
            {"q": "What is API testing?", "keywords": ["postman","requests","endpoint","response"]},
            {"q": "Explain cloud computing models.", "keywords": ["iaas","paas","saas","deployment"]}
        ],
        "Data Scientist": [
            {"q": "What is bias-variance tradeoff?", "keywords": ["bias","variance","overfitting","underfitting"]},
            {"q": "Explain random forest.", "keywords": ["ensemble","decision tree","bagging","prediction"]},
            {"q": "What is gradient boosting?", "keywords": ["sequential","weak learners","boosting","xgboost"]},
            {"q": "Explain time series forecasting.", "keywords": ["arima","trend","seasonality","stationarity"]},
            {"q": "How do you evaluate regression model?", "keywords": ["rmse","mae","r-squared","error"]}
        ],
        "AI/ML Engineer": [
            {"q": "What is computer vision?", "keywords": ["image","object detection","classification","opencv"]},
            {"q": "Explain word embeddings.", "keywords": ["vector","semantic","similarity","nlp"]},
            {"q": "What is attention mechanism?", "keywords": ["weights","transformer","focus","context"]},
            {"q": "Explain LSTM.", "keywords": ["sequential","memory","cell state","gates"]},
            {"q": "What is autoencoder?", "keywords": ["encoder","decoder","latent","representation"]}
        ],
        "Full Stack Developer": [
            {"q": "What is Node.js?", "keywords": ["javascript","runtime","server-side","event-driven"]},
            {"q": "Explain authentication vs authorization.", "keywords": ["identity","access","permission","token"]},
            {"q": "What is GraphQL?", "keywords": ["query","api","schema","flexible"]},
            {"q": "How do you optimize website performance?", "keywords": ["cache","minify","lazy loading","cdn"]},
            {"q": "Explain microservices architecture.", "keywords": ["services","independent","api","scalable"]}
        ],
        "HR Manager": [
            {"q": "What is talent management?", "keywords": ["acquisition","development","retention","growth"]},
            {"q": "How do you manage payroll?", "keywords": ["salary","compliance","accuracy","system"]},
            {"q": "Explain grievance handling.", "keywords": ["complaint","resolution","communication","trust"]},
            {"q": "What is succession planning?", "keywords": ["future","leadership","replacement","strategy"]},
            {"q": "How do you measure training effectiveness?", "keywords": ["feedback","performance","evaluation","improvement"]}
        ]
    },
    # ------------------ Add HCL ------------------
    "HCL": {
        "Software Engineer": [
            {"q": "Why do you want to join HCL?", "keywords": ["IT services","innovation","career growth","client-focused"]},
            {"q": "Explain difference between stack and queue.", "keywords": ["fifo","lifo","order","structure"]},
            {"q": "What is exception handling in Java?", "keywords": ["try","catch","finally","throw"]},
            {"q": "Explain SDLC phases.", "keywords": ["planning","design","implementation","testing"]},
            {"q": "What is polymorphism?", "keywords": ["overloading","overriding","oop","inheritance"]}
        ],
        "Data Scientist": [
            {"q": "What is logistic regression?", "keywords": ["binary","classification","sigmoid","probability"]},
            {"q": "Explain decision trees.", "keywords": ["split","gini","entropy","classification"]},
            {"q": "What is k-means?", "keywords": ["clustering","centroid","unsupervised","distance"]},
            {"q": "Explain cross-validation.", "keywords": ["k-fold","train","test","validation"]},
            {"q": "What is feature scaling?", "keywords": ["normalization","standardization","min-max","z-score"]}
        ],
        "AI/ML Engineer": [
            {"q": "Explain CNN architecture.", "keywords": ["convolution","pooling","filters","layers"]},
            {"q": "What is reinforcement learning?", "keywords": ["reward","policy","agent","environment"]},
            {"q": "Explain transformers.", "keywords": ["attention","sequence","nlp","parallel"]},
            {"q": "What is unsupervised learning?", "keywords": ["clustering","grouping","patterns","data"]},
            {"q": "Explain model deployment.", "keywords": ["api","docker","flask","scalability"]}
        ],
        "Full Stack Developer": [
            {"q": "What is middleware?", "keywords": ["request","response","processing","server"]},
            {"q": "Explain REST vs SOAP.", "keywords": ["http","xml","json","protocol"]},
            {"q": "How do you handle sessions?", "keywords": ["cookies","token","storage","authentication"]},
            {"q": "What is ORM?", "keywords": ["object","relational","mapping","database"]},
            {"q": "Explain version control workflow.", "keywords": ["branch","merge","pull request","commit"]}
        ],
        "HR Manager": [
            {"q": "How do you improve workplace culture?", "keywords": ["values","engagement","diversity","trust"]},
            {"q": "What is compensation management?", "keywords": ["salary","benefits","equity","packages"]},
            {"q": "Explain training needs analysis.", "keywords": ["skills","gap","assessment","development"]},
            {"q": "What is employee lifecycle?", "keywords": ["recruitment","onboarding","development","exit"]},
            {"q": "How do you ensure compliance in HR?", "keywords": ["laws","regulations","policies","audit"]}
        ]
    },
    # ------------------ Add Tech Mahindra ------------------
    "Tech Mahindra": {
        "Software Engineer": [
            {"q": "Why do you want to join Tech Mahindra?", "keywords": ["telecom","innovation","digital","growth"]},
            {"q": "Explain linked list types.", "keywords": ["singly","doubly","circular","node"]},
            {"q": "What is inheritance in OOP?", "keywords": ["base","derived","reuse","hierarchy"]},
            {"q": "Explain cloud deployment models.", "keywords": ["public","private","hybrid","community"]},
            {"q": "What is exception propagation?", "keywords": ["method","stack","error","handling"]}
        ],
        "Data Scientist": [
            {"q": "What is neural network?", "keywords": ["neurons","layers","activation","weights"]},
            {"q": "Explain ensemble learning.", "keywords": ["bagging","boosting","voting","stacking"]},
            {"q": "What is dimensionality reduction?", "keywords": ["pca","variance","features","compression"]},
            {"q": "Explain sampling techniques.", "keywords": ["random","stratified","systematic","bias"]},
            {"q": "How do you deal with imbalanced data?", "keywords": ["oversampling","undersampling","smote","balanced"]}
        ],
        "AI/ML Engineer": [
            {"q": "What is supervised learning?", "keywords": ["input","output","labels","training"]},
            {"q": "Explain unsupervised learning.", "keywords": ["clustering","patterns","unlabeled","exploration"]},
            {"q": "What is reinforcement learning?", "keywords": ["reward","policy","agent","environment"]},
            {"q": "Explain overfitting and underfitting.", "keywords": ["variance","bias","generalization","model"]},
            {"q": "What is hyperparameter tuning?", "keywords": ["grid search","random search","parameters","optimization"]}
        ],
        "Full Stack Developer": [
            {"q": "What is REST API?", "keywords": ["endpoint","http","client-server","stateless"]},
            {"q": "Explain session vs token.", "keywords": ["authentication","authorization","storage","security"]},
            {"q": "What is webpack?", "keywords": ["bundler","modules","javascript","assets"]},
            {"q": "Explain responsive design.", "keywords": ["media queries","flexbox","grid","adaptive"]},
            {"q": "How do you secure web apps?", "keywords": ["csrf","xss","encryption","firewall"]}
        ],
        "HR Manager": [
            {"q": "What is employee engagement?", "keywords": ["motivation","participation","satisfaction","involvement"]},
            {"q": "Explain recruitment process.", "keywords": ["job posting","screening","interview","selection"]},
            {"q": "What is performance management?", "keywords": ["goals","review","feedback","evaluation"]},
            {"q": "How do you handle attrition?", "keywords": ["retention","strategy","culture","growth"]},
            {"q": "What is HR technology?", "keywords": ["automation","tools","hrms","analytics"]}
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
