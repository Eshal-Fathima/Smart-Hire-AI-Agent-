from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Smart Hire AI Agent!"

if __name__ == "__main__":
    app.run(debug=True)

from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os

authenticator = IAMAuthenticator(os.getenv("IBM_API_KEY"))
speech_to_text = SpeechToTextV1(authenticator=authenticator)
speech_to_text.set_service_url(os.getenv("IBM_URL"))