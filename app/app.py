from flask import Flask, request, render_template
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os

app = Flask(__name__)

# ------------------ Routes ------------------ #
@app.route("/")
def home():
    return "Hello, Smart Hire AI Agent!"

# Example route to test IBM Watson connection
@app.route("/test-watson")
def test_watson():
    if not os.getenv("IBM_API_KEY") or not os.getenv("IBM_URL"):
        return "IBM credentials not set!"
    return "IBM Watson Speech-to-Text is configured!"

# ------------------ IBM Watson Setup ------------------ #
# Only initialize if API key and URL exist
ibm_api_key = os.getenv("IBM_API_KEY")
ibm_url = os.getenv("IBM_URL")

if ibm_api_key and ibm_url:
    authenticator = IAMAuthenticator(ibm_api_key)
    speech_to_text = SpeechToTextV1(authenticator=authenticator)
    speech_to_text.set_service_url(ibm_url)
else:
    print("⚠️ IBM Watson credentials are missing!")

# ------------------ Run App ------------------ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT env var
    app.run(host="0.0.0.0", port=port, debug=True)
