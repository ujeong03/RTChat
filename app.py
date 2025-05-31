from flask import Flask, request, jsonify, render_template
from chat_daily import RT_Daily_Chatbot
from chat_reall_sess import RT_ChatRecallSession
from chat_theme import RT_Theme_Chatbot
from dotenv import load_dotenv, find_dotenv
from google.cloud import texttospeech, speech
import os
import openai
import uuid
from datetime import datetime

# 환경 변수 로드
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

app = Flask(__name__)
daily_bot = RT_Daily_Chatbot()
recall_session = RT_ChatRecallSession()
theme_bot = RT_Theme_Chatbot()
tts_client = texttospeech.TextToSpeechClient()
stt_client = speech.SpeechClient()

@app.route("/")
def index():
    return render_template("chatbot.html")

@app.route("/start", methods=["GET"])
def start_conversation():
    daily_bot.reset()
    first_message = daily_bot.start_conversation()
    return jsonify({"response": first_message})

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "message is required"}), 400

    reply = daily_bot.ask(user_input)
    return jsonify({"response": reply})

@app.route("/stt", methods=["POST"])
def stt():
    file = request.files["file"]
    audio = file.read()

    client = speech.SpeechClient()

    audio_config = speech.RecognitionAudio(content=audio)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        language_code="ko-KR"
    )

    try:
        response = client.recognize(config=config, audio=audio_config)

        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript

        return jsonify({"transcript": transcript})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tts", methods=["POST"])
def generate_tts():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"error": "text is required"}), 400

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        name="ko-KR-Chirp3-HD-Achernar",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    filename = f"static/audio_{uuid.uuid4().hex}.mp3"
    with open(filename, "wb") as out:
        out.write(response.audio_content)

    return jsonify({"audio_url": "/" + filename})

@app.route("/recall-session", methods=["POST"])
def recall_session_api():
    date = request.json.get("date", datetime.today().strftime("%Y-%m-%d"))
    try:
        diary_content = recall_session.get_diary_content(date)
        if not diary_content:
            return jsonify({"error": f"No diary content found for date: {date}"}), 404

        qnas = recall_session.generate_recall_questions()
        return jsonify({"questions": qnas})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route("/theme/start", methods=["GET"])
def theme_start_conversation():
    try:
        first_message = theme_bot.start_conversation()
        return jsonify({"response": first_message})
    except Exception as e:
        return jsonify({"error": f"Failed to start theme conversation: {str(e)}"}), 500

@app.route("/theme/ask", methods=["POST"])
def theme_ask():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "message is required"}), 400

    try:
        reply = theme_bot.ask(user_input)
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"error": f"Failed to process theme conversation: {str(e)}"}), 500

@app.route("/theme/select", methods=["GET"])
def theme_select():
    try:
        selected_theme = theme_bot.select_theme()
        return jsonify({"selected_theme": selected_theme})
    except Exception as e:
        return jsonify({"error": f"Failed to select theme: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)