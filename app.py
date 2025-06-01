from flask import Flask, request, jsonify, render_template, session
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
app.secret_key = "1234567890abcdef"

# 챗봇 객체
daily_bot = RT_Daily_Chatbot()
recall_session = RT_ChatRecallSession()
theme_bot = RT_Theme_Chatbot()

# 구글 TTS, STT 클라이언트
tts_client = texttospeech.TextToSpeechClient()
stt_client = speech.SpeechClient()

@app.route("/")
def index():
    return render_template("chatbot.html")

@app.route("/start", methods=["GET"])
def start_conversation():
    user_id = request.args.get("user_id", str(uuid.uuid4()))
    session["user_id"] = user_id
    daily_bot.reset()
    first_message = daily_bot.start_conversation()
    return jsonify({"response": first_message})

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("message", "")
    user_id = session.get("user_id")
    if not user_input or not user_id:
        return jsonify({"error": "Invalid input or session"}), 400

    reply = daily_bot.ask(user_input, user_id=user_id)
    return jsonify({"response": reply})

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

@app.route("/stt", methods=["POST"])
def stt():
    file = request.files["file"]
    audio = file.read()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        language_code="ko-KR"
    )
    audio_config = speech.RecognitionAudio(content=audio)
    try:
        response = stt_client.recognize(config=config, audio=audio_config)
        transcript = "".join([r.alternatives[0].transcript for r in response.results])
        return jsonify({"transcript": transcript})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recall-session/test", methods=["GET"])
def recall_test_page():
    return render_template("recall_chatbot.html")

@app.route("/recall-session/start", methods=["POST"])
def start_recall_session():
    user_id = request.json.get("user_id")
    date = request.json.get("date", datetime.today().strftime("%Y-%m-%d"))

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    try:
        # 일기 내용 가져오기
        diary_content = recall_session.get_diary_content(date, user_id)
        if not diary_content:
            return jsonify({"error": f"No diary content found for date: {date}"}), 404

        # Document 객체를 JSON 직렬화 가능한 형태로 변환
        diary_content_serializable = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in diary_content
        ]

        # 질문 생성
        qnas = recall_session.generate_recall_questions(user_id)
        print("Generated QnAs:", qnas)  # 디버깅 로그 추가

        if not qnas:
            return jsonify({"error": "Failed to generate recall questions"}), 500

        # 질문 유형 추가
        question_types = ["시간 지남력", "장소 지남력", "기억력"]
        questions_with_types = [
    {
        "type": question_types[idx],
        "question": qa["질문"],
        "answer": qa["답변"]
    }
    for idx, qa in enumerate(qnas)
]

        # 세션 데이터 저장
        session["diary_content"] = diary_content_serializable
        session["questions"] = questions_with_types
        session["question_index"] = 0

        return jsonify({
            "questions": questions_with_types,
            "current_question": questions_with_types[0]
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    

@app.route("/recall-session/answer", methods=["POST"])
def process_recall_answer():
    user_answer = request.json.get("user_answer", "")
    question_index = session.get("question_index", 0)
    questions = session.get("questions", [])
    diary_content = session.get("diary_content", "")

    if not questions or question_index >= len(questions):
        return jsonify({"error": "No more questions available."}), 400

    current_qa = questions[question_index]
    is_correct, feedback, hint, score = recall_session.evaluate_user_answer(
        recall_question=current_qa["question"],
        recall_answer=current_qa["answer"],
        user_answer=user_answer,
        diary_content=diary_content
    )

    if is_correct:
        session["question_index"] = question_index + 1

    return jsonify({
        "is_correct": is_correct,
        "feedback": feedback,
        "hint": hint,
        "score": score,
        "next_question": questions[question_index + 1] if question_index + 1 < len(questions) else None
    })

@app.route("/theme/test", methods=["GET"])
def theme_test_page():
    return render_template("theme_chatbot.html")

@app.route("/theme/start", methods=["GET"])
def theme_start_conversation():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is missing from session"}), 400

    try:
        first_message = theme_bot.start_conversation(user_id=user_id)
        return jsonify({"response": first_message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/theme/ask", methods=["POST"])
def theme_ask():
    user_input = request.json.get("message", "")
    user_id = session.get("user_id")
    if not user_input or not user_id:
        return jsonify({"error": "Invalid input or session"}), 400

    try:
        reply = theme_bot.ask(user_input, user_id=user_id)
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/theme/select", methods=["GET"])
def theme_select():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is missing from session"}), 400

    try:
        selected_theme = theme_bot.select_theme(user_id=user_id)
        return jsonify({"selected_theme": selected_theme})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
