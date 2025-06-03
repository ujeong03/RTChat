from flask import Flask, request, jsonify, render_template
from chat_daily import RT_Daily_Chatbot
from chat_reall_sess import RT_ChatRecallSession
from chat_theme import RT_Theme_Chatbot
from dotenv import load_dotenv, find_dotenv
from google.cloud import texttospeech, speech
import os
import openai
import uuid
from datetime import datetime, timedelta, timezone
from flask_cors import CORS
import jwt

# 환경 변수 로드
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# JWT 설정
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "your_jwt_secret_key")  # 환경 변수에서 로드하거나 기본값 설정
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 3600  # 토큰 유효 시간 (1시간)

def generate_jwt(user_id):
    """
    JWT 토큰 생성 함수
    :param user_id: 사용자 ID
    :return: JWT 토큰
    """
    # 현재 UTC 시간 계산
    current_time = datetime.now(timezone.utc)
    
    # 토큰에 포함될 페이로드 정의
    payload = {
        "user_id": user_id,  # 사용자 ID
        "iat": current_time,  # 토큰 생성 시간
        "exp": current_time + timedelta(seconds=JWT_EXP_DELTA_SECONDS)  # 토큰 만료 시간
    }
    # JWT 토큰 생성
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def get_jwt_payload():
    """
    Authorization 헤더에서 JWT 토큰을 추출하고 검증하여 페이로드 반환
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None, jsonify({"error": "Authorization token is required"}), 401

    token = auth_header.split(" ")[1]  # 'Bearer <token>'에서 <token>만 추출
    payload = verify_jwt(token)
    if not payload:
        return None, jsonify({"error": "Invalid or expired token"}), 401

    return payload, None, None

def verify_jwt(token):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None  # 토큰이 만료됨
    except jwt.InvalidTokenError:
        return None  # 토큰이 유효하지 않음
    
app = Flask(__name__)
CORS(app,
     supports_credentials=True,
     origins=[
         "http://localhost:5174",
         "https://localhost:5174",
         "https://nabiya.site"
     ])


app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key")  # 기본값 설정 가능

# 챗봇 객체
daily_bot = RT_Daily_Chatbot()
recall_session = RT_ChatRecallSession()
theme_bot = RT_Theme_Chatbot()

# 구글 TTS, STT 클라이언트
tts_client = texttospeech.TextToSpeechClient()
stt_client = speech.SpeechClient()

@app.route("/auth/token", methods=["POST"])
def get_token():
    user_id = request.json.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    token = generate_jwt(user_id)
    return jsonify({"token": token})


@app.route("/")
def index():
    return render_template("chatbot.html")

@app.route("/start", methods=["GET"])
def start_conversation():
    payload, error_response, status_code = get_jwt_payload()
    if error_response:
        return error_response, status_code

    user_id = payload["user_id"]

    daily_bot.reset()
    first_message = daily_bot.start_conversation()
    return jsonify({"response": first_message})

@app.route("/ask", methods=["POST"])
def ask():
    payload, error_response, status_code = get_jwt_payload()
    if error_response:
        return error_response, status_code

    user_id = payload["user_id"]
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "message is required"}), 400

    try:
        result = daily_bot.ask(user_input, user_id=user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# tts
@app.route("/tts", methods=["POST"])
def generate_tts():
    payload, error_response, status_code = get_jwt_payload()
    if error_response:
        return error_response, status_code

    user_id = payload["user_id"]
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

    return jsonify({"audio_url": "/" + filename, "filename": filename})

# tts 재생 후 삭제
@app.route("/tts/delete", methods=["POST"])
def delete_tts():
    payload, error_response, status_code = get_jwt_payload()
    if error_response:
        return error_response, status_code

    user_id = payload["user_id"]
    filename = request.json.get("filename", "")
    if not filename:
        return jsonify({"error": "filename is required"}), 400

    try:
        os.remove(filename)
        return jsonify({"message": "File deleted successfully"})
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/stt", methods=["POST"])
def stt():
    payload, error_response, status_code = get_jwt_payload()
    if error_response:
        return error_response, status_code

    user_id = payload["user_id"]
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
    payload, error_response, status_code = get_jwt_payload()
    if error_response:
        return error_response, status_code

    user_id = payload["user_id"]
    date = request.json.get("date", datetime.today().strftime("%Y-%m-%d"))

    try:
        diary_content = recall_session.get_diary_content(date, user_id)
        if not diary_content:
            return jsonify({"error": f"No diary content found for date: {date}"}), 404

        diary_content_serializable = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in diary_content
        ]

        qnas = recall_session.generate_recall_questions(user_id)
        if not qnas:
            return jsonify({"error": "Failed to generate recall questions"}), 500

        question_types = ["시간 지남력", "장소 지남력", "기억력"]
        questions_with_types = [
            {"type": question_types[idx], "question": qa["질문"], "answer": qa["답변"]}
            for idx, qa in enumerate(qnas)
        ]

        return jsonify({
            "questions": questions_with_types,
            "current_question": questions_with_types[0],
            "diary_content": diary_content_serializable
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/recall-session/answer", methods=["POST"])
def process_recall_answer():
    payload, error_response, status_code = get_jwt_payload()
    if error_response:
        return error_response, status_code

    user_id = payload["user_id"]
    print("Request data:", request.json)  # 요청 데이터 출력

    user_answer = request.json.get("user_answer", "")
    question_index = request.json.get("question_index", 0)
    questions = request.json.get("questions", [])  # 수정: "question" -> "questions"
    diary_content = request.json.get("diary_content", "")

    if not questions or question_index >= len(questions):
        return jsonify({"error": "No more questions available."}), 400

    current_qa = questions[question_index]
    is_correct, feedback, hint, score = recall_session.evaluate_user_answer(
        recall_question=current_qa["question"],
        recall_answer=current_qa["answer"],
        user_answer=user_answer,
        diary_content=diary_content
    )

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
    payload, error_response, status_code = get_jwt_payload()
    if error_response:
        return error_response, status_code

    user_id = payload["user_id"]

    try:
        first_message = theme_bot.start_conversation(user_id=user_id)
        return jsonify({"response": first_message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/theme/ask", methods=["POST"])
def theme_ask():
    payload, error_response, status_code = get_jwt_payload()
    if error_response:
        return error_response, status_code

    user_id = payload["user_id"]
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "message is required"}), 400

    try:
        result = theme_bot.ask(user_input, user_id=user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/theme/select", methods=["GET"])
def theme_select():
    payload, error_response, status_code = get_jwt_payload()
    if error_response:
        return error_response, status_code

    user_id = payload["user_id"]

    try:
        selected_theme = theme_bot.select_theme(user_id=user_id)
        return jsonify({"selected_theme": selected_theme})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)