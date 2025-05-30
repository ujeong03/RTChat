from flask import Flask, request, jsonify, render_template
from chat_daily import RT_Daily_Chatbot
from dotenv import load_dotenv, find_dotenv
import os
import openai

# 환경 변수 로드
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# Flask 앱 생성
app = Flask(__name__)

# 챗봇 초기화
daily_bot = RT_Daily_Chatbot(openai.api_key)

@app.route("/", methods=["GET"])
def index():
    return render_template("chatbot.html")  # templates/index.html 반환

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
