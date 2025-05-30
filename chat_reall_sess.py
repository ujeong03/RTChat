import os
import openai
from datetime import datetime
from dotenv import load_dotenv
from diary_db_management import DiaryDBManager
# .env에서 API 키 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class RT_ChatRecallSession:
    def __init__(self):
        self.quiz_prompt_path = "./RT_CHAT/prompt/recall_sess_quiz_gen_prompt_1.txt"
        self.assist_prompt_path = "./RT_CHAT/prompt/recall_sess_assistant_prompt_1.txt"
        self.db_manager = DiaryDBManager(persist_path="vectorstore/diary_faiss")  # 가정: DiaryDBManager 클래스가 정의되어 있음
        self.chat_history = []  # 대화 기록 저장용

    def load_prompt(self, path: str, **kwargs) -> str:
        with open(path, "r", encoding="utf-8") as f:
            template = f.read()
        return template.format(**kwargs)
    
    def get_diary_content(self, date: str):
        """ 최근 일주일의 일기 작성 가져오기"""
        diary = self.db_manager.get_diary_7days_by_date(date)
        if diary:
            return diary
        else:
            raise ValueError(f"해당 날짜의 일기가 없습니다: {date}")


    # 일기 내용 기반으로 회상 질문 생성하기
    def generate_recall_questions(self):
        today = datetime.today().strftime("%Y-%m-%d")
        diary_contents = self.get_diary_content(today)

        prompt = self.load_prompt(self.quiz_prompt_path, date=today, diary_content=diary_contents)

        response = openai.ChatCompletion.create(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
        )

        reply = response.choices[0].message.content.strip()
        self.chat_history.append({"role": "assistant", "content": reply})
        return reply

    # 사용자 답변 평가하기
    def evaluate_user_answer(self, recall_question, recall_answer, user_answer, diary_content):
        prompt = self.load_prompt(
            self.assist_prompt_path,
            recall_question=recall_question,
            recall_answer=recall_answer,
            user_answer=user_answer,
            diary_content=diary_content
        )

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
        )

        content = response.choices[0].message.content.strip()
        if "정답" in content or "맞았어요" in content:
            return True, content, None
        elif "힌트" in content:
            parts = content.split("힌트:")
            return False, parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""
        else:
            return False, content, ""

    def run_session(self, diary_content: str):
        qnas = self.generate_recall_questions()

        for idx, recall_type in enumerate(["시간 지남력", "장소 지남력", "기억력"]):
            qa = qnas[idx] if isinstance(qnas, list) else qnas  # GPT 응답 구조에 따라
            print(f"\n🧩 질문 {idx+1} ({recall_type}): {qa['질문']}")
            user_answer = input("👉 당신의 답변: ")

            is_correct, feedback, hint = self.evaluate_user_answer(
                qa["질문"], qa["답변"], user_answer, diary_content
            )
            print("✅ 평가 결과:", feedback)
            if not is_correct and hint:
                print("💡 힌트:", hint)

if __name__ == "__main__":
    session = RT_ChatRecallSession()
    today = datetime.today().strftime("%Y-%m-%d")
    diary_content = session.get_diary_content(today)
    
    # 일기 내용이 없으면 종료
    if not diary_content:
        print(f"해당 날짜({today})의 일기가 없습니다.")
    else:
        session.run_session(diary_content)  # 세션 실행