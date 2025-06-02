import os
import json
import openai
from datetime import datetime
from dotenv import load_dotenv
from diary_db_management import DiaryDBManager
# .env에서 API 키 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class RT_ChatRecallSession:
    def __init__(self):
        self.quiz_prompt_path = "./prompt/recall_sess_quiz_gen_prompt_3.txt"
        self.assist_prompt_path = "./prompt/recall_sess_assistant_prompt_3.txt"
        self.db_manager = DiaryDBManager(persist_path="vectorstore/diary_faiss")  # 가정: DiaryDBManager 클래스가 정의되어 있음
        self.chat_history = []  # 대화 기록 저장용
        self.client = openai.OpenAI(api_key=openai.api_key)

    def load_prompt(self, path: str, **kwargs) -> str:
        with open(path, "r", encoding="utf-8") as f:
            template = f.read()
        return template.format(**kwargs)
    
    def get_diary_content(self, date: str, user_id: str):
        """ 최근 일주일의 일기 작성 가져오기"""
        diary = self.db_manager.get_diary_7days_by_date(user_id,date)
        if diary:
            return diary
        else:
            raise ValueError(f"해당 날짜의 일기가 없습니다: {date}")


    # 일기 내용 기반으로 회상 질문 생성하기

    def generate_recall_questions(self, user_id: str):  
        today = datetime.today().strftime("%Y-%m-%d")
        diary_contents = self.get_diary_content(today, user_id)

        prompt = self.load_prompt(self.quiz_prompt_path, date=today, diary_content=diary_contents)
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
        )

        reply = response.choices[0].message.content.strip()
        # self.chat_history.append({"role": "assistant", "content": reply})

        # JSON 파싱 시도
        try:
            parsed = json.loads(reply)
            return parsed  # list[dict]
        except json.JSONDecodeError as e:
            print("❌ GPT 응답 파싱 실패:", e)
            print("📝 원본 응답:", reply)
            return []  # 빈 리스트 반환하여 이후 코드에서 예외 처리 가능하게

    def evaluate_user_answer(self, recall_question, recall_answer, user_answer, diary_content):
        system_instruction = """
            당신은 심리 회상 퀴즈 전문가입니다.
            사용자의 답변이 의미적으로 정답(70% 이상 의미 유사)이라면 '정답'이라고 판단하고 수치화한 후,
            공감 어린 피드백을 제공합니다.

            틀렸다고 판단되면 '힌트'를 주되, 정답을 직접 말하지 말고, 일기 내용을 바탕으로 간접적인 묘사나 단서만 줍니다.

            반드시 아래 JSON 형식으로 출력하세요:

                {{
                "status": "정답" 또는 "힌트",
                "score": 0.0,  # 의미 유사도 점수 (0.0 ~ 100.0)
                "feedback": "공감 피드백 또는 유도 질문",
                "hint": "힌트 텍스트 (정답이면 빈 문자열)"
                }}
        """
        # 프롬프트 로드
        prompt = self.load_prompt(
            self.assist_prompt_path,
            recall_question=recall_question,
            recall_answer=recall_answer,
            user_answer=user_answer,
            diary_content=diary_content
        )

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )

        content = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(content)
            return parsed["status"] == "정답", parsed["feedback"], parsed.get("hint", ""), parsed["score"]
        except Exception as e:
            print("❌ JSON 파싱 실패:", e)
            print("📝 원본 응답:", content)
            return False, content, ""
            
    def run_session(self, diary_content: str, user_id: str):
        qnas = self.generate_recall_questions(user_id)

        for idx, recall_type in enumerate(["시간 지남력", "장소 지남력", "기억력"]):
            qa = qnas[idx]
            print(f"\n🧩 질문 {idx+1} ({recall_type}): {qa['질문']}")

            attempts = 0
            is_correct = False
            user_answer = ""

            while not is_correct and attempts < 5:  # 최대 5회까지 유도
                user_answer = input("👉 당신의 답변: ")
                self.chat_history.append({"role": "user", "content": user_answer})
                is_correct, feedback, hint,score = self.evaluate_user_answer(
                    qa["질문"], qa["답변"], user_answer, diary_content
                )

                print("✅ 평가 결과:", feedback)
                print("점수:", score)  # 점수는 hint로 전달됨
                if not is_correct and hint:
                    print("💡 힌트:", hint)
                    self.chat_history.append({"role": "assistant", "content": hint})

                attempts += 1

            if not is_correct:
                print("❌ 여러 번 시도했지만 정답을 기억해내지 못했어요.")


if __name__ == "__main__":
    # 로컬 테스트용 user_id 생성
    user_id = "test_user"

    # RT_ChatRecallSession 인스턴스 생성
    session = RT_ChatRecallSession()

    # 오늘 날짜 가져오기
    today = datetime.today().strftime("%Y-%m-%d")

    try:
        # 일기 내용 가져오기
        diary_content = session.get_diary_content(today, user_id)

        # 일기 내용이 없으면 종료
        if not diary_content:
            print(f"해당 날짜({today})의 일기가 없습니다.")
        else:
            # 회상 세션 실행
            session.run_session(diary_content, user_id)

            # 대화 기록 출력
            print("\n--- 대화 기록 ---")
            for entry in session.chat_history:
                print(f"{entry['role']}: {entry['content']}")
    except ValueError as e:
        print(f"❌ 오류: {e}")
    except Exception as e:
        print(f"❌ 예기치 못한 오류 발생: {e}")