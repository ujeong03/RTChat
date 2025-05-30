import openai
from datetime import datetime
import os
from dotenv import load_dotenv, find_dotenv
from diary_db_management import DiaryDBManager
from typing import List, Tuple
from langchain_core.documents import Document
from datetime import datetime
import re

class RT_Daily_Chatbot:
    def __init__(self, api_key):
        self.prompt_path = "./RT_CHAT/prompt/daily_prompt_test3.txt"
        self.client = openai.OpenAI(api_key=api_key)
        self.chat_history = []
        self.db_manager = DiaryDBManager(persist_path="vectorstore/diary_faiss")

    def _load_prompt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
        
    def start_conversation(self) -> str:
         # 첫 인사 및 유도 질문
        first_message = "안녕하세요. 오늘 하루는 어땠어요? 기억에 남는 일이 있었나요?"
        self.chat_history.append({"role": "assistant", "content": first_message})
        return first_message
    
    def reset(self):
        """챗봇 상태 초기화"""
        self.history = []
        
    def ask(self, user_input: str) -> str:
        self.chat_history.append({"role": "user", "content": user_input})

        # ✅ 대화 종료 여부 판단 후 일기 저장
        if self.is_conversation_ending():
            self.chat_history.append({"role": "user", "content": "대화를 종료합니다."})

            # 💬 대화 마무리 멘트 추가
            farewell = "오늘 이야기를 들을 수 있어서 기뻤어요. 내일도 기다리고 있을게요 😊"
            self.chat_history.append({"role": "assistant", "content": farewell})

            diary_title, diary_body = self.generate_diary()
            self.save_diary(diary_title,diary_body)
            self.db_manager.create_or_update_index(
                [diary_body],
                [{
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "title": diary_title  # ⬅️ 여기 추가됨
                }]
            )
            return farewell + "\n\n(일기가 저장되었어요. 프로그램을 종료합니다.)"

        # 키워드 추출
        keywords = self.extract_keywords(user_input)

        # 키워드 통합 검색
        query = self.gpt_build_query(keywords)

        results = self.db_manager.search(keywords,query)

        recalled_diaries = []
        for i, doc in enumerate(results):
            kw = keywords[i] if i < len(keywords) else "관련된 주제"
            recalled_diaries.append((kw, doc))

        # 회상 없으면 일반 대화
        try:
            # 회상 응답 먼저 생성
            if recalled_diaries:
                recall_reply = self.generate_emotional_recall_reply(recalled_diaries)

                # 💬 회상 응답을 대화 이력에 추가
                if not '아니오' in recall_reply:
                  self.chat_history.append({"role": "assistant", "content": recall_reply})
                  return recall_reply
     
                else:
                    print("❗회상 응답이 문맥에 어울리지 않아 일반 대화로 전환")

            # 💬 회상할 내용이 없으면 일반 대화 응답 생성
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=self.chat_history,
                temperature=0.7,
                max_tokens=200
            )
            reply = response.choices[0].message.content
            self.chat_history.append({"role": "assistant", "content": reply})

            return reply
            
        except Exception as e:
            print(f"❗예외 발생: {e}")
            return "음... 지금은 대화가 조금 어려운 것 같아요. 조금 있다가 다시 얘기해볼까요?"


    def gpt_build_query(self, keywords: list[str]) -> str:
        prompt = f"다음 키워드를 기반으로 기존 일기에서 비슷한 내용이 있는지 검색할거야. FAISS 벡터 DB에서 검색할 거고, 다른 말을 덧붙이지 말고 그걸 위한 자연스러운 문장 쿼리 하나만 출력해. {', '.join(keywords)}"
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()

    # 사용자의 답변에서 회상의 키워드 추출하기
    def extract_keywords(self, text: str) -> List[str]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "다음 문장에서 중요한 키워드 3개만 뽑아줘. 장소, 사람, 사건 중심으로. 다른 말은 붙이지말고 키워드만 추출해."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=30
            )
            keywords_text = response.choices[0].message.content
            keywords = [kw.strip() for kw in keywords_text.replace('\n', ',').split(',') if kw.strip()]
            return keywords[:3]
        except:
            return []       
        

    def load_prompt(self, filepath: str, **kwargs) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        return prompt_template.format(**kwargs)
    
    
    # 회상된 일기 내용으로 감정 회상 응답 생성하기
    def generate_emotional_recall_reply(self, recalled_diaries: List[Tuple[str, Document]]) -> str:
        try:
            # 전체 일기 내용을 하나로 합치기
            combined_diary_content = "\n\n".join(
                f"[{kw}]\n{doc.page_content}" for kw, doc in recalled_diaries
            )

            # 프롬프트 로딩 및 포맷팅
            prompt = self.load_prompt(
                "./RT_CHAT/prompt/recall_prompt_test3.txt",
                chat_history=self.get_chat_history_as_text(),
                diary_content=combined_diary_content
            )

            # GPT 호출
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            generated_reply = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❗회상 생성 실패: {e}")
            generated_reply = "과거의 일기 내용을 회상하는 데 문제가 발생했어요."

        # 대화 기록 및 응답 반환
        self.chat_history.append({"role": "assistant", "content": generated_reply})
        return generated_reply

    def get_chat_history_as_text(self, limit=5) -> str:
        lines = []
        for msg in self.chat_history[-limit:]:
            role = "당신" if msg["role"] == "user" else "AI"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    # 사용자가 대화를 종료하려는 의도 확인하기
    def is_conversation_ending(self) -> bool:
        try:
            messages = [
                {"role": "system", "content": "사용자의 마지막 발화가 대화를 끝내려는 의도인지 판단해 주세요. 반드시 '예' 또는 '아니오'로만 대답해 주세요."},
                {"role": "user", "content": "고마워, 다음에 또 이야기할게."},
                {"role": "assistant", "content": "예"},
                {"role": "user", "content": "그래"},
                {"role": "assistant", "content": "예"},
                {"role": "user", "content": "오늘은 여기까지 할게."},
                {"role": "assistant", "content": "예"},
                {"role": "user", "content": "안녕히 계세요."},
                {"role": "assistant", "content": "예"},
                {"role": "user", "content": "잠깐만, 그 얘기 다시 해줘"},
                {"role": "assistant", "content": "아니오"},
                {"role": "user", "content": self.chat_history[-1]["content"]}
            ]

            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=5
            )
            answer = response.choices[0].message.content.strip().lower()
            return "예" in answer
        except Exception as e:
            print(f"[❗대화 종료 판단 실패] {e}")
            return False

    # 일기 생성하기    
    def generate_diary(self) -> Tuple[str, str]:
        summary_prompt = {
            "role": "system",
            "content": self._load_prompt("./RT_CHAT/prompt/diary_gen_prompt.txt")
        }

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[summary_prompt] + self.chat_history,
                temperature=0.7,
                max_tokens=300
            )
            diary_full_text = response.choices[0].message.content.strip()

            # ✅ 제목과 본문을 분리 (예: "제목: ~~~\n본문: ~~~")
            if "제목 :" in diary_full_text and "본문 :" in diary_full_text:
                title = diary_full_text.split("제목 :")[1].split("본문 :")[0].strip()
                body = diary_full_text.split("본문 :")[1].strip()
            else:
                title = "무제"
                body = diary_full_text

            return title, body
        except Exception as e:
            return "에러", "일기를 생성하는 데 문제가 발생했어요."


    def sanitize_filename(self, title: str) -> str:
        # 파일명에 사용할 수 없는 문자 제거 및 공백을 밑줄로 변경
        sanitized = re.sub(r'[\\/*?:"<>|]', '', title)
        sanitized = sanitized.replace(' ', '_')
        return sanitized    


    # 일기 저장
    def save_diary(self, title: str, body: str):
        today = datetime.now().strftime("%Y-%m-%d")
        save_dir = "./RT_CHAT/diary/daily_diaries"
        os.makedirs(save_dir, exist_ok=True)

        # 제목을 파일명으로 사용하되, 파일명에 적합하도록 정리
        safe_title = self.sanitize_filename(title)
        file_name = f"diary_{today}_{safe_title}.txt"
        file_path = os.path.join(save_dir, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"제목: {title}\n\n{body}")

        # 벡터 DB 업데이트
        self.db_manager.create_or_update_index(
            diary_texts=[body],
            metadata_list=[{
                "date": today,
                "title": title,
                "daily_diary" : "daily_diary"
            }]
        )


if __name__ == "__main__":
    
    _ = load_dotenv(find_dotenv())

    openai.api_key  = os.getenv("OPENAI_API_KEY")

    daily_bot = RT_Daily_Chatbot(openai.api_key)

     # 💬 챗봇이 먼저 질문
    print("🤖 챗봇:", daily_bot.start_conversation())

    while True:
        msg = input("🙋 사용자: ")
        
        reply = daily_bot.ask(msg)  # ✅ 한 번만 호출!
        print("🤖 챗봇:", reply)
        print("----- 디버깅용")

        if "프로그램을 종료합니다" in reply:
            break