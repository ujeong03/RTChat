import traceback
import openai
from datetime import datetime
import os
from dotenv import load_dotenv, find_dotenv
from diary_db_management import DiaryDBManager
from typing import List, Tuple
from langchain_core.documents import Document
from datetime import datetime
import re
from collections import Counter


class RT_Theme_Chatbot:
    def __init__(self, api_key, prompt_path):
        self.client = openai.OpenAI(api_key=api_key)
        self.system_prompt = {
            "role": "system",
            "content": self._load_prompt(prompt_path)
        }
        self.chat_history = [self.system_prompt]
        self.db_manager = DiaryDBManager(persist_path="vectorstore/diary_faiss")

    def _load_prompt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
        
    def select_theme(self) -> str:
        try:
            docs = self.db_manager.search_all_diaries()
            themes = []
            for doc in docs:
                theme = doc.metadata.get("theme")
                if theme and theme.strip():  # None이나 빈 문자열 아닌 경우
                    themes.append(theme.strip())
                else:
                    themes.append("기타")
            theme_count = Counter(themes)
    
            theme_count_str = "\n".join([f"{k}: {v}" for k, v in theme_count.items()])
            prompt = self.load_prompt(
                "./RT_CHAT/prompt/theme_prompt_test2.txt",
                theme_count=theme_count_str
            )
       
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"system","content":prompt}],
                temperature=0.7,
                max_tokens=200
            )

            response_text = response.choices[0].message.content.strip()
            match = re.search(r'\d+', response_text)

            if match:
                selected_theme_num = int(match.group())
            else:
                # fallback 처리
                selected_theme_num = None
            selected_theme_num = response.choices[0].message.content.strip()


            self.chat_history.append({"role": "assistant", "content": selected_theme_num})
            return selected_theme_num

        except Exception as e:
            print(f"❗첫 대화 메시지 생성 실패: {e}")
            traceback.print_exc()
            return '지금은 대화가 어려워요.'
            
        
    
    def change_theme(self, selected_theme_num:str) -> str:
        try:
            # 테마 변경 프롬프트 로드
            theme_list = [ "school", "hobby", "growth", "family", "event", "work", "food", "military", "holiday"]
            selected_theme_num = selected_theme_num
            theme_name = theme_list[int(selected_theme_num)-1]

            prompt = self.load_prompt(
                f"./RT_CHAT/prompt/theme_prompt_{selected_theme_num}_{theme_name}.txt",
            )
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini" 
                ,  
                messages=[
                    {"role": "system", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            generated_reply = response.choices[0].message.content.strip()
            self.chat_history.append({"role": "assistant", "content": generated_reply})
            return generated_reply
        except Exception as e:      
            print(f"❗테마 변경 실패: {e}")
            return "테마 변경에 실패했습니다. 다시 시도해 주세요."
        
    def start_conversation(self) -> str:
        try:
            selected_theme_num = self.select_theme()
            selected_theme_bot = self.change_theme(selected_theme_num)
            self.chat_history.append({"role": "assistant", "content": selected_theme_bot})
            return selected_theme_bot
        except Exception as e:  
            print(f"❗대화 시작 실패: {e}")
            return "대화를 시작하는 데 문제가 발생했습니다. 다시 시도해 주세요."

    
    def ask(self, user_input: str) -> str:
        self.chat_history.append({"role": "user", "content": user_input})

        # 🔍 키워드 추출
        keywords = self.extract_keywords(user_input)
        print(f"🔍 추출된 키워드: {keywords}")

        # 🔍 키워드 통합 검색: 키워드 전체를 하나의 query로 처리
        query = " ".join(keywords)
        results = self.db_manager.search(query, top_k=3)

        # 🔄 결과 출력
        recalled_diaries = []
        print(f"🔍 회상 통합 쿼리: {query}")
        for i, doc in enumerate(results):
            print(f"🔍 회상 결과 {i+1}: {doc.page_content[:40]}...")
            recalled_diaries.append((query, doc))


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

            # ✅ 대화 종료 여부 판단 후 일기 저장
            if self.is_conversation_ending():
                self.chat_history.append({"role": "user", "content": "대화를 종료합니다."})

                # 💬 대화 마무리 멘트 추가
                farewell = "오늘 이야기를 들을 수 있어서 기뻤어요. 내일도 기다리고 있을게요 😊"
                self.chat_history.append({"role": "assistant", "content": farewell})

                diary_title, diary_theme, diary_body = self.generate_diary()
                self.save_diary(diary_title,diary_body)
                self.db_manager.create_or_update_index(
                    [diary_body],
                    [{
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "title": diary_title  # ⬅️ 여기 추가됨
                        ,"theme" : diary_theme
                    }]
                )
                return farewell + "\n\n(일기가 저장되었어요. 프로그램을 종료합니다.)"

            return reply
            
        except Exception as e:
            print(f"❗예외 발생: {e}")
            return "음... 지금은 대화가 조금 어려운 것 같아요. 조금 있다가 다시 얘기해볼까요?"

    # 사용자의 답변에서 회상의 키워드 추출하기
    def extract_keywords(self, text: str) -> List[str]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "다음 문장에서 중요한 키워드 3개만 뽑아줘. 장소, 사람, 사건 중심으로."},
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
        
    def load_prompt(self, path: str, **kwargs) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                template = f.read()
        except Exception as e:
            raise RuntimeError(f"프롬프트 파일을 불러오는 데 실패했습니다: {e}")

        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            if placeholder in template:
                template = template.replace(placeholder, value)
            else:
                print(f"⚠️ 경고: '{key}'는 템플릿에 존재하지 않는 placeholder입니다.")

        return template
    
    # 회상된 일기 내용으로 감정 회상 응답 생성하기
    def generate_emotional_recall_reply(self, recalled_diaries: List[Tuple[str, Document]]) -> str:
        recall_lines = []

        for kw, doc in recalled_diaries:
            try:
                prompt = self.load_prompt(
                    "./RT_CHAT/prompt/recall_prompt_test1.txt",
                    chat_history=self.get_chat_history_as_text(),
                    diary_content=doc.page_content
                )
                response = self.client.chat.completions.create(
                    model="gpt-4.1-mini" 
                    ,  
                    messages=[
                        {"role": "system", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=300
                )
                generated_reply = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"❗회상 생성 실패: {e}")
                generated_reply = f"'{kw}'에 대해 이야기하셨던 부분이 있어요. ({doc.page_content[:50]}...)"

            recall_lines.append(generated_reply)

        reply = "\n".join(recall_lines)
        self.chat_history.append({"role": "assistant", "content": reply})
        return reply

    def get_chat_history_as_text(self, limit=5) -> str:
        lines = []
        for msg in self.chat_history[-limit:]:
            role = "당신" if msg["role"] == "user" else "AI"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    # 사용자가 대화를 종료하려는 의도 확인하기
    def is_conversation_ending(self) -> bool:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "다음 대화에서 사용자가 대화를 끝내려고 하는 의도가 있는지를 파악해줘. '예' 또는 '아니오'만 대답해."},
                    {"role": "user", "content": self.chat_history[-1]["content"]},
                ],
                temperature=0.0,
                max_tokens=10
            )
            answer = response.choices[0].message.content.strip()
            
            return "예" in answer
        except Exception as e:
            print(f"[❗대화 종료 판단 실패] {e}")
            return False
        
    # 일기 생성하기    
    def generate_diary(self) -> Tuple[str, str,str]:
        summary_prompt = {
            "role": "system",
            "content": self._load_prompt("./RT_CHAT/prompt/diary_gen_prompt_test2.txt")
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
                theme = diary_full_text.split("주제 :")[1].split("본문 :")[1].strip()
                body = diary_full_text.split("본문 :")[1].strip()
            else:
                title = "무제"
                theme = "기타"
                body = diary_full_text

            return title, theme, body
        except Exception as e:
            return "에러", "일기를 생성하는 데 문제가 발생했어요."

    def sanitize_filename(self, title: str) -> str:
        # 파일명에 사용할 수 없는 문자 제거 및 공백을 밑줄로 변경
        sanitized = re.sub(r'[\\/*?:"<>|]', '', title)
        sanitized = sanitized.replace(' ', '_')
        return sanitized    

    # 일기 저장
    def save_diary(self, title: str, body: str, theme:str):
        today = datetime.now().strftime("%Y-%m-%d")
        save_dir = "./RT_CHAT/diary/theme_diaries"
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
                "theme" : theme
            }]
        )


if __name__ == "__main__":
    
    _ = load_dotenv(find_dotenv())

    openai.api_key  = os.getenv("OPENAI_API_KEY")

    prompt_path = os.path.join("./RT_CHAT/prompt", "theme_prompt_test2.txt")

    theme_bot = RT_Theme_Chatbot(openai.api_key, prompt_path)

     # 💬 챗봇이 먼저 질문
    print("🤖 챗봇:", theme_bot.start_conversation())

    while True:
        msg = input("🙋 사용자: ")
        
        reply = theme_bot.ask(msg)  # ✅ 한 번만 호출!
        print("🤖 챗봇:", reply)
        print("----- 디버깅용")

        if "프로그램을 종료합니다" in reply:
            break