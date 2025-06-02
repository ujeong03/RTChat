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

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class RT_Theme_Chatbot:
    def __init__(self):
        self.prompt_path = './prompt/theme_prompt_test2.txt'
        self.client = openai.OpenAI(api_key=openai.api_key)
        self.chat_history = []
        self.db_manager = DiaryDBManager(persist_path="vectorstore/diary_faiss")

    def _load_prompt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
        
    def reset(self):
        """챗봇 상태 초기화"""
        self.history = []
        
    def select_theme(self, user_id: str) -> str:
        try:
            docs = self.db_manager.search_all_diaries(user_id)
            themes = []
            for doc in docs:
                theme = doc.metadata.get("theme")
                if theme and theme.strip():  # None이나 빈 문자열 아닌 경우
                    themes.append(theme.strip())

            theme_count = Counter(themes)
    
            theme_count_str = "\n".join([f"{k}: {v}" for k, v in theme_count.items()])
            prompt = self.load_prompt(
                "./prompt/theme_prompt_test2.txt",
                theme_count=theme_count_str
            )
       
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
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
                f"./prompt/theme_prompt_{selected_theme_num}_{theme_name}.txt",
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
        
    def start_conversation(self,user_id:str) -> str:
        try:
            selected_theme_num = self.select_theme(user_id)
            selected_theme_bot = self.change_theme(selected_theme_num)
            self.chat_history.append({"role": "assistant", "content": selected_theme_bot})
            return selected_theme_bot
        except Exception as e:  
            print(f"❗대화 시작 실패: {e}")
            return "대화를 시작하는 데 문제가 발생했습니다. 다시 시도해 주세요."

    
    def ask(self, user_input: str, user_id: str) -> str:
        self.chat_history.append({"role": "user", "content": user_input})

        # ✅ 대화 종료 여부 판단 후 일기 저장
        if self.is_conversation_ending():
            self.chat_history.append({"role": "user", "content": "대화를 종료합니다."})

            # 💬 대화 마무리 멘트 추가
            farewell = "오늘 이야기를 들을 수 있어서 기뻤어요. 내일도 기다리고 있을게요 😊"
            self.chat_history.append({"role": "assistant", "content": farewell})

            diary_title, diary_theme, diary_body = self.generate_diary()
            self.save_diary(diary_title,diary_body,diary_theme,user_id)
    
            return farewell + "\n\n(일기가 저장되었어요. 프로그램을 종료합니다.)"


        # 키워드 추출
        keywords = self.extract_keywords(user_input)

        # 키워드 통합 검색
        query = self.gpt_build_query(keywords)

        results = self.db_manager.search(user_id,keywords,query)

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
                "./prompt/recall_prompt_test3.txt",
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
            # 대화 히스토리가 너무 짧은 경우, 종료로 간주하지 않음
            if len(self.chat_history) <= 2:
                return False

            # 최근 대화 6개만 사용
            recent_history = self.chat_history[-6:]

            # 시스템 지시 및 히스토리 포함 메시지 구성
            messages = [
                {"role": "system", "content": (
                    "다음은 사용자와 챗봇 사이의 최근 대화입니다.\n"
                    "이 대화의 마지막 사용자 발화가 대화를 끝내려는 의도인지 판단해 주세요.\n"
                    "반드시 '예' 또는 '아니오'로만 대답해 주세요.\n"
                    "대화 시작 인사('안녕', '하이', '안녕하세요') 등은 종료가 아닙니다.\n"
                )}
            ]
            messages.extend(recent_history)
            messages.append({
                "role": "system",
                "content": "위 대화에서 마지막 사용자의 발화는 대화를 끝내려는 의도입니까? 반드시 '예' 또는 '아니오'로만 대답하세요."
            })

            # GPT 모델 호출
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",  # 또는 "gpt-3.5-turbo", "gpt-4" 등
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
    def generate_diary(self) -> Tuple[str, str,str]:
        summary_prompt = {
            "role": "system",
            "content": self._load_prompt("./prompt/diary_gen_prompt_test2.txt")
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
        # 파일명에 사용할 수 없는 문자 및 줄바꿈 제거
        sanitized = re.sub(r'[\\/*?:"<>|\r\n]', '', title)
        sanitized = sanitized.replace(' ', '_')
        # 파일명 길이 제한 (예: 100자 이하로 자르기)
        return sanitized[:100]

    # 일기 저장
    def save_diary(self, title: str, body: str, theme:str,user_id: str):
        today = datetime.now().strftime("%Y-%m-%d")
        save_dir = "./diary/theme_diaries"
        os.makedirs(save_dir, exist_ok=True)

        # 제목을 파일명으로 사용하되, 파일명에 적합하도록 정리
        safe_title = self.sanitize_filename(title)
        file_name = f"diary_{today}_{safe_title}.txt"
        file_path = os.path.join(save_dir, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"제목: {title}\n\n{body}")

        # 벡터 DB 업데이트
        self.db_manager.create_or_update_index(
            user_id=user_id,  # 유저 ID 추가
            diary_texts=[body],
            metadata_list=[{
                "date": today,
                "title": title,
                "theme" : theme                            
            }]
        )


if __name__ == "__main__":
    # 로컬 테스트용 user_id 생성
    user_id = "test_user"

    # RT_Theme_Chatbot 인스턴스 생성
    theme_bot = RT_Theme_Chatbot()

    # 💬 챗봇이 먼저 질문
    print("🤖 챗봇:", theme_bot.start_conversation(user_id))

    while True:
        # 사용자 입력 받기
        msg = input("🙋 사용자: ")

        # 대화 종료 조건 확인
        if msg.lower() in ["종료", "exit", "quit"]:
            print("프로그램을 종료합니다.")
            break

        # 챗봇 응답 생성
        reply = theme_bot.ask(msg, user_id=user_id)  # user_id 전달
        print("🤖 챗봇:", reply)

        # 디버깅용 출력
        print("----- 디버깅용 -----")
        print(f"현재 대화 기록: {theme_bot.chat_history}")