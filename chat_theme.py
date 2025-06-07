import openai
from datetime import datetime
import os
from dotenv import load_dotenv
import requests
from diary_db_management import DiaryDBManager
from typing import List, Tuple, Dict, Optional
from collections import Counter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"
PROMPT_DIARY_GEN_PATH = "./prompt/diary_gen_prompt.txt"

# 프롬프트 파일 경로 상수 정의
PROMPT_SELECT_THEME_PATH = "./prompt/theme_select_prompt.txt"
PROMPT_FIRST_QUESTION_PATH = "./prompt/theme_first_question_prompt.txt"
PROMPT_FOLLOW_UP_PATH = "./prompt/theme_follow_up_prompt.txt"
PROMPT_END_CHECK_PATH = "./prompt/end_check_prompt.txt"

class RT_Theme_Chatbot:
    def __init__(self, db_manager: DiaryDBManager = None):
        openai.api_key = OPENAI_API_KEY
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.db_manager = db_manager
        self.chat_history: List[Dict[str, str]] = []
        self.awaiting_end_confirmation: bool = False
        self.current_theme: str = "학교/학창시절"  # 기본 테마 설정

    def reset(self) -> None:
        """챗봇 상태 초기화"""
        self.chat_history.clear()

    # ──────────────────────────────────   
    # 1) 테마 선택 단계 (ReACT)
    def select_theme_react(self, user_id: str) -> str:
        profile = self._fetch_user_profile(user_id)
        profile_info = self._format_profile_info(profile)

        # 테마별 회상 횟수 가져오기
        docs = self.db_manager.search_all_diaries(user_id)
        themes = [doc.metadata.get("theme", "").strip() for doc in docs if doc.metadata.get("theme")]
        counts = Counter(themes)

        # 테마 리스트 문자열
        theme_order = {
            1: "학교/학창시절", 2: "여가/취미", 3: "출생/성장",
            4: "결혼/가족", 5: "특별한 사건", 6: "직업/일",
            7: "음식/간식", 8: "군대 경험", 9: "명절/절기"
        }
        theme_list_str = "\n".join(f"{num}: {theme_order[num]}" for num in theme_order)
        theme_count_str = "\n".join(
            f"{theme_order[num]}: {counts.get(theme_order[num], 0)}"
            for num in theme_order
        )

        # 프롬프트 파일 로드 및 형식 지정
        select_prompt_template = self._load_prompt(PROMPT_SELECT_THEME_PATH)
        select_prompt = select_prompt_template.format(
            profile_info=profile_info,
            theme_list_str=theme_list_str,
            theme_count_str=theme_count_str
        )

        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": select_prompt}],
            temperature=0.0,
            max_tokens=3
        )
        return resp.choices[0].message.content.strip()

    # ──────────────────────────────────
    # 2) 첫 회상 질문 생성 단계 (ReACT)
    def generate_first_question_react(self, user_id: str, selected_num: str) -> str:
        profile = self._fetch_user_profile(user_id)
        profile_info = self._format_profile_info(profile)
        mapping = {
            "1": "학교/학창시절", "2": "여가/취미", "3": "출생/성장",
            "4": "결혼/가족", "5": "특별한 사건", "6": "직업/일",
            "7": "음식/간식", "8": "군대 경험", "9": "명절/절기"
        }
        theme_name = mapping.get(selected_num, "학교/학창시절")
        self.current_theme = theme_name  # 현재 테마 저장

        observation = "없음"  # (필요 시 여기에 과거 일기 요약 삽입)

        # 프롬프트 파일 로드 및 형식 지정
        question_prompt_template = self._load_prompt(PROMPT_FIRST_QUESTION_PATH)
        question_prompt = question_prompt_template.format(
            theme_name=theme_name,
            profile_info=profile_info,
            chat_history=self._get_chat_history_text(limit=5),
            observation=observation
        )

        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": question_prompt}],
            temperature=0.7,
            max_tokens=200
        )
        return resp.choices[0].message.content.strip()

    # ──────────────────────────────────
    # 3) 대화 시작
    def start_conversation(self, user_id: str) -> str:
        selected_num = self.select_theme_react(user_id)
        first_q = self.generate_first_question_react(user_id, selected_num)
        self.chat_history.append({"role": "assistant", "content": first_q})
        return first_q

    # ──────────────────────────────────
    # 4) 후속 대화 (ask)
    def ask(self, user_input: str, user_id: str) -> Dict[str, Optional[str]]:
        # 1) 사용자 메시지 기록
        self.chat_history.append({"role": "user", "content": user_input})

        # 2) 프로필 조회 & 포맷
        user_profile = self._fetch_user_profile(user_id)
        profile_info = self._format_profile_info(user_profile)

        # --- 3) 종료 확인 단계 ---
        # 만약 이전 턴에 "끝내고 싶다"는 신호가 와서 확인 질문을 던진 상태라면
        if self.awaiting_end_confirmation:
            # 이번 턴(지금 들어온 user_input)이 실제로 다시 "끝내고 싶다"는 뉘앙스인지 확인
            if self._is_conversation_ending():
                # 두 번째로 종료 의사를 보였으므로 진짜 종료 처리
                farewell = "오늘 이야기를 들을 수 있어서 기뻤어요. 내일도 기다리고 있을게요 😊"
                self._append_assistant_message(farewell)
                diary_title, diary_body, diary_theme = self._generate_diary()
                diary_result = self._save_diary(diary_title, diary_body, diary_theme, user_id)
                # awaiting_end_confirmation 초기화
                self.awaiting_end_confirmation = False
                return {
                    "response": f"{farewell}\n\n(일기가 저장되었어요. 프로그램을 종료합니다.)",
                    "diary": diary_result,
                }
            else:
                # 두 번째로 종료 의사를 보이지 않았으므로 "확인 대기" 상태 해제
                self.awaiting_end_confirmation = False
                # 이후 일반 ReACT 로직으로 자연스럽게 넘어가도록 함

        # --- 4) 첫 번째 종료 의사 판단 ---
        # 아직 awaiting_end_confirmation이 False였다면, 종료 의도인지 판단
        if self._is_conversation_ending():
            # 첫 번째로 종료 의도가 감지됨 → 확인 질문만 던지고 반환
            self.awaiting_end_confirmation = True
            confirm = "혹시 지금 대화를 마무리하시고 싶으신가요? 다른 이야기는 다음에 또 나눠요 😊"
            self._append_assistant_message(confirm)
            return {"response": confirm}
        
        # 4) Thought 1: "사용자 발화를 이해하고, 어떤 후속 질문을 던질지 고민"
        # 프롬프트 파일 로드 및 형식 지정
        thought2_prompt_template = self._load_prompt(PROMPT_FOLLOW_UP_PATH)
        thought2_prompt = thought2_prompt_template.format(
            profile_info=profile_info,
            chat_history=self._get_chat_history_text(limit=5),
            user_input=user_input
        )

        resp2 = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": thought2_prompt}],
            temperature=0.7,
            max_tokens=200
        )
        answer = resp2.choices[0].message.content.strip()
        self.chat_history.append({"role": "assistant", "content": answer})
        return {"response": answer}


    def _fetch_user_profile(self, user_id: str) -> Dict[str, Optional[str]]:
        url = f"https://nabiya.site/api/users/get_user_info/?user_id={user_id}"
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[프로필 조회 실패] user_id={user_id} / error={e}")
            return {}

    def _format_profile_info(self, profile: Dict[str, Optional[str]]) -> str:
        name = profile.get("name", "알 수 없음")
        gender = profile.get("gender", "알 수 없음")
        birth_date = profile.get("birth_date")
        age = self._calculate_age(birth_date) if birth_date else "알 수 없음"
        married = profile.get("married", "알 수 없음")
        family = profile.get("family_relationship", "알 수 없음")
        return f"이름: {name}, 성별: {gender}, 나이: {age}, 결혼 여부: {married}, 가족 관계: {family}"
    
    @staticmethod
    def _calculate_age(birth_date: str) -> int:
        try:
            birth_year = int(birth_date.split("-")[0])
            return datetime.now().year - birth_year
        except Exception:
            return -1

    def _get_chat_history_text(self, limit: int = 5) -> str:
        lines = []
        for msg in self.chat_history[-limit:]:
            role = "당신" if msg["role"] == "user" else "AI"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def _is_conversation_ending(self) -> bool:
        """
        마지막 사용자 발화가 '대화를 끝내고 싶다'는 의도인지 판단합니다.
        단, 이미 한 번 확인 질문을 던진 상태(awaiting_end_confirmation=True)라면
        이번에도 종료 의도라면 True를 반환하고, 아니라면 False로 되돌립니다.
        """
        # 대화 기록이 너무 짧으면 종료 아님
        if len(self.chat_history) <= 2:
            return False

        # 최근 6개 메시지 가져오기
        recent_msgs = self.chat_history[-6:]
        
        # 프롬프트 파일 로드 및 형식 지정
        system_prompt_template = self._load_prompt(PROMPT_END_CHECK_PATH)
        system_prompt = system_prompt_template.format(
            chat_history=self._get_chat_history_text(limit=6)
        )
        
        messages = [{"role": "system", "content": system_prompt}] + recent_msgs + [
            {"role": "system", "content": "위 대화에서 마지막 사용자의 발화는 대화를 끝내려는 의도입니까? 반드시 '예' 또는 '아니오'로만 대답하세요."}
        ]

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=5,
            )
            answer = response.choices[0].message.content.strip().lower()
            is_intent = "예" in answer
        except Exception as e:
            print(f"[대화 종료 판단 실패] {e}")
            return False

        # ── 검증 로직 ──
        if is_intent:
            if self.awaiting_end_confirmation:
                # 이미 한번 "끝내도 되겠냐" 물어본 뒤, 사용자가 다시 종료 의사 표시 → 진짜 종료
                self.awaiting_end_confirmation = False
                return True
            else:
                # 처음으로 "끝내고 싶다"는 의도로 판정 → 확인 질문을 던지기 위해 True가 아닌 "확인 필요" 상태만 표시
                self.awaiting_end_confirmation = True
                return False
        else:
            # 종료 의사 없음 → confirmation 대기 상태 초기화
            self.awaiting_end_confirmation = False
            return False
        
    def _append_assistant_message(self, content: str) -> None:
        self.chat_history.append({"role": "assistant", "content": content})

    def _load_prompt(self, file_path: str) -> str:
        """프롬프트 파일을 로드합니다."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"[프롬프트 파일 없음] {file_path}")
            # 기본 프롬프트 반환 또는 에러 처리
            return ""

    def _extract_theme_from_chat(self) -> str:
        """대화 내용에서 현재 테마를 추출합니다."""
        # 이미 테마가 저장되어 있으면 그대로 반환
        if hasattr(self, 'current_theme') and self.current_theme:
            return self.current_theme
        
        # 테마 매핑
        theme_mapping = {
            "학교": "학교/학창시절", 
            "학창시절": "학교/학창시절",
            "여가": "여가/취미",
            "취미": "여가/취미",
            "출생": "출생/성장",
            "성장": "출생/성장",
            "결혼": "결혼/가족",
            "가족": "결혼/가족",
            "특별한 사건": "특별한 사건",
            "직업": "직업/일",
            "일": "직업/일",
            "음식": "음식/간식",
            "간식": "음식/간식",
            "군대": "군대 경험",
            "명절": "명절/절기",
            "절기": "명절/절기"
        }
        
        # 첫 번째 챗봇 메시지 확인 (테마 선택 결과 포함)
        if self.chat_history and len(self.chat_history) > 0:
            for msg in self.chat_history:
                if msg["role"] == "assistant":
                    message = msg["content"].lower()
                    
                    # 테마 키워드 찾기
                    for keyword, theme in theme_mapping.items():
                        if keyword.lower() in message:
                            return theme
        
        # 기본값 반환
        return "학교/학창시절"  # 기본값은 가장 우선순위가 높은 테마로 설정


    def _generate_diary(self) -> Tuple[str, str, str]:
        """대화 기록을 바탕으로 일기 제목, 본문, 테마를 생성합니다."""
        prompt = self._load_prompt(PROMPT_DIARY_GEN_PATH)
        messages = [{"role": "system", "content": prompt}] + self.chat_history
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=300,
            )
            diary_text = response.choices[0].message.content.strip()
            
            # 제목과 본문 추출
            if "제목 :" in diary_text and "본문 :" in diary_text:
                title = diary_text.split("제목 :")[1].split("본문 :")[0].strip()
                body = diary_text.split("본문 :")[1].strip()
            else:
                title, body = "무제", diary_text
                
            # 대화 내용에서 현재 테마 추출
            theme = self._extract_theme_from_chat()
            
            return title, body, theme
        except Exception as e:
            print(f"[일기 생성 실패] {e}")
            return "에러", "일기를 생성하는 데 문제가 발생했어요.", "일반"

    def _save_diary(self, title: str, body: str, theme:str,user_id: str):
        today = datetime.now().strftime("%Y-%m-%d")

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

        # 일기 저장 성공 시, 로컬에 txt 파일로 저장
        diary_path = f"diary/theme_diaries/{user_id}_{today}_{title}.txt"
        os.makedirs(os.path.dirname(diary_path), exist_ok=True)
        with open(diary_path, "w", encoding="utf-8") as f:
            f.write(f"제목: {title}\n\n본문:\n{body}")
        print(f"[일기 저장 완료] {diary_path}")
        # 일기 저장 결과 반환

        # JSON 형식으로 반환
        return {
            "title": title,
            "body": body
        }


if __name__ == "__main__":
    user_id = "test_user"
    global_db_manager = DiaryDBManager(persist_path="vectorstore/diary_faiss")

    theme_bot = RT_Theme_Chatbot(db_manager=global_db_manager)

    print("🤖 챗봇:", theme_bot.start_conversation(user_id))

    while True:
        msg = input("🙋 사용자: ")
        if msg.lower() in ["종료", "exit", "quit"]:
            print("프로그램을 종료합니다.")
            break

        reply = theme_bot.ask(msg, user_id=user_id)
        print("🤖 챗봇:", reply)