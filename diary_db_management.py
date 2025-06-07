from datetime import datetime, timedelta
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai


class DiaryDBManager:
    def __init__(self, persist_path="vectorstore/diary_faiss"):
        self.persist_path = persist_path
        self.embedding = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # FAISS 데이터베이스 로드 또는 초기화
        if os.path.exists(persist_path):
            try:
                self.vectordb = FAISS.load_local(persist_path, self.embedding, allow_dangerous_deserialization=True)
                print(f"[FAISS 로드 완료] '{persist_path}'에서 데이터베이스를 로드했습니다.")
            except Exception as e:
                print(f"[FAISS 로드 실패] {e}. 새로 생성합니다.")
                self._initialize_faiss_database()
        else:
            print(f"[FAISS 초기화] '{persist_path}' 경로가 없습니다. 새로 생성합니다.")
            os.makedirs(persist_path, exist_ok=True)
            self._initialize_faiss_database()

    def _initialize_faiss_database(self):
        # 기본 텍스트를 사용하여 FAISS 데이터베이스 생성
        default_texts = ["기본 텍스트입니다. 데이터베이스를 초기화합니다."]
        self.vectordb = FAISS.from_texts(default_texts, self.embedding)
        self.vectordb.save_local(self.persist_path)
        print(f"[FAISS 저장 완료] '{self.persist_path}'에 새 데이터베이스를 생성했습니다.")

    def create_or_update_index(self, user_id: str, diary_texts: list[str], metadata_list: list[dict]):
        docs = []
        for text, meta in zip(diary_texts, metadata_list):
            meta["user_id"] = user_id  # 사용자 ID 포함
            docs.append(Document(page_content=text, metadata=meta))

        new_db = FAISS.from_documents(docs, self.embedding)

        if self.vectordb:
            self.vectordb.merge_from(new_db)
        else:
            self.vectordb = new_db

        self.vectordb.save_local(self.persist_path)


    def search(self, user_id: str, kw_list: list[str], query: str, top_k=3, score_threshold=2.0, min_match: int = 1):
        results = self.vectordb.similarity_search_with_score(query, k=top_k * 10)
        seen_texts = set()
        filtered_docs = []

        query_keywords = [kw.strip().lower() for kw in kw_list if len(kw.strip()) > 1]

        def keyword_match_info(doc: Document) -> tuple[int, list[str]]:
            content = doc.page_content.lower()
            matched = [kw for kw in query_keywords if kw in content]
            return len(matched), matched

        for doc, score in results:
            if doc.metadata.get("user_id") != user_id:  # 유저 필터
                continue

            doc_text = doc.page_content.strip()
            if doc_text in seen_texts or score > score_threshold:
                continue

            match_count, matched_keywords = keyword_match_info(doc)
            if match_count >= min_match:
                doc.metadata["match_count"] = match_count
                doc.metadata["matched_keywords"] = matched_keywords
                filtered_docs.append((doc, score))
                seen_texts.add(doc_text)

        re_ranked = sorted(
            filtered_docs,
            key=lambda x: (-x[0].metadata["match_count"], x[1])
        )

        return [doc for doc, _ in re_ranked[:top_k]]

    
    def search_all_diaries(self, user_id: str) -> list[Document]:
        """
        주어진 user_id에 해당하는 모든 일기 문서를 반환합니다.
        """
        try:
            all_docs = self.vectordb.similarity_search(query="", k=1000)  # 일단 다 가져오기
            user_docs = [doc for doc in all_docs if doc.metadata.get("user_id") == user_id]
            return user_docs
        except Exception as e:
            print(f"[ERROR] search_all_diaries 실패: {e}")
            return []


    def get_diary_7days_by_date(self, user_id: str, date: str) -> list[Document]:
        if not self.vectordb:
            return []

        reference_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = reference_date - timedelta(days=6)
        results = []

        for doc in self.vectordb.docstore._dict.values():
            if doc.metadata.get("user_id") != user_id and doc.metadata.get('daily_diary') == "daily_diary":
                continue

            diary_date_str = doc.metadata.get("date")
            if not diary_date_str:
                continue

            try:
                diary_date = datetime.strptime(diary_date_str, "%Y-%m-%d")
                if start_date <= diary_date <= reference_date:
                    results.append(doc)
            except ValueError:
                continue

        return results
    