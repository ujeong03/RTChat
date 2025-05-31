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
        if os.path.exists(persist_path):
            self.vectordb = FAISS.load_local(persist_path, self.embedding, allow_dangerous_deserialization=True)
        else:
            self.vectordb = None

    def create_or_update_index(self, diary_texts: list[str], metadata_list: list[dict]):
        docs = []
        for text, meta in zip(diary_texts, metadata_list):
            # 문단/문장 단위로 분할하고 각 chunk에 동일한 metadata 복사
            split_docs = [Document(page_content=text, metadata=meta)]
            docs.extend(split_docs)

        new_db = FAISS.from_documents(docs, self.embedding)

        if self.vectordb:
            self.vectordb.merge_from(new_db)
        else:
            self.vectordb = new_db

        self.vectordb.save_local(self.persist_path)

    def search(self, kw_list: list[str], query: str, top_k=3, score_threshold=2.0, min_match: int = 1):
        results = self.vectordb.similarity_search_with_score(query, k=top_k * 10)  # 후보군 확보
        seen_texts = set()
        filtered_docs = []

        query_keywords = [kw.strip().lower() for kw in kw_list if len(kw.strip()) > 1]

        def keyword_match_info(doc: Document) -> tuple[int, list[str]]:
            content = doc.page_content.lower()
            matched = [kw for kw in query_keywords if kw in content]
            return len(matched), matched

        for doc, score in results:
            doc_text = doc.page_content.strip()
            if doc_text in seen_texts or score > score_threshold:
                continue

            match_count, matched_keywords = keyword_match_info(doc)
            if match_count >= min_match:
                doc.metadata["match_count"] = match_count
                doc.metadata["matched_keywords"] = matched_keywords
                filtered_docs.append((doc, score))
                seen_texts.add(doc_text)

        # 키워드 매치 + 유사도 거리 기반 재정렬
        re_ranked = sorted(
            filtered_docs,
            key=lambda x: (-x[0].metadata["match_count"], x[1])  # 키워드 매치 많고, 유사도 가까운 순
        )

        final_docs = [doc for doc, _ in re_ranked[:top_k]]

       
        return final_docs
    
    def search_all_diaries(self):
        """
        FAISS 벡터스토어에 저장된 모든 일기 문서를 반환합니다.
        """
        try:
            # FAISS에서는 빈 쿼리로도 유사도 검색 가능.
            # 모든 문서를 가져오기 위해 큰 k 설정 (예: 1000)
            return self.vectordb.similarity_search(query="", k=1000)
        except Exception as e:
            print(f"[ERROR] search_all_diaries 실패: {e}")
            return []

    
    def get_diary_7days_by_date(self, date: str) -> list[Document]:
        """
        주어진 날짜(YYYY-MM-DD) 기준으로 7일 이내에 작성된 일기들을 반환.
        metadata에 'date' 키가 존재하고, 포맷은 'YYYY-MM-DD'라고 가정.
        """
        if not self.vectordb:
            return []

        # 문자열을 datetime 객체로 변환
        reference_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = reference_date - timedelta(days=6)  # 7일 간격이므로 오늘 포함 6일 전부터

        # 모든 문서 가져와서 날짜 필터링
        results = []
        for doc in self.vectordb.docstore._dict.values():
            diary_date_str = doc.metadata.get("date")
            if not diary_date_str:
                continue
            try:
                diary_date = datetime.strptime(diary_date_str, "%Y-%m-%d")
                if start_date <= diary_date <= reference_date:
                    results.append(doc)
            except ValueError:
                continue  # 날짜 포맷 오류 무시

        return results