import os
import time
import re
from typing import List, Dict, Tuple
import numpy as np
from dotenv import load_dotenv
import openai
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()

class RAGSystem:
    def __init__(self, embedding_model: str = 'openai'):
        """RAG 시스템 초기화"""
        self.embedding_model = embedding_model
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        openai.api_key = self.api_key
        
        # 임베딩 모델 선택
        if embedding_model == 'openai':
            self.embedding_dim = 1536
            self.embedder = None  # OpenAI API 사용
        elif embedding_model == 'sentence-bert':
            self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_dim = 384
        else:
            raise ValueError(f"지원하지 않는 임베딩 모델: {embedding_model}")
        
        # 벡터 저장소
        self.index = None
        self.documents = []
        self.chunks = []
        self.metadata = []
        
    def _chunk_document(self, content: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
        """
        문서를 청크로 분할 (학생생활규정 최적화)
        
        학생생활규정의 구조를 고려하여 장(章), 조(條)를 기준으로 분할
        """
        chunks = []
        
        # 정규식으로 장(章) 분리
        chapters = re.split(r'(제\s*\d+\s*장[^\n]*)', content)
        
        current_chapter = ""
        for i, part in enumerate(chapters):
            if re.match(r'제\s*\d+\s*장', part):
                current_chapter = part.strip()
            elif part.strip():
                # 조(條) 단위로 분할
                sections = re.split(r'(제\s*\d+\s*조[^\n]*)', part)
                
                current_section = ""
                for j, section in enumerate(sections):
                    if re.match(r'제\s*\d+\s*조', section):
                        current_section = section.strip()
                    elif section.strip():
                        # 최종 청크 생성
                        chunk_text = f"{current_chapter}\n{current_section}\n{section.strip()}"
                        
                        if len(chunk_text) > chunk_size:
                            # 길이가 크면 추가 분할
                            sub_chunks = self._split_long_chunk(chunk_text, chunk_size, overlap)
                            chunks.extend(sub_chunks)
                        else:
                            chunks.append({
                                'text': chunk_text,
                                'chapter': current_chapter,
                                'section': current_section
                            })
        
        return chunks if chunks else self._fallback_chunking(content, chunk_size, overlap)
    
    def _split_long_chunk(self, text: str, chunk_size: int, overlap: int) -> List[Dict]:
        """긴 청크를 추가로 분할"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            # 문장 단위로 자르기
            last_period = chunk.rfind('.')
            if last_period > 0 and end < len(text):
                end = start + last_period + 1
            
            chunks.append({
                'text': text[start:end],
                'chapter': '',
                'section': ''
            })
            
            start = end - overlap
        
        return chunks
    
    def _fallback_chunking(self, content: str, chunk_size: int, overlap: int) -> List[Dict]:
        """기본 청크 분할 (폴백)"""
        chunks = []
        start = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            
            # 문장 단위로 자르기
            last_period = content.rfind('.', start, end)
            if last_period > start:
                end = last_period + 1
            
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'chapter': '',
                    'section': ''
                })
            
            start = end - overlap
        
        return chunks
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """텍스트의 임베딩 벡터 생성"""
        if self.embedding_model == 'openai':
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-3-small"
            )
            return np.array(response['data'][0]['embedding']).astype('float32')
        else:
            return self.embedder.encode(text, convert_to_numpy=True).astype('float32')
    
    def add_document(self, content: str) -> Dict:
        """문서 추가 및 벡터화"""
        # 문서 청크 분할
        chunks = self._chunk_document(content)
        
        # 각 청크의 임베딩 생성
        embeddings = []
        for chunk in chunks:
            embedding = self._get_embedding(chunk['text'])
            embeddings.append(embedding)
            self.chunks.append(chunk['text'])
            self.metadata.append({
                'chapter': chunk.get('chapter', ''),
                'section': chunk.get('section', ''),
                'document_id': len(self.documents)
            })
        
        # FAISS 인덱스 생성 또는 업데이트
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        
        self.documents.append(content)
        
        return {
            'success': True,
            'chunks_count': len(chunks),
            'document_id': len(self.documents) - 1
        }
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """관련 청크 검색"""
        if self.index is None or len(self.chunks) == 0:
            return []
        
        # 쿼리 임베딩
        query_embedding = self._get_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # 유사도 검색
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                # 거리를 유사도로 변환 (L2 거리는 작을수록 유사)
                similarity = 1 / (1 + distances[0][i])
                results.append((self.chunks[idx], similarity, self.metadata[idx]))
        
        return results
    
    def _rerank_chunks(self, query: str, chunks: List[Tuple[str, float, Dict]]) -> List[Tuple[str, float, Dict]]:
        """Re-ranking을 통한 청크 순서 재정렬"""
        if not chunks:
            return chunks
        
        # 쿼리와 청크의 키워드 매칭 스코어 계산
        query_keywords = set(query.lower().split())
        
        reranked = []
        for chunk, similarity, metadata in chunks:
            chunk_keywords = set(chunk.lower().split())
            keyword_overlap = len(query_keywords & chunk_keywords) / len(query_keywords) if query_keywords else 0
            
            # 최종 스코어 = 임베딩 유사도 * 0.7 + 키워드 매칭 * 0.3
            final_score = similarity * 0.7 + keyword_overlap * 0.3
            
            reranked.append((chunk, final_score, metadata))
        
        # 스코어 기준으로 정렬
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    def _extract_reference_info(self, chunk: str, metadata: Dict) -> str:
        """청크에서 참고 정보 추출 (장, 조 정보)"""
        # 정규식으로 장, 조 정보 추출
        chapter_match = re.search(r'제\s*(\d+)\s*장\s*([^\n]*)', chunk)
        section_match = re.search(r'제\s*(\d+)\s*조\s*([^\n]*)', chunk)
        
        reference = []
        
        if chapter_match:
            reference.append(f"제{chapter_match.group(1)}장 {chapter_match.group(2)}")

        
        if section_match:
            reference.append(f"제{section_match.group(1)}조 {section_match.group(2)}")
        
        if metadata.get('chapter'):
            reference.append(metadata['chapter'])
        
        if metadata.get('section'):
            reference.append(metadata['section'])
        
        return ' | '.join(reference) if reference else '학생생활규정'
    
    def generate_answer(self, question: str, use_rag: bool = True, use_reranking: bool = False, 
                       embedding_model: str = None) -> Dict:
        """질문에 대한 답변 생성"""
        start_time = time.time()
        
        # 임베딩 모델 변경 시 처리
        if embedding_model and embedding_model != self.embedding_model:
            self.embedding_model = embedding_model
        
        source_documents = []
        context = ""
        
        # RAG 사용 시 관련 청크 검색
        if use_rag and self.index is not None:
            retrieved_chunks = self._retrieve_relevant_chunks(question, top_k=5)
            
            # Re-ranking 적용
            if use_reranking:
                retrieved_chunks = self._rerank_chunks(question, retrieved_chunks)
            
            # 컨텍스트 구성
            for chunk, similarity, metadata in retrieved_chunks[:3]:
                context += f"\n{chunk}\n"
                reference = self._extract_reference_info(chunk, metadata)
                source_documents.append({
                    'text': chunk[:200] + '...' if len(chunk) > 200 else chunk,
                    'reference': reference,
                    'similarity': float(similarity)
                })
        
        # 프롬프트 구성
        if use_rag and context:
            system_prompt = """당신은 학교 생활규정에 대한 전문가 조수입니다.
주어진 학생생활규정 문서를 바탕으로 정확하고 구체적인 답변을 제공하세요.

중요한 지침:
1. 반드시 제공된 문서의 내용만을 기반으로 답변하세요.
2. 문서에 없는 내용은 "문서에 명시되지 않았습니다"라고 답변하세요.
3. 구체적인 조항 번호(예: 제22조)를 언급하세요.
4. 학생의 입장에서 이해하기 쉽게 설명하세요.
5. 정확성을 최우선으로 하세요. 추측이나 일반화는 피하세요."""
            
            user_message = f"""다음 학생생활규정 문서를 참고하여 질문에 답변해주세요.

【학생생활규정 발췌】
{context}

【학생의 질문】
{question}

위 문서의 내용을 바탕으로 정확하게 답변해주세요."""
        else:
            system_prompt = """당신은 학교 생활규정에 대한 일반적인 조수입니다.
학생의 질문에 일반적인 지식을 바탕으로 답변하세요."""
            
            user_message = f"""다음 질문에 답변해주세요:
{question}"""
        
        # OpenAI API 호출
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,  # 정확성 우선
                max_tokens=500
            )
            
            answer = response['choices'][0]['message']['content']
        except Exception as e:
            answer = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        
        response_time = time.time() - start_time
        
        # 신뢰도 계산
        confidence = 0.0
        if source_documents:
            confidence = np.mean([doc['similarity'] for doc in source_documents])
        
        return {
            'answer': answer,
            'source_documents': source_documents,
            'confidence': float(confidence),
            'response_time': response_time,
            'use_rag': use_rag,
            'use_reranking': use_reranking
        }
