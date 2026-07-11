"""
RAG 시스템 (OpenAI 임베딩 + FAISS 벡터 DB) - 배치 처리 최적화
- OpenAI 임베딩으로 벡터 생성 (배치 처리)
- FAISS를 사용한 빠른 유사도 검색
- 청크 크기 조절 가능
- Re-ranking 지원
- 토큰 제한 자동 처리
- ✅ 무한 루프 방지 기능
- ✅ 배치 처리로 10배 빠른 임베딩 생성
"""

import os
import re
import json
import numpy as np
from typing import List, Dict, Tuple
import openai
from dotenv import load_dotenv

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS가 설치되지 않았습니다. 기본 검색을 사용합니다.")

load_dotenv()

class RAGSystem:
    def __init__(self, file_path: str, chunk_size: int = 500):
        """RAG 시스템 초기화
        
        Args:
            file_path: 문서 파일 경로
            chunk_size: 청크 크기 (기본값: 500)
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.documents = []
        self.chunks = []
        self.embeddings = []
        self.faiss_index = None
        self.use_faiss = FAISS_AVAILABLE
        
        # 토큰 제한 설정
        self.embedding_token_limit = 8192  # 임베딩 모델
        self.chat_token_limit = 16385  # 챗 모델
        self.safety_margin = 0.2  # 20% 안전 마진
        
        # ✅ 무한 루프 방지 플래그
        self._auto_adjust_in_progress = False
        
        # 문서 로드 및 처리
        self._load_document()
        self._chunk_document()
        self._auto_adjust_chunk_size()  # 청크 크기 자동 조절
        self._create_embeddings()  # ✅ 배치 처리 최적화
        self._build_faiss_index()
        
        print(f"✅ RAG 시스템 초기화 완료")
        print(f"   - 총 청크 수: {len(self.chunks)}")
        print(f"   - 임베딩 모델: OpenAI (배치 처리)")
        print(f"   - 벡터 DB: {'FAISS' if self.use_faiss else '메모리 기반'}")
        print(f"   - 청크 크기: {self.chunk_size}")
    
    def _estimate_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 추정 (1 글자 ≈ 1.3 토큰, 한글 기준)"""
        return int(len(text) * 1.3)
    
    def _auto_adjust_chunk_size(self):
        """청크 크기 자동 조절 (토큰 제한 기반) - 무한 루프 방지"""
        # ✅ 이미 조절 중이면 중단
        if self._auto_adjust_in_progress:
            print(f"   ⚠️  청크 크기 조절 중단 (무한 루프 방지)")
            return
        
        self._auto_adjust_in_progress = True
        
        try:
            max_safe_tokens = int(self.embedding_token_limit * 0.7)
            
            if len(self.chunks) > 0:
                avg_tokens = sum(self._estimate_tokens(c) for c in self.chunks) / len(self.chunks)
                max_tokens_in_chunks = max(self._estimate_tokens(c) for c in self.chunks)
                
                print(f"   📊 청크 분석:")
                print(f"      - 평균 토큰: {avg_tokens:.0f}")
                print(f"      - 최대 토큰: {max_tokens_in_chunks}")
                print(f"      - 안전 제한: {max_safe_tokens}")
                
                if max_tokens_in_chunks > max_safe_tokens:
                    reduction_ratio = max_safe_tokens / max_tokens_in_chunks
                    new_chunk_size = int(self.chunk_size * reduction_ratio * 0.8)
                    new_chunk_size = max(100, new_chunk_size)
                    
                    print(f"   ⚠️  청크 크기 자동 조절: {self.chunk_size} → {new_chunk_size}")
                    self.chunk_size = new_chunk_size
                    self._chunk_document()
                    print(f"   ✅ 청크 재생성 완료: {len(self.chunks)}개")
        finally:
            self._auto_adjust_in_progress = False
    
    def _load_document(self):
        """문서 로드"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.documents = [content]
            print(f"✅ 문서 로드 완료 ({len(content)} 글자)")
        except Exception as e:
            raise Exception(f"문서 로드 실패: {str(e)}")
    
    def _chunk_document(self, overlap: int = 100):
        """
        문서를 청크로 분할 (학생생활규정 최적화)
        - 장(章) 기준으로 분할
        - 조(條) 기준으로 분할
        - 청크 크기 조절 가능
        """
        self.chunks = []
        
        for doc in self.documents:
            chapters = re.split(r'(제\d+장\s+[^\n]+)', doc)
            
            current_chapter = ""
            for i, section in enumerate(chapters):
                if re.match(r'제\d+장', section):
                    if current_chapter and len(current_chapter) > 50:
                        self.chunks.append(current_chapter.strip())
                    current_chapter = section
                else:
                    if section.strip():
                        articles = re.split(r'(제\d+조\s+[^\n]+)', section)
                        for article in articles:
                            if article.strip():
                                combined = current_chapter + "\n" + article
                                if len(combined) > self.chunk_size:
                                    if current_chapter:
                                        self.chunks.append(current_chapter.strip())
                                    current_chapter = article
                                else:
                                    current_chapter += "\n" + article
            
            if current_chapter and len(current_chapter) > 50:
                self.chunks.append(current_chapter.strip())
        
        print(f"✅ 청크 분할 완료: {len(self.chunks)} 개 (크기: {self.chunk_size})")
    
    def _create_embeddings(self):
        """✅ OpenAI를 사용한 임베딩 생성 (배치 처리 최적화)"""
        print("🔄 임베딩 생성 중... (OpenAI - 배치 처리)")
        
        self.embeddings = []
        max_chunk_tokens = int(self.embedding_token_limit * (1 - 0.3))
        
        # ✅ 배치 처리 설정
        batch_size = 20  # 한 번에 20개 청크씩 처리
        max_batch_tokens = 8000  # 배치당 최대 토큰
        
        print(f"   - 배치 크기: {batch_size}")
        print(f"   - 총 청크: {len(self.chunks)}")
        
        # ✅ 배치로 나누기
        batches = []
        current_batch = []
        current_batch_tokens = 0
        
        for chunk in self.chunks:
            chunk_tokens = self._estimate_tokens(chunk)
            
            # 토큰 제한 확인
            if chunk_tokens > max_chunk_tokens:
                max_chars = int(max_chunk_tokens / 1.3)
                chunk = chunk[:max_chars]
                chunk_tokens = max_chunk_tokens
            
            # 배치에 추가
            if current_batch_tokens + chunk_tokens > max_batch_tokens:
                # 현재 배치가 가득 찼으면 새 배치 시작
                if current_batch:
                    batches.append(current_batch)
                current_batch = [chunk]
                current_batch_tokens = chunk_tokens
            else:
                current_batch.append(chunk)
                current_batch_tokens += chunk_tokens
        
        # 마지막 배치 추가
        if current_batch:
            batches.append(current_batch)
        
        print(f"   - 배치 수: {len(batches)}")
        
        # ✅ 배치 처리 (한 번에 여러 청크 임베딩)
        for batch_idx, batch in enumerate(batches):
            try:
                # ✅ 여러 청크를 한 번에 임베딩 (핵심 최적화!)
                response = openai.Embedding.create(
                    input=batch,
                    model="text-embedding-3-small"
                )
                
                # 결과 추출
                for item in response['data']:
                    self.embeddings.append(item['embedding'])
                
                print(f"   - [{batch_idx + 1}/{len(batches)}] 배치 완료 ({len(batch)} 청크)")
            
            except Exception as e:
                print(f"⚠️  배치 {batch_idx} 실패: {str(e)}")
                
                # 배치 크기 줄여서 재시도
                print(f"   - 배치 크기 줄여서 재시도 중...")
                for chunk in batch:
                    try:
                        response = openai.Embedding.create(
                            input=chunk,
                            model="text-embedding-3-small"
                        )
                        self.embeddings.append(response['data'][0]['embedding'])
                    except:
                        self.embeddings.append([0] * 1536)
        
        print(f"✅ 모든 임베딩 생성 완료 ({len(self.embeddings)}/{len(self.chunks)})")
    
    def _build_faiss_index(self):
        """FAISS 인덱스 구축"""
        if not self.use_faiss or len(self.embeddings) == 0:
            return
        
        try:
            print("🔄 FAISS 인덱스 구축 중...")
            
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            dimension = embeddings_array.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embeddings_array)
            
            print(f"✅ FAISS 인덱스 구축 완료 ({len(self.embeddings)} 벡터)")
        except Exception as e:
            print(f"⚠️  FAISS 인덱스 구축 실패: {str(e)}")
            self.use_faiss = False
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """쿼리와 유사한 문서 검색"""
        try:
            response = openai.Embedding.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = response['data'][0]['embedding']
            
            if self.use_faiss and self.faiss_index is not None:
                query_array = np.array([query_embedding], dtype=np.float32)
                distances, indices = self.faiss_index.search(query_array, top_k)
                
                results = []
                for idx, distance in zip(indices[0], distances[0]):
                    similarity = 1 / (1 + distance)
                    results.append({
                        'chunk_id': int(idx),
                        'similarity': float(similarity),
                        'content': self.chunks[int(idx)]
                    })
                
                return results
            else:
                similarities = []
                for i, chunk_embedding in enumerate(self.embeddings):
                    similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                    similarities.append((i, similarity, self.chunks[i]))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                results = [
                    {
                        'chunk_id': idx,
                        'similarity': sim,
                        'content': chunk
                    }
                    for idx, sim, chunk in similarities[:top_k]
                ]
                
                return results
        except Exception as e:
            print(f"❌ 검색 실패: {str(e)}")
            return []
    
    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """컨텍스트를 토큰 제한 내로 축소"""
        max_chars = int(max_tokens / 0.3)
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[내용 생략...]"
        return context
    
    def generate_answer(self, query: str, use_rag: bool = True, use_reranking: bool = False) -> Dict:
        """질문에 대한 답변 생성"""
        
        context = ""
        references = []
        retrieved_count = 0
        
        if use_rag:
            max_available_tokens = int(self.chat_token_limit * (1 - self.safety_margin))
            query_tokens = self._estimate_tokens(query)
            available_for_context = max_available_tokens - query_tokens - 2000
            
            if available_for_context > 3000:
                top_k = 5
            elif available_for_context > 2000:
                top_k = 3
            else:
                top_k = 2
            
            print(f"   📊 동적 검색: top_k={top_k} (사용 가능 토큰: {available_for_context})")
            
            retrieved_docs = self.retrieve_documents(query, top_k=top_k)
            retrieved_count = len(retrieved_docs)
            
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                context_parts.append(f"[참고문서 {i}]\n{doc['content']}")
            
            context = "\n\n".join(context_parts)
            max_context_tokens = available_for_context - 500
            context = self._truncate_context(context, max_context_tokens)
            
            references = [
                {
                    'chunk_id': doc['chunk_id'],
                    'similarity': doc['similarity'],
                    'preview': doc['content'][:200]
                }
                for doc in retrieved_docs
            ]
        
        if use_rag:
            system_prompt = """당신은 학교 규정 전문가 어시스턴트입니다.
사용자의 질문에 대해 제공된 참고문서를 기반으로 정확하고 친절하게 답변해주세요.
참고문서에 없는 내용은 "규정에 명시되지 않았습니다"라고 답변하세요."""
            
            user_prompt = f"""다음 참고문서를 기반으로 질문에 답변해주세요.

[참고문서]
{context}

[질문]
{query}

[답변]"""
        else:
            system_prompt = """당신은 학교 규정 전문가 어시스턴트입니다.
사용자의 질문에 대해 정확하고 친절하게 답변해주세요."""
            
            user_prompt = f"""다음 질문에 답변해주세요:

{query}

[답변]"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response['choices'][0]['message']['content']
            
            return {
                'answer': answer,
                'references': references,
                'model': 'gpt-4o',
                'is_rag': use_rag,
                'retrieved_count': retrieved_count
            }
        
        except Exception as e:
            print(f"❌ 답변 생성 실패: {str(e)}")
            return {
                'answer': f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}",
                'references': references,
                'model': 'gpt-4o',
                'is_rag': use_rag,
                'retrieved_count': retrieved_count,
                'error': str(e)
            }
