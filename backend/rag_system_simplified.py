"""
RAG 시스템 (OpenAI 임베딩 + FAISS 벡터 DB)
- OpenAI 임베딩으로 벡터 생성
- FAISS를 사용한 빠른 유사도 검색
- 청크 크기 조절 가능
- Re-ranking 지원
- 토큰 제한 자동 처리
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
        
        # 문서 로드 및 처리
        self._load_document()
        self._chunk_document()
        self._auto_adjust_chunk_size()  # 청크 크기 자동 조절
        self._create_embeddings()
        self._build_faiss_index()
        
        print(f"✅ RAG 시스템 초기화 완료")
        print(f"   - 총 청크 수: {len(self.chunks)}")
        print(f"   - 임베딩 모델: OpenAI")
        print(f"   - 벡터 DB: {'FAISS' if self.use_faiss else '메모리 기반'}")
        print(f"   - 청크 크기: {self.chunk_size}")
    
    def _estimate_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 추정 (1 글자 ≈ 1.3 토큰, 한글 기준)"""
        # 한글은 1 글자 ≈ 1.3 토큰
        # 영문은 1 글자 ≈ 0.25 토큰
        # 보수적으로 1.3으로 설정
        return int(len(text) * 1.3)
    
    def _auto_adjust_chunk_size(self):
        """청크 크기 자동 조절 (토큰 제한 기반)"""
        # 임베딩 모델 제한: 8192 토큰
        # 안전 마진: 30%
        max_safe_tokens = int(self.embedding_token_limit * 0.7)
        
        # 현재 청크의 평균 토큰
        if len(self.chunks) > 0:
            avg_tokens = sum(self._estimate_tokens(c) for c in self.chunks) / len(self.chunks)
            max_tokens_in_chunks = max(self._estimate_tokens(c) for c in self.chunks)
            
            print(f"   📊 청크 분석:")
            print(f"      - 평균 토큰: {avg_tokens:.0f}")
            print(f"      - 최대 토큰: {max_tokens_in_chunks}")
            print(f"      - 안전 제한: {max_safe_tokens}")
            
            # 최대 청크가 제한을 초과하면 청크 크기 축소
            if max_tokens_in_chunks > max_safe_tokens:
                # 새 청크 크기 계산
                reduction_ratio = max_safe_tokens / max_tokens_in_chunks
                new_chunk_size = int(self.chunk_size * reduction_ratio * 0.8)  # 추가 20% 안전 마진
                new_chunk_size = max(100, new_chunk_size)  # 최소 100
                
                print(f"   ⚠️  청크 크기 자동 조절: {self.chunk_size} → {new_chunk_size}")
                self.chunk_size = new_chunk_size
                self._chunk_document()
                print(f"   ✅ 청크 재생성 완료: {len(self.chunks)}개")
    
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
            # 장(章) 기준 분할
            chapters = re.split(r'(제\d+장\s+[^\n]+)', doc)
            
            current_chapter = ""
            for i, section in enumerate(chapters):
                if re.match(r'제\d+장', section):
                    # 새로운 장 시작
                    if current_chapter and len(current_chapter) > 50:
                        self.chunks.append(current_chapter.strip())
                    current_chapter = section
                else:
                    # 조(條) 기준으로 추가 분할
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
        """OpenAI를 사용한 임베딩 생성 (토큰 제한 처리)"""
        print("🔄 임베딩 생성 중... (OpenAI)")
        
        self.embeddings = []
        # 임베딩 모델 토큰 제한: 8192
        # 안전 마진: 30% (2457 토큰)
        # 사용 가능: 5735 토큰
        max_chunk_tokens = int(self.embedding_token_limit * (1 - 0.3))
        
        print(f"   - 최대 청크 토큰: {max_chunk_tokens}")
        
        for i, chunk in enumerate(self.chunks):
            try:
                # 토큰 제한 확인
                chunk_tokens = self._estimate_tokens(chunk)
                
                if chunk_tokens > max_chunk_tokens:
                    # 청크가 너무 크면 자동 분할
                    max_chars = int(max_chunk_tokens / 1.3)
                    truncated_chunk = chunk[:max_chars]
                    print(f"   ⚠️  청크 {i} 크기 초과: {chunk_tokens} → {self._estimate_tokens(truncated_chunk)} 토큰으로 축소")
                    chunk = truncated_chunk
                
                # OpenAI 임베딩 API 호출
                response = openai.Embedding.create(
                    input=chunk,
                    model="text-embedding-3-small"
                )
                embedding = response['data'][0]['embedding']
                self.embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    print(f"   - {i + 1}/{len(self.chunks)} 임베딩 생성 완료")
            except Exception as e:
                print(f"⚠️  청크 {i} 임베딩 실패: {str(e)}")
                # 재시도 1: 70% 크기로
                try:
                    max_chars = int(max_chunk_tokens / 1.3 * 0.7)
                    response = openai.Embedding.create(
                        input=chunk[:max_chars],
                        model="text-embedding-3-small"
                    )
                    embedding = response['data'][0]['embedding']
                    self.embeddings.append(embedding)
                    print(f"   ✅ 청크 {i} 재시도 1 성공 (70%)")
                except:
                    # 재시도 2: 50% 크기로
                    try:
                        max_chars = int(max_chunk_tokens / 1.3 * 0.5)
                        response = openai.Embedding.create(
                            input=chunk[:max_chars],
                            model="text-embedding-3-small"
                        )
                        embedding = response['data'][0]['embedding']
                        self.embeddings.append(embedding)
                        print(f"   ✅ 청크 {i} 재시도 2 성공 (50%)")
                    except:
                        # 재시도 3: 30% 크기로
                        try:
                            max_chars = int(max_chunk_tokens / 1.3 * 0.3)
                            response = openai.Embedding.create(
                                input=chunk[:max_chars],
                                model="text-embedding-3-small"
                            )
                            embedding = response['data'][0]['embedding']
                            self.embeddings.append(embedding)
                            print(f"   ✅ 청크 {i} 재시도 3 성공 (30%)")
                        except:
                            self.embeddings.append([0] * 1536)  # 기본값
                            print(f"   ❌ 청크 {i} 최종 실패 - 기본값 사용")
        
        print(f"✅ 모든 임베딩 생성 완료 ({len(self.embeddings)}/{len(self.chunks)})")
    
    def _build_faiss_index(self):
        """FAISS 인덱스 구축"""
        if not self.use_faiss or len(self.embeddings) == 0:
            return
        
        try:
            print("🔄 FAISS 인덱스 구축 중...")
            
            # 임베딩을 numpy 배열로 변환
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            
            # FAISS 인덱스 생성 (L2 거리)
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
        """쿼리와 유사한 문서 검색 (FAISS 또는 메모리 기반, 토큰 제한 처리)"""
        try:
            # 쿼리 임베딩 생성
            response = openai.Embedding.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = response['data'][0]['embedding']
            
            if self.use_faiss and self.faiss_index is not None:
                # FAISS를 사용한 검색 (빠름)
                query_array = np.array([query_embedding], dtype=np.float32)
                distances, indices = self.faiss_index.search(query_array, top_k)
                
                results = []
                for idx, distance in zip(indices[0], distances[0]):
                    # L2 거리를 유사도로 변환 (작을수록 유사)
                    similarity = 1 / (1 + distance)
                    results.append({
                        'chunk_id': int(idx),
                        'similarity': float(similarity),
                        'content': self.chunks[int(idx)]
                    })
                
                return results
            else:
                # 메모리 기반 검색 (느림)
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
        """
        질문에 대한 답변 생성 (토큰 제한 자동 처리)
        
        Args:
            query: 사용자 질문
            use_rag: RAG 사용 여부
            use_reranking: Re-ranking 사용 여부
        
        Returns:
            {
                'answer': 생성된 답변,
                'references': 참고 문서,
                'model': 사용된 모델,
                'is_rag': RAG 사용 여부
            }
        """
        
        # 1. RAG 검색 (선택사항)
        context = ""
        references = []
        retrieved_count = 0
        
        if use_rag:
            # 동적 검색 수 조절
            max_available_tokens = int(self.chat_token_limit * (1 - self.safety_margin))
            query_tokens = self._estimate_tokens(query)
            available_for_context = max_available_tokens - query_tokens - 2000  # 2000: 답변용
            
            # 검색할 청크 수 동적 결정
            if available_for_context > 3000:
                top_k = 5
            elif available_for_context > 2000:
                top_k = 3
            else:
                top_k = 2
            
            print(f"   📊 동적 검색: top_k={top_k} (사용 가능 토큰: {available_for_context})")
            
            # 문서 검색
            retrieved_docs = self.retrieve_documents(query, top_k=top_k)
            retrieved_count = len(retrieved_docs)
            
            # Re-ranking (선택사항)
            if use_reranking and len(retrieved_docs) > 1:
                retrieved_docs = self._rerank_documents(query, retrieved_docs)
            
            # 토큰 제한 내에서 최대한 많은 문서 포함
            context_parts = []
            current_tokens = 0
            
            for i, doc in enumerate(retrieved_docs):
                doc_tokens = self._estimate_tokens(doc['content'])
                if current_tokens + doc_tokens < available_for_context:
                    context_parts.append(f"[문서 {i+1}]\n{doc['content']}")
                    current_tokens += doc_tokens
                else:
                    # 남은 공간에 맞게 축소
                    remaining_tokens = available_for_context - current_tokens
                    if remaining_tokens > 500:
                        truncated = self._truncate_context(doc['content'], remaining_tokens)
                        context_parts.append(f"[문서 {i+1}]\n{truncated}")
                    break
            
            context = "\n\n".join(context_parts)
            
            # 참고 문서 추출
            references = self._extract_references(retrieved_docs[:3])
        
        # 2. 프롬프트 구성
        if use_rag:
            system_prompt = """당신은 학교 생활규정에 대해 정확하고 친절하게 답변하는 AI 어시스턴트입니다.

다음 규정 문서를 참고하여 학생의 질문에 정확하게 답변하세요.
제공된 문서에 명확한 답변이 없으면 "해당 내용은 제공된 문서에 없습니다"라고 답변하세요.
절대로 추측이나 일반적인 지식으로 답변하지 마세요.

제공된 규정 문서:
{context}

주의: 정확성이 가장 중요합니다. 확실하지 않으면 모른다고 답변하세요.""".format(context=context)
        else:
            system_prompt = """당신은 일반적인 AI 어시스턴트입니다.
사용자의 질문에 일반적인 지식을 바탕으로 답변하세요."""
        
        # 3. ChatGPT API 호출 (토큰 제한 확인)
        try:
            # 최종 토큰 확인
            total_tokens = self._estimate_tokens(system_prompt) + self._estimate_tokens(query)
            max_allowed = int(self.chat_token_limit * (1 - self.safety_margin))
            
            if total_tokens > max_allowed:
                print(f"   ⚠️  토큰 초과: {total_tokens} > {max_allowed}")
                # 컨텍스트 추가 축소
                max_context_tokens = max_allowed - self._estimate_tokens(query) - 500
                context = self._truncate_context(context, max_context_tokens)
                
                # 프롬프트 재구성
                if use_rag:
                    system_prompt = """당신은 학교 생활규정에 대해 정확하고 친절하게 답변하는 AI 어시스턴트입니다.

다음 규정 문서를 참고하여 학생의 질문에 정확하게 답변하세요.
제공된 문서에 명확한 답변이 없으면 "해당 내용은 제공된 문서에 없습니다"라고 답변하세요.

제공된 규정 문서:
{context}""".format(context=context)
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,  # 정확성 우선
                max_tokens=500
            )
            
            answer = response['choices'][0]['message']['content']
            
            return {
                'answer': answer,
                'references': references,
                'model': 'gpt-3.5-turbo',
                'is_rag': use_rag,
                'use_reranking': use_reranking,
                'retrieved_documents': retrieved_count
            }
        except Exception as e:
            return {
                'answer': f"❌ 답변 생성 실패: {str(e)}",
                'references': [],
                'model': 'gpt-3.5-turbo',
                'is_rag': use_rag,
                'use_reranking': use_reranking,
                'retrieved_documents': retrieved_count
            }
    
    def _rerank_documents(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Re-ranking: 검색된 문서를 쿼리와의 관련성으로 재정렬
        
        간단한 구현: 쿼리 키워드가 많이 포함된 문서를 우선순위
        """
        query_keywords = set(query.lower().split())
        
        for doc in documents:
            content_lower = doc['content'].lower()
            keyword_count = sum(1 for kw in query_keywords if kw in content_lower)
            doc['rerank_score'] = keyword_count + doc['similarity']
        
        documents.sort(key=lambda x: x.get('rerank_score', x['similarity']), reverse=True)
        return documents
    
    def _extract_references(self, documents: List[Dict]) -> List[str]:
        """참고 문서 추출 (청크 내용 자체를 참고문서로 사용)"""
        references = []
        
        for i, doc in enumerate(documents):
            content = doc['content'].strip()
            
            # 청크 내용이 너무 길면 축소
            if len(content) > 300:
                content = content[:300] + "..."
            
            # 참고문서 추가
            references.append(content)
        
        return references
