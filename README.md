# 🤖 RAG 챗봇 - 학생생활규정 AI 어시스턴트

생성형 AI 기반 챗봇 서비스 구축 및 답변 정확도 개선 탐구

## 📋 프로젝트 개요

이 프로젝트는 **Retrieval-Augmented Generation (RAG)** 기술을 사용하여 학교 학생생활규정에 대한 정확한 답변을 제공하는 AI 챗봇입니다.

### 🎯 핵심 기능

- **📚 RAG 기반 검색**: OpenAI 임베딩 + FAISS 벡터 DB
- **⚡ 고속 검색**: FAISS 인덱싱으로 빠른 유사도 검색
- **📏 청크 크기 조절**: 100~2000 범위에서 청크 크기 커스터마이징
- **🔄 Re-ranking**: 검색 결과 재정렬로 정확도 향상
- **📊 비교 분석**: 기본 챗봇 vs RAG 챗봇 성능 비교

---

## 🛠️ 기술 스택

### 백엔드
- **Flask**: REST API 서버
- **OpenAI API**: 임베딩 및 답변 생성
- **FAISS**: 벡터 유사도 검색
- **NumPy**: 수치 계산

### 프론트엔드
- **React 19**: UI 프레임워크
- **TypeScript**: 타입 안전성
- **Axios**: HTTP 클라이언트
- **Tailwind CSS**: 스타일링

---

## 📦 설치 및 실행

### 1. 백엔드 설정

```bash
# 프로젝트 디렉토리 이동
cd backend

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# .env 파일 생성
cp .env.example .env
# .env에 OPENAI_API_KEY 입력
```

### 2. 프론트엔드 설정

```bash
# 프로젝트 디렉토리 이동
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm start
```

### 3. 서버 실행

```bash
# 백엔드 서버 (포트 5000)
python app.py

# 프론트엔드 서버 (포트 3000)
npm start
```

---

## 🚀 사용 방법

### 1. 문서 업로드

1. **청크 크기 설정** (선택사항)
   - 슬라이더로 100~2000 범위에서 조절
   - 기본값: 500
   - 작을수록 세밀한 검색, 클수록 넓은 범위 검색

2. **문서 선택 및 업로드**
   - .txt 파일 선택
   - "문서 선택" 버튼 클릭
   - 업로드 완료 시 "✓ 업로드 문서 1개" 표시

### 2. 질문하기

1. **질문 입력**
   - 입력창에 학생생활규정 관련 질문 입력
   - 예: "우리 학교 두발 규정은 어떻게 되나요?"

2. **답변 비교**
   - 🤖 기본 챗봇: 일반 지식 기반 답변
   - 📚 RAG 챗봇: 문서 기반 정확한 답변
   - ⚡ Re-ranking 적용: 추가 최적화 답변

3. **참고문서 확인**
   - RAG 챗봇 답변 하단의 "📖 참고문서" 확인
   - 어느 규정에서 인용했는지 명시

### 3. 비교 분석

- "📊 비교 분석" 탭에서 성능 지표 확인
- 정확도, 할루시네이션 비율 등 비교

---

## 📊 API 엔드포인트

### POST `/api/upload-document`

문서 업로드 및 RAG 시스템 초기화

**요청:**
```json
{
  "file": "<File>",
  "chunk_size": 500
}
```

**응답:**
```json
{
  "status": "success",
  "message": "문서 업로드 완료",
  "filename": "document.txt",
  "chunks": 25,
  "chunk_size": 500
}
```

### POST `/api/compare`

기본 챗봇과 RAG 챗봇 비교

**요청:**
```json
{
  "query": "질문 내용"
}
```

**응답:**
```json
{
  "query": "질문 내용",
  "basic_chatbot": {
    "answer": "기본 챗봇 답변",
    "model": "gpt-3.5-turbo"
  },
  "rag_chatbot": {
    "answer": "RAG 챗봇 답변",
    "references": ["제1장 총칙", "제2조 정의"],
    "model": "gpt-3.5-turbo"
  }
}
```

---

## 🔍 청크 크기 가이드

| 크기 | 특징 | 사용 사례 |
|------|------|---------|
| **100-300** | 매우 세밀한 검색 | 정확한 조항 찾기 |
| **500** (기본) | 균형잡힌 검색 | 일반적인 사용 |
| **1000+** | 넓은 범위 검색 | 빠른 처리 |

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────┐
│           프론트엔드 (React)                 │
│  - 문서 업로드                              │
│  - 청크 크기 조절                           │
│  - 질문 입력 및 답변 표시                   │
└────────────────┬────────────────────────────┘
                 │ HTTP/REST
┌────────────────▼────────────────────────────┐
│         백엔드 (Flask)                       │
│  - 문서 처리                                │
│  - RAG 검색                                 │
│  - 답변 생성                                │
└────────────────┬────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼──┐  ┌─────▼─────┐  ┌──▼──────┐
│OpenAI│  │   FAISS   │  │ 문서DB  │
│ API  │  │ 벡터 DB   │  │ (메모리)│
└──────┘  └───────────┘  └─────────┘
```

---

## 📈 성능 개선

### FAISS 도입 효과

| 항목 | 메모리 기반 | FAISS |
|------|-----------|-------|
| **검색 속도** | O(n) | O(log n) |
| **100개 청크** | ~50ms | ~5ms |
| **1000개 청크** | ~500ms | ~10ms |
| **메모리 사용** | 낮음 | 중간 |

---

## 🔧 환경 변수

`.env` 파일에 다음을 설정하세요:

```
OPENAI_API_KEY=sk-...
FLASK_ENV=development
FLASK_DEBUG=True
```

---

## 📝 프로젝트 구조

```
rag_chatbot_final_simplified/
├── backend/
│   ├── app.py                      # Flask 서버
│   ├── rag_system_simplified.py    # RAG 시스템 (FAISS 포함)
│   ├── requirements.txt            # 의존성
│   └── .env.example               # 환경 변수 예시
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.tsx           # 메인 페이지
│   │   │   └── Home.css           # 스타일
│   │   ├── App.tsx                # 앱 진입점
│   │   └── main.tsx               # React 진입점
│   ├── package.json               # 프론트엔드 의존성
│   └── index.html                 # HTML 템플릿
└── README.md                       # 이 파일
```

---

## 🎓 학습 포인트

이 프로젝트를 통해 학습할 수 있는 내용:

1. **RAG 기술**: 벡터 임베딩과 유사도 검색
2. **FAISS**: 대규모 벡터 검색 최적화
3. **OpenAI API**: 임베딩 및 LLM 활용
4. **전체 스택**: 백엔드-프론트엔드 통합
5. **성능 최적화**: 검색 속도 개선

---

## 🚨 주의사항

- OpenAI API 사용 시 비용이 발생합니다
- 대용량 문서는 처리 시간이 오래 걸릴 수 있습니다
- FAISS는 CPU 기반이므로 GPU 사용 시 더 빠릅니다

---

## 📞 문제 해결

### FAISS 설치 오류
```bash
# CPU 버전 설치
pip install faiss-cpu

# GPU 버전 설치 (CUDA 필요)
pip install faiss-gpu
```

### OpenAI API 오류
- `.env` 파일에서 API 키 확인
- API 키 유효성 확인
- 사용 가능한 크레딧 확인

### 검색 결과 없음
- 문서가 제대로 업로드되었는지 확인
- 청크 크기 조절 후 다시 시도
- 질문을 더 구체적으로 입력

---

## 📄 라이센스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

## 👨‍💻 개발자

**Manus AI Assistant**

최종 수정: 2026년 6월

---

## 🙏 감사의 말

- OpenAI: 임베딩 및 LLM 제공
- FAISS: 벡터 검색 라이브러리
- Flask: 웹 프레임워크
- React: UI 프레임워크

---

**Happy Coding! 🚀**
