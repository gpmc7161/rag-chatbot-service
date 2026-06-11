# 개선된 RAG 챗봇 시스템 - PC 설치 및 실행 가이드

학생생활규정 기반 RAG 챗봇을 PC에서 실행하기 위한 완벽한 가이드입니다.

---

## 📋 사전 준비사항

### 1. 필수 소프트웨어 설치

#### Python 3.12 설치 (중요!)
- **다운로드**: [Python 3.12.9](https://www.python.org/downloads/release/python-3129/)
- **설치 시 주의**: "Add Python to PATH" 체크박스 반드시 선택
- **확인**: 터미널에서 `py -3.12 --version` 입력 후 버전 확인

#### Node.js 설치 (프론트엔드용)
- **다운로드**: [Node.js LTS](https://nodejs.org/)
- **설치**: 기본 설정으로 진행
- **확인**: 터미널에서 `node -v` 입력 후 버전 확인

### 2. OpenAI API 키 발급

#### 단계별 가이드
1. [OpenAI Platform](https://platform.openai.com/) 접속
2. 로그인 (ChatGPT 계정과 동일)
3. 왼쪽 메뉴에서 **"API keys"** 클릭
4. **"+ Create new secret key"** 버튼 클릭
5. 키 이름 입력 (예: "MyRAGProject")
6. 생성된 키 복사 및 **안전하게 저장**

#### 결제 설정
1. **Settings > Billing** 메뉴 이동
2. **"Add payment details"** 클릭
3. 신용카드 등록
4. 최소 $5 충전 (약 7,000원)
5. **"Usage limits"**에서 월 최대 금액 설정 (예: $10)

---

## 🚀 설치 및 실행 단계

### 1단계: 프로젝트 폴더 구조 확인

```
rag_chatbot_final/
├── backend/
│   ├── app.py
│   ├── rag_system_final.py
│   ├── run_experiment.py
│   ├── requirements.txt
│   ├── .env.example
│   └── data/
│       └── 학생생활규정.txt
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
└── docs/
    └── FINAL_SETUP_GUIDE.md
```

### 2단계: 백엔드 설정

#### 2-1. 가상 환경 생성
```bash
# 프로젝트 폴더로 이동
cd rag_chatbot_final/backend

# Python 3.12로 가상 환경 생성
py -3.12 -m venv venv

# 가상 환경 활성화
# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

#### 2-2. 의존성 설치
```bash
# 가상 환경이 활성화된 상태에서 (터미널 앞에 (venv) 표시 확인)
pip install -r requirements.txt
```

**설치 중 오류 발생 시:**
- `faiss-cpu` 오류: `pip install faiss-cpu` 단독 실행
- 기타 오류: 각 패키지를 개별 설치
  ```bash
  pip install Flask==2.3.3
  pip install Flask-CORS==4.0.0
  pip install python-dotenv==1.0.0
  pip install openai==0.27.8
  pip install numpy==1.24.3
  pip install sentence-transformers==2.2.2
  ```

#### 2-3. 환경 변수 설정
```bash
# .env.example을 .env로 복사
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

`.env` 파일을 텍스트 에디터로 열어서 수정:
```
OPENAI_API_KEY=sk-...여기에_발급받은_API_키_입력...
```

#### 2-4. 학생생활규정 파일 준비
```bash
# data 폴더 생성 (없으면)
mkdir data

# 학생생활규정 파일을 data 폴더에 저장
# 파일명: 학생생활규정.txt
```

### 3단계: 백엔드 서버 실행

```bash
# 가상 환경이 활성화된 상태에서
python app.py
```

**성공 메시지:**
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### 4단계: 프론트엔드 설정 (새 터미널에서)

```bash
# 프로젝트 폴더로 이동
cd rag_chatbot_final/frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm start
```

**성공 메시지:**
```
Compiled successfully!
You can now view the app in your browser.
Local: http://localhost:3000
```

---

## 🧪 자동 실험 실행

### 배치 실험 스크립트 실행

```bash
# 백엔드 폴더에서 (가상 환경 활성화 상태)
python run_experiment.py
```

**실험 진행 과정:**
1. 학생생활규정 파일 로드
2. RAG 시스템 초기화
3. 20개 질문으로 자동 실험
4. 기본 챗봇 vs RAG 챗봇 vs Re-ranking 비교
5. 결과를 CSV와 JSON으로 저장

**결과 저장 위치:**
```
experiment_results/
├── experiment_results_YYYYMMDD_HHMMSS.csv
└── experiment_results_YYYYMMDD_HHMMSS.json
```

---

## 🎯 사용 방법

### 웹 인터페이스 사용

1. 브라우저에서 `http://localhost:3000` 접속
2. 학생생활규정 파일 업로드
3. 질문 입력
4. 다음 옵션 선택:
   - **임베딩 모델**: OpenAI 또는 Sentence-BERT
   - **Re-ranking**: 토글 버튼으로 활성화/비활성화
5. 답변 확인 및 참고문서 확인

### API 직접 호출

#### 1. RAG 시스템 초기화
```bash
curl -X POST http://localhost:5000/api/initialize \
  -H "Content-Type: application/json" \
  -d '{"embedding_model": "openai"}'
```

#### 2. 문서 업로드
```bash
curl -X POST http://localhost:5000/api/upload-document \
  -F "file=@학생생활규정.txt"
```

#### 3. 질문 및 답변
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "두발 규정이 무엇인가요?",
    "use_rag": true,
    "use_reranking": false,
    "embedding_model": "openai"
  }'
```

#### 4. 기본 vs RAG 비교
```bash
curl -X POST http://localhost:5000/api/compare \
  -H "Content-Type: application/json" \
  -d '{"question": "두발 규정이 무엇인가요?"}'
```

---

## 🐛 문제 해결

### 1. "OPENAI_API_KEY가 설정되지 않았습니다" 오류
**해결:**
- `.env` 파일이 `backend` 폴더에 있는지 확인
- `.env` 파일에 API 키가 올바르게 입력되었는지 확인
- 파일 저장 후 서버 재시작

### 2. "faiss-cpu 설치 실패" 오류
**해결:**
```bash
# 방법 1: 버전 지정 없이 설치
pip install faiss-cpu

# 방법 2: 아나콘다 사용 시
conda install -c pytorch faiss-cpu

# 방법 3: 개별 설치
pip uninstall faiss-cpu
pip install --no-cache-dir faiss-cpu
```

### 3. "포트 5000이 이미 사용 중입니다" 오류
**해결:**
```bash
# 다른 포트로 실행
# app.py의 마지막 줄 수정:
app.run(debug=True, host='0.0.0.0', port=5001)
```

### 4. "문서에 명시되지 않았습니다" 답변이 자주 나옴
**해결:**
- 학생생활규정 파일이 올바르게 업로드되었는지 확인
- 파일 인코딩이 UTF-8인지 확인
- 청크 크기 조정 (rag_system_final.py의 chunk_size 수정)

### 5. 응답이 느린 경우
**해결:**
- OpenAI API 상태 확인
- 네트워크 연결 확인
- 임베딩 모델을 Sentence-BERT로 변경 (더 빠름)

### 6. "참고문서"가 잘못 표시되는 경우
**해결:**
- 학생생활규정 파일의 형식 확인 (제1장, 제1조 형식)
- 파일 내용이 정확한지 확인
- 서버 재시작

---

## 💰 비용 안내

### 예상 비용 (월별)
- **기본 챗봇**: $0 (API 호출 없음)
- **RAG 챗봇 (OpenAI 임베딩)**:
  - 20개 질문 실험: 약 $0.05
  - 월 100개 질문: 약 $0.25
- **RAG 챗봇 (Sentence-BERT)**:
  - 임베딩 비용: $0 (로컬 처리)
  - 답변 생성만: 약 $0.15/월

### 비용 절감 팁
1. **Sentence-BERT 사용**: 임베딩 비용 0원
2. **온도 설정 낮추기**: 더 짧고 정확한 답변 (토큰 절감)
3. **청크 크기 최적화**: 검색 정확도와 비용 균형
4. **배치 처리**: 한 번에 여러 질문 처리

---

## 📊 성능 최적화 팁

### 1. 임베딩 모델 선택
| 모델 | 속도 | 정확도 | 비용 | 추천 |
|------|------|--------|------|------|
| OpenAI | 중간 | 높음 | $0.02/1K | 정확도 중시 |
| Sentence-BERT | 빠름 | 중간 | $0 | 속도 중시 |

### 2. Re-ranking 사용 시기
- ✅ 정확도가 중요한 경우
- ✅ 답변 신뢰도가 낮은 경우
- ❌ 빠른 응답이 필요한 경우

### 3. 청크 크기 조정
```python
# rag_system_final.py의 _chunk_document 함수
chunk_size = 500  # 기본값
# 작게: 정확도 ↑, 속도 ↓
# 크게: 정확도 ↓, 속도 ↑
```

---

## 🔍 디버깅 팁

### 로그 확인
```bash
# 백엔드 로그 (터미널에 출력됨)
# 각 API 호출의 요청/응답 확인

# 프론트엔드 로그 (브라우저 개발자 도구)
# F12 또는 우클릭 > 검사 > Console 탭
```

### API 테스트
```bash
# Postman 또는 curl로 API 직접 테스트
# 각 엔드포인트의 요청/응답 확인
```

---

## 📚 추가 리소스

- [OpenAI API 문서](https://platform.openai.com/docs)
- [Sentence-BERT 문서](https://www.sbert.net/)
- [FAISS 문서](https://github.com/facebookresearch/faiss)
- [Flask 문서](https://flask.palletsprojects.com/)
- [React 문서](https://react.dev/)

---

## 🎓 학습 자료

본 탐구보고서와 함께 참고할 수 있는 한글 자료:

1. **윤여찬, 김수균** (2025). "생성형 AI를 위한 RAG 기술 동향 및 전망"
2. **정천수** (2023). "LLM 애플리케이션 아키텍처를 활용한 생성형 AI 서비스 구현"
3. **김규석 등** (2026). "파이썬으로 만드는 초경량 한국어 LLM 챗봇"

---

## 📞 지원

문제가 발생하거나 질문이 있으면:
1. 위의 "문제 해결" 섹션 확인
2. 터미널 오류 메시지 정확히 읽기
3. 파일 경로 및 인코딩 확인

---

**마지막 업데이트**: 2026년 5월
**버전**: 1.0 (최종 개선 버전)
