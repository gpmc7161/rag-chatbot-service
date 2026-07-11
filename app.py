"""
간단한 RAG 챗봇 백엔드 (OpenAI만 사용)
- Sentence-BERT 제거로 CPU 사용량 대폭 감소
- 빠른 응답
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from rag_system_simplified import RAGSystem
import json
import time
import zipfile
import xml.etree.ElementTree as ET
import re

load_dotenv()

app = Flask(__name__)
CORS(app)

# 전역 RAG 시스템 인스턴스
rag_system = None
current_document_path = None

# 데이터 폴더 생성
os.makedirs('data', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

def _convert_hwp_to_text(hwp_path: str) -> str:
    """HWP 파일을 텍스트로 변환"""
    try:
        with zipfile.ZipFile(hwp_path, 'r') as hwp_zip:
            xml_files = [f for f in hwp_zip.namelist() if 'Section' in f and f.endswith('.xml')]
            if not xml_files:
                xml_files = [f for f in hwp_zip.namelist() if f.endswith('.xml')]
            
            text_content = ""
            for xml_file in xml_files:
                try:
                    xml_data = hwp_zip.read(xml_file)
                    text = re.sub(r'<[^>]+>', '', xml_data.decode('utf-8', errors='ignore'))
                    text = re.sub(r'\s+', ' ', text)
                    text_content += text + "\n"
                except:
                    pass
            
            return text_content.strip()
    except Exception as e:
        print(f"⚠️  HWP 변환 실패: {str(e)}")
        return ""

def _auto_load_document():
    """서버 시작 시 uploads 폴더에서 문서 자동 로드"""
    global rag_system, current_document_path
    
    try:
        # uploads 폴더에서 txt 또는 hwp 파일 찾기
        uploads_dir = 'uploads'
        if os.path.exists(uploads_dir):
            files = [f for f in os.listdir(uploads_dir) if f.endswith(('.txt', '.hwp'))]
            
            if files:
                # 가장 최근 파일 선택
                file_path = os.path.join(uploads_dir, files[0])
                file_size = os.path.getsize(file_path)
                
                print(f"\n📄 자동 문서 로드 중...")
                print(f"   - 파일: {files[0]}")
                print(f"   - 크기: {file_size / 1024:.1f} KB")
                
                # HWP 파일이면 변환
                if files[0].endswith('.hwp'):
                    print(f"   - HWP 파일 변환 중...")
                    text_content = _convert_hwp_to_text(file_path)
                    # 임시 txt 파일로 저장
                    temp_txt_path = os.path.join(uploads_dir, 'temp_converted.txt')
                    with open(temp_txt_path, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    file_path = temp_txt_path
                    print(f"   - 변환 완료")
                
                # RAG 시스템 초기화 (기본 청크 크기 500)
                chunk_size = 500
                rag_system = RAGSystem(file_path, chunk_size=chunk_size)
                current_document_path = file_path
                
                print(f"✅ 문서 자동 로드 완료")
                print(f"   - 청크 수: {len(rag_system.chunks)}")
                print(f"   - 청크 크기: {chunk_size}")
                return True
    except Exception as e:
        print(f"⚠️  문서 자동 로드 실패: {str(e)}")
    
    return False

@app.route('/api/health', methods=['GET'])
def health():
    """헬스 체크"""
    return jsonify({
        'status': 'ok',
        'rag_initialized': rag_system is not None,
        'document_loaded': current_document_path is not None
    })

@app.route('/api/upload-document', methods=['POST'])
def upload_document():
    """문서 업로드 및 RAG 시스템 초기화"""
    global rag_system, current_document_path
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 없습니다'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '파일을 선택해주세요'}), 400
        
        # 청크 크기 파라미터 (기본값: 500)
        chunk_size = request.form.get('chunk_size', 500, type=int)
        chunk_size = max(100, min(2000, chunk_size))
        
        # 파일 저장
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        print(f"\n📄 문서 업로드: {file.filename}")
        print(f"📍 경로: {file_path}")
        print(f"📍 청크 크기: {chunk_size}")
        
        # HWP 파일이면 변환
        if file.filename.endswith('.hwp'):
            print(f"   - HWP 파일 변환 중...")
            text_content = _convert_hwp_to_text(file_path)
            # 임시 txt 파일로 저장
            temp_txt_path = os.path.join('uploads', 'temp_converted.txt')
            with open(temp_txt_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            file_path = temp_txt_path
            print(f"   - 변환 완료")
        
        # RAG 시스템 초기화 (OpenAI만 사용)
        print("🔄 RAG 시스템 초기화 중...")
        rag_system = RAGSystem(file_path, chunk_size=chunk_size)
        current_document_path = file_path
        
        return jsonify({
            'status': 'success',
            'message': '문서 업로드 완료',
            'filename': file.filename,
            'chunks': len(rag_system.chunks),
            'chunk_size': chunk_size
        })
    
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    챗봇 응답 생성
    
    Request:
    {
        'query': '질문',
        'use_rag': true/false,
        'use_reranking': true/false
    }
    """
    global rag_system
    
    try:
        if rag_system is None:
            return jsonify({'error': '문서가 로드되지 않았습니다. 먼저 문서를 업로드하세요.'}), 400
        
        data = request.json
        query = data.get('query', '')
        use_rag = data.get('use_rag', True)
        use_reranking = data.get('use_reranking', False)
        
        if not query:
            return jsonify({'error': '질문을 입력해주세요'}), 400
        
        print(f"\n💬 질문: {query}")
        print(f"   - RAG 사용: {use_rag}")
        print(f"   - Re-ranking: {use_reranking}")
        
        # 답변 생성
        result = rag_system.generate_answer(
            query=query,
            use_rag=use_rag,
            use_reranking=use_reranking
        )
        
        print(f"✅ 답변 생성 완료")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare():
    """
    기본 챗봇과 RAG 챗봇 비교
    자동 청크 크기 조절 기능 포함
    
    Request:
    {
        'query': '질문'
    }
    """
    global rag_system
    
    try:
        if rag_system is None:
            return jsonify({'error': '문서가 로드되지 않았습니다'}), 400
        
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': '질문을 입력해주세요'}), 400
        
        print(f"\n📊 비교 분석: {query}")
        
        # 자동 청크 크기 조절 함수
        def auto_adjust_chunk_size(total_chunks, query_length):
            """
            토큰 제한을 고려한 자동 청크 크기 조절
            - 최대 컨텍스트: 16,385 토큰
            - 안전 마진: 30% (4,915 토큰)
            - 사용 가능: 약 11,470 토큰
            - 평균 토큰: 1 글자 ≈ 0.3 토큰
            """
            MAX_TOKENS = 16385
            SAFETY_MARGIN = 0.3  # 30% 안전 마진
            AVAILABLE_TOKENS = int(MAX_TOKENS * (1 - SAFETY_MARGIN))
            
            # 쿼리 토큰 (평균 0.3 토큰/글자)
            query_tokens = int(query_length * 0.3)
            
            # 참고문서용 남은 토큰
            available_for_docs = AVAILABLE_TOKENS - query_tokens - 1000  # 1000: 답변 생성용
            
            # 검색할 청크 수 (기본 3개)
            num_chunks = 3
            
            # 청크당 평균 토큰
            tokens_per_chunk = available_for_docs // num_chunks
            
            # 글자 수로 변환 (1 글자 ≈ 3.3 토큰)
            chars_per_chunk = int(tokens_per_chunk / 0.3)
            
            # 안전한 청크 크기 계산 (최소 100, 최대 2000)
            safe_chunk_size = max(100, min(2000, chars_per_chunk))
            
            print(f"   - 쿼리 토큰: {query_tokens}")
            print(f"   - 참고문서 토큰: {available_for_docs}")
            print(f"   - 권장 청크 크기: {safe_chunk_size}")
            
            return safe_chunk_size
        
        # 자동 청크 크기 조절
        recommended_chunk_size = auto_adjust_chunk_size(
            len(rag_system.chunks),
            len(query)
        )
        
        # 현재 청크 크기가 너무 크면 조절
        if rag_system.chunk_size > recommended_chunk_size:
            print(f"   ⚠️  청크 크기 자동 조절: {rag_system.chunk_size} → {recommended_chunk_size}")
            # 새로운 청크 크기로 재처리
            old_chunk_size = rag_system.chunk_size
            rag_system.chunk_size = recommended_chunk_size
            rag_system._chunk_document()
            print(f"   ✅ 청크 재생성 완료: {len(rag_system.chunks)}개")
        
        # 기본 챗봇 (RAG 없음)
        print("   - 기본 챗봇 생성 중...")
        basic_start = time.time()
        basic_result = rag_system.generate_answer(query, use_rag=False)
        basic_time = int((time.time() - basic_start) * 1000)
        
        # RAG 챗봇
        print("   - RAG 챗봇 생성 중...")
        rag_start = time.time()
        rag_result = rag_system.generate_answer(query, use_rag=True)
        rag_time = int((time.time() - rag_start) * 1000)
        
        print(f"   - 기본 챗봇 응답시간: {basic_time}ms")
        print(f"   - RAG 챗봇 응답시간: {rag_time}ms")
        
        return jsonify({
            'query': query,
            'basic_chatbot': {
                'answer': basic_result['answer'],
                'model': basic_result['model'],
                'response_time': basic_time
            },
            'rag_chatbot': {
                'answer': rag_result['answer'],
                'references': rag_result['references'],
                'model': rag_result['model'],
                'response_time': rag_time
            },
            'chunk_size_adjusted': rag_system.chunk_size
        })
    
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/experiment', methods=['POST'])
def run_experiment():
    """
    자동 실험 실행 (20개 질문)
    
    Request:
    {
        'questions': ['질문1', '질문2', ...]
    }
    """
    global rag_system
    
    try:
        if rag_system is None:
            return jsonify({'error': '문서가 로드되지 않았습니다'}), 400
        
        data = request.json
        questions = data.get('questions', [])
        
        if not questions:
            return jsonify({'error': '질문 목록이 없습니다'}), 400
        
        print(f"\n🔬 실험 시작: {len(questions)}개 질문")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"   [{i}/{len(questions)}] {question}")
            
            # 기본 챗봇
            basic = rag_system.generate_answer(question, use_rag=False)
            
            # RAG 챗봇
            rag = rag_system.generate_answer(question, use_rag=True)
            
            results.append({
                'question': question,
                'basic_answer': basic['answer'],
                'rag_answer': rag['answer'],
                'references': rag['references']
            })
        
        print(f"✅ 실험 완료: {len(results)}개 결과")
        
        return jsonify({
            'status': 'success',
            'total_questions': len(questions),
            'results': results
        })
    
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/questions', methods=['GET'])
def get_questions():
    """학생생활규정 기반 20개 질문 반환"""
    questions = [
        # 용의복장 관련 (5개)
        "우리 학교 두발 규정은 어떻게 되나요?",
        "교복 착용 시 주의사항이 있나요?",
        "액세서리 착용이 가능한가요?",
        "체육복 착용 규정을 알려주세요",
        "계절에 따른 복장 규정이 있나요?",
        
        # 학습 및 진로 (4개)
        "학교에서 휴대전화 사용이 가능한가요?",
        "자습시간 규정은 어떻게 되나요?",
        "진로 상담을 받으려면 어떻게 해야 하나요?",
        "도서관 이용 규정을 알려주세요",
        
        # 보건 및 안전 (4개)
        "금지 물품이 무엇인가요?",
        "응급 상황 발생 시 신고 절차는?",
        "학교 안전 교육은 언제 실시되나요?",
        "보건실 이용 방법을 알려주세요",
        
        # 인성 및 대인관계 (4개)
        "학교폭력 신고 방법은?",
        "선후배 예절 규정이 있나요?",
        "욕설이나 비속어 사용 시 조치는?",
        "따돌림 신고는 어디로 해야 하나요?",
        
        # 생활지도 (3개)
        "지각이나 결석 시 처리 절차는?",
        "상담 신청은 어떻게 하나요?",
        "훈육과 훈계의 차이는?"
    ]
    
    return jsonify({'questions': questions})

# 앱 초기화 시 자동 로드 (gunicorn 환경에서도 동작)
_auto_load_document()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"\n🚀 Flask 서버 시작")
    print(f"📍 http://localhost:{port}" )
    print(f"🔧 모드: {os.getenv('FLASK_ENV', 'production')}")
    print(f"📝 임베딩: OpenAI")
    
    print()
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.getenv('FLASK_ENV') == 'development'
    )
