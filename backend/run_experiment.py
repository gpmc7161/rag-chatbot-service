#!/usr/bin/env python3
"""
RAG 챗봇 자동 실험 스크립트

학생생활규정 기반 20개 질문으로 자동 실험을 수행하고 결과를 분석합니다.
"""

import os
import sys
import json
import csv
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import openai
from rag_system_final import RAGSystem

load_dotenv()

# 테스트 질문 20개 (학생생활규정 기반)
TEST_QUESTIONS = [
    # 용의복장 관련 (5개)
    "우리 학교 두발 규정은 어떻게 되나요?",
    "학교에서 액세서리 착용이 가능한가요?",
    "체육복 착용 시간은 언제인가요?",
    "교복 외에 다른 옷을 입을 수 있나요?",
    "두발 검사는 언제 하나요?",
    
    # 학습 및 진로 (4개)
    "학교에서 휴대전화를 사용할 수 있나요?",
    "자습시간에는 어떤 활동이 가능한가요?",
    "진로 상담을 받으려면 어떻게 해야 하나요?",
    "도서관 이용 시간은 몇 시부터 몇 시까지인가요?",
    
    # 보건 및 안전 (4개)
    "학교에서 금지된 물품은 무엇인가요?",
    "응급 상황이 발생하면 어디에 신고해야 하나요?",
    "학교 안전 교육은 언제 실시되나요?",
    "흡연이나 음주는 어떻게 처벌되나요?",
    
    # 인성 및 대인관계 (4개)
    "학교폭력이 발생했을 때 신고 절차는 어떻게 되나요?",
    "선후배 사이의 예절은 어떻게 지켜야 하나요?",
    "욕설이나 비속어 사용 시 어떤 조치가 있나요?",
    "따돌림을 당했을 때 누구에게 상담받을 수 있나요?",
    
    # 생활지도 (3개)
    "지각이나 결석 시 어떤 조치가 있나요?",
    "상담을 신청하려면 어떻게 해야 하나요?",
    "훈육이나 훈계 기록은 어떻게 남나요?"
]

def load_regulations(file_path: str) -> str:
    """학생생활규정 파일 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"오류: {file_path} 파일을 찾을 수 없습니다.")
        sys.exit(1)

def run_experiment(rag_system: RAGSystem, output_dir: str = './experiment_results'):
    """자동 실험 실행"""
    
    # 결과 디렉토리 생성
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = []
    
    print("=" * 80)
    print("RAG 챗봇 자동 실험 시작")
    print("=" * 80)
    print(f"총 {len(TEST_QUESTIONS)}개 질문으로 실험을 진행합니다.\n")
    
    for idx, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{idx}/{len(TEST_QUESTIONS)}] 질문: {question}")
        
        try:
            # 기본 챗봇 답변
            print("  → 기본 챗봇 답변 생성 중...")
            basic_result = rag_system.generate_answer(
                question=question,
                use_rag=False,
                use_reranking=False
            )
            
            # RAG 챗봇 답변
            print("  → RAG 챗봇 답변 생성 중...")
            rag_result = rag_system.generate_answer(
                question=question,
                use_rag=True,
                use_reranking=False
            )
            
            # Re-ranking 적용 답변
            print("  → Re-ranking 적용 답변 생성 중...")
            reranking_result = rag_system.generate_answer(
                question=question,
                use_rag=True,
                use_reranking=True
            )
            
            result = {
                'question_id': idx,
                'question': question,
                'category': get_question_category(idx),
                'basic_answer': basic_result['answer'],
                'basic_response_time': basic_result['response_time'],
                'rag_answer': rag_result['answer'],
                'rag_confidence': rag_result['confidence'],
                'rag_response_time': rag_result['response_time'],
                'rag_sources': json.dumps(rag_result['source_documents'], ensure_ascii=False),
                'reranking_answer': reranking_result['answer'],
                'reranking_confidence': reranking_result['confidence'],
                'reranking_response_time': reranking_result['response_time'],
                'reranking_sources': json.dumps(reranking_result['source_documents'], ensure_ascii=False)
            }
            
            results.append(result)
            print(f"  ✓ 완료 (RAG 신뢰도: {rag_result['confidence']:.2f})\n")
            
        except Exception as e:
            print(f"  ✗ 오류: {str(e)}\n")
            result = {
                'question_id': idx,
                'question': question,
                'category': get_question_category(idx),
                'error': str(e)
            }
            results.append(result)
        
        time.sleep(1)  # API 호출 간격
    
    # 결과 저장
    save_results(results, output_dir, timestamp)
    
    # 통계 출력
    print_statistics(results)
    
    print("\n" + "=" * 80)
    print(f"실험 완료! 결과가 {output_dir}에 저장되었습니다.")
    print("=" * 80)

def get_question_category(question_id: int) -> str:
    """질문 ID로 카테고리 반환"""
    if 1 <= question_id <= 5:
        return "용의복장"
    elif 6 <= question_id <= 9:
        return "학습및진로"
    elif 10 <= question_id <= 13:
        return "보건및안전"
    elif 14 <= question_id <= 17:
        return "인성및대인관계"
    else:
        return "생활지도"

def save_results(results: list, output_dir: str, timestamp: str):
    """결과를 CSV와 JSON으로 저장"""
    
    # CSV 저장
    csv_file = os.path.join(output_dir, f'experiment_results_{timestamp}.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    # JSON 저장
    json_file = os.path.join(output_dir, f'experiment_results_{timestamp}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ CSV 저장: {csv_file}")
    print(f"✓ JSON 저장: {json_file}")

def print_statistics(results: list):
    """통계 출력"""
    print("\n" + "=" * 80)
    print("실험 통계")
    print("=" * 80)
    
    total = len(results)
    successful = sum(1 for r in results if 'error' not in r)
    failed = total - successful
    
    print(f"\n총 질문 수: {total}")
    print(f"성공: {successful}")
    print(f"실패: {failed}")
    
    if successful > 0:
        # RAG 신뢰도 평균
        rag_confidences = [r['rag_confidence'] for r in results if 'rag_confidence' in r]
        if rag_confidences:
            avg_confidence = sum(rag_confidences) / len(rag_confidences)
            print(f"\nRAG 평균 신뢰도: {avg_confidence:.2f}")
        
        # 응답 시간 비교
        basic_times = [r['basic_response_time'] for r in results if 'basic_response_time' in r]
        rag_times = [r['rag_response_time'] for r in results if 'rag_response_time' in r]
        
        if basic_times and rag_times:
            avg_basic_time = sum(basic_times) / len(basic_times)
            avg_rag_time = sum(rag_times) / len(rag_times)
            print(f"\n기본 챗봇 평균 응답 시간: {avg_basic_time:.2f}초")
            print(f"RAG 챗봇 평균 응답 시간: {avg_rag_time:.2f}초")
        
        # 카테고리별 통계
        print("\n카테고리별 통계:")
        categories = {}
        for r in results:
            cat = r.get('category', '기타')
            if cat not in categories:
                categories[cat] = {'total': 0, 'success': 0}
            categories[cat]['total'] += 1
            if 'error' not in r:
                categories[cat]['success'] += 1
        
        for cat, stats in categories.items():
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {cat}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

if __name__ == '__main__':
    # 학생생활규정 파일 경로
    regulations_file = './data/학생생활규정.txt'
    
    # 파일이 없으면 생성
    if not os.path.exists(regulations_file):
        print(f"오류: {regulations_file} 파일이 필요합니다.")
        print("학생생활규정 파일을 ./data/ 디렉토리에 저장해주세요.")
        sys.exit(1)
    
    # 학생생활규정 로드
    print("학생생활규정 로드 중...")
    regulations = load_regulations(regulations_file)
    
    # RAG 시스템 초기화
    print("RAG 시스템 초기화 중...")
    rag_system = RAGSystem(embedding_model='openai')
    
    # 문서 추가
    print("문서 처리 중...")
    rag_system.add_document(regulations)
    
    # 실험 실행
    run_experiment(rag_system)
