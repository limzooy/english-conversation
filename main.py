#!/usr/bin/env python3
"""
영어 회화 연습 프로그램
- OpenAI Whisper: 음성 인식
- OpenAI GPT-4o: 대화 + 문법 교정
- OpenAI TTS: AI 음성 출력
- Google Sheets / 로컬 CSV: 대화 기록 저장
"""

import os
import sys
import uuid
import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DIVIDER = "─" * 60


def print_header():
    print(f"\n{'='*60}")
    print("  🗣️  영어 회화 연습 프로그램")
    print(f"{'='*60}")
    print("  • 마이크로 영어를 말하거나 텍스트로 입력하세요")
    print("  • AI가 대화 상대가 되어 문법도 교정해 드립니다")
    print("  • 대화 내용은 자동으로 기록됩니다")
    print(f"{'='*60}\n")


def print_correction(result: dict):
    if result["has_correction"]:
        print(f"\n  📝 문법 교정")
        print(f"  {DIVIDER[:40]}")
        print(f"  원문:  {result['original']}")
        print(f"  수정:  {result['corrected']}")
        if result["explanation"]:
            print(f"  설명:  {result['explanation']}")

    if result.get("pronunciation_tip"):
        print(f"\n  🔊 발음 팁: {result['pronunciation_tip']}")


def get_input_mode() -> str:
    print(f"\n  [Enter] 마이크 입력   [t] 텍스트 입력   [q] 종료   [s] 세션 요약")
    choice = input("  선택 > ").strip().lower()
    return choice


def print_session_summary(conversation_history: list, session_id: str):
    print(f"\n{'='*60}")
    print(f"  📊 세션 요약 (ID: {session_id})")
    print(f"{'='*60}")
    total_turns = len([m for m in conversation_history if m["role"] == "user"])
    print(f"  총 대화 횟수: {total_turns}회")
    print(f"{'='*60}\n")


def main():
    print_header()

    # API 키 확인
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY=sk-... 를 추가하세요.")
        sys.exit(1)

    client = OpenAI(api_key=openai_api_key)

    # 모듈 초기화
    from ai_handler import AIHandler
    from speech_handler import SpeechHandler
    from local_saver import LocalSaver

    ai = AIHandler(client)
    speech = SpeechHandler(client)

    # 저장소 초기화 (Google Sheets 또는 로컬 CSV)
    saver = None
    spreadsheet_id = os.getenv("SPREADSHEET_ID")
    credentials_file = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")

    if spreadsheet_id and os.path.exists(credentials_file):
        try:
            from sheets_handler import SheetsHandler
            saver = SheetsHandler(credentials_file, spreadsheet_id)
            print("✅ Google Sheets 연결 완료")
        except Exception as e:
            print(f"⚠️  Google Sheets 연결 실패: {e}")
            print("   로컬 CSV 파일에 저장합니다.")

    if saver is None:
        saver = LocalSaver(directory=os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversation_history.csv")
        print(f"💾 대화 기록 저장 위치: {csv_path}")

    session_id = str(uuid.uuid4())[:8].upper()
    conversation_history = []

    print(f"\n  세션 ID: {session_id}")

    # AI 인사말
    print("\n  AI 로딩 중...")
    greeting = ai.get_initial_greeting()
    print(f"\n  🤖 AI: {greeting['response']}")
    speech.speak(greeting["response"])

    # 메인 대화 루프
    while True:
        choice = get_input_mode()

        if choice in ("q", "quit", "exit"):
            print_session_summary(conversation_history, session_id)
            print("  수고하셨습니다! 영어 실력이 쑥쑥 늘 거예요! 👏\n")
            break

        if choice == "s":
            print_session_summary(conversation_history, session_id)
            continue

        # 사용자 입력 받기
        user_text = ""
        if choice == "t":
            user_text = input("  영어로 입력 > ").strip()
        else:
            user_text = speech.listen()
            if user_text:
                print(f"\n  🎤 인식된 문장: {user_text}")
            else:
                print("  음성을 인식하지 못했습니다. 다시 시도해 주세요.")
                continue

        if not user_text:
            continue

        if user_text.lower() in ("quit", "exit", "bye"):
            print_session_summary(conversation_history, session_id)
            print("  수고하셨습니다! 👏\n")
            break

        # AI 응답 생성
        print("\n  💭 생각 중...")
        try:
            result = ai.chat(user_text, conversation_history)
        except Exception as e:
            print(f"  ❌ AI 응답 오류: {e}")
            continue

        # 결과 출력
        print(f"\n  🤖 AI: {result['response']}")
        print_correction(result)

        # AI 음성 출력
        speech.speak(result["response"])

        # 대화 기록 업데이트
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": result["response"]})

        # 저장
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            saver.append_row(
                session_id=session_id,
                timestamp=timestamp,
                user_sentence=user_text,
                corrected_sentence=result.get("corrected", ""),
                correction_explanation=result.get("explanation", ""),
                ai_response=result["response"],
                pronunciation_tip=result.get("pronunciation_tip", ""),
                has_correction=result["has_correction"],
            )
        except Exception as e:
            print(f"  ⚠️  저장 실패: {e}")


if __name__ == "__main__":
    main()
