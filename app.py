"""영어 회화 웹 앱 - Flask 백엔드"""

import os
import csv
import json
import uuid
import datetime
import tempfile

from flask import Flask, render_template, request, jsonify, session, send_file
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

# Google Sheets DB (SPREADSHEET_ID 환경변수가 있을 때만 활성화)
_db = None

def get_db():
    global _db
    if _db is None and os.environ.get("SPREADSHEET_ID"):
        try:
            from sheets_db import SheetsDB
            _db = SheetsDB(os.environ["SPREADSHEET_ID"])
        except Exception as e:
            app.logger.error(f"Sheets DB init failed: {e}")
    return _db

BASE_DIR = os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
CONVERSATION_CSV = os.path.join(BASE_DIR, "conversation_history.csv")
MEMO_CSV = os.path.join(BASE_DIR, "memos.csv")

CONV_HEADERS = ["날짜/시간", "세션 ID", "내 영어 문장", "수정된 문장", "수정 필요", "수정 설명", "AI 응답", "발음 팁"]
MEMO_HEADERS = ["날짜", "메모"]
TRAINING_CSV = os.path.join(BASE_DIR, "training_progress.csv")
TRAINING_HEADERS = ["날짜", "요일번호", "완료문장수", "정답수"]
TRAINING_SENTENCES_CSV = os.path.join(BASE_DIR, "training_sentences.csv")
TRAINING_SENTENCES_HEADERS = ["요일번호", "한국어문장", "평가결과"]
PHRASE_PROGRESS_CSV = os.path.join(BASE_DIR, "phrase_progress.csv")
PHRASE_PROGRESS_HEADERS = ["표현", "카테고리", "완료날짜"]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def init_csvs():
    if not os.path.exists(CONVERSATION_CSV):
        with open(CONVERSATION_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(CONV_HEADERS)
    if not os.path.exists(MEMO_CSV):
        with open(MEMO_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(MEMO_HEADERS)
    if not os.path.exists(TRAINING_CSV):
        with open(TRAINING_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(TRAINING_HEADERS)
    if not os.path.exists(TRAINING_SENTENCES_CSV):
        with open(TRAINING_SENTENCES_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(TRAINING_SENTENCES_HEADERS)
    if not os.path.exists(PHRASE_PROGRESS_CSV):
        with open(PHRASE_PROGRESS_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(PHRASE_PROGRESS_HEADERS)


init_csvs()


def get_conversations_for_date(date_str):
    db = get_db()
    if db:
        return db.get_conversations_for_date(date_str)
    rows = []
    try:
        with open(CONVERSATION_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["날짜/시간"].startswith(date_str):
                    rows.append(dict(row))
    except FileNotFoundError:
        pass
    return rows


def get_dates_with_conversations():
    db = get_db()
    if db:
        return db.get_dates_with_conversations()
    dates = set()
    try:
        with open(CONVERSATION_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = row["날짜/시간"][:10]
                if date:
                    dates.add(date)
    except FileNotFoundError:
        pass
    return sorted(list(dates))


DAILY_WORDS_FILE = os.path.join(BASE_DIR, "daily_words.json")
SHOWN_WORDS_FILE = os.path.join(BASE_DIR, "shown_words.json")


def load_shown_words() -> list:
    db = get_db()
    if db:
        cached = db.get_cache("shown_words")
        return json.loads(cached) if cached else []
    if os.path.exists(SHOWN_WORDS_FILE):
        with open(SHOWN_WORDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_shown_words(new_words: list):
    db = get_db()
    existing = load_shown_words()
    combined = list(set(existing + [w.lower() for w in new_words]))
    if db:
        db.set_cache("shown_words", json.dumps(combined, ensure_ascii=False))
        return
    with open(SHOWN_WORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False)


def get_daily_words():
    """오늘 날짜의 비즈니스 단어/표현 반환 (하루 1회 생성 후 캐시, 중복 제외)"""
    today = datetime.date.today().isoformat()

    # 캐시 확인
    db = get_db()
    if db:
        cached_date = db.get_cache("daily_words_date")
        if cached_date == today:
            cached_data = db.get_cache("daily_words_json")
            if cached_data:
                return json.loads(cached_data)
    elif os.path.exists(DAILY_WORDS_FILE):
        with open(DAILY_WORDS_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("date") == today:
            return cached["data"]

    # 이전에 나온 단어 목록
    shown = load_shown_words()
    exclude_str = ", ".join(shown) if shown else "없음"

    # 새로 생성
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"""Give me 5 useful business English words and 2 business expressions for today.
IMPORTANT: Do NOT use any of these words (already shown before): {exclude_str}
Choose completely different words each time.
Respond ONLY in this JSON format:
{{
  "words": [
    {{"word": "...", "part": "noun/verb/adj", "meaning": "한국어 뜻", "example": "short example sentence"}},
    {{"word": "...", "part": "...", "meaning": "...", "example": "..."}},
    {{"word": "...", "part": "...", "meaning": "...", "example": "..."}},
    {{"word": "...", "part": "...", "meaning": "...", "example": "..."}},
    {{"word": "...", "part": "...", "meaning": "...", "example": "..."}}
  ],
  "expressions": [
    {{"phrase": "...", "meaning": "한국어 뜻", "example": "short example sentence"}},
    {{"phrase": "...", "meaning": "한국어 뜻", "example": "short example sentence"}}
  ]
}}"""}],
            response_format={"type": "json_object"},
            temperature=0.9,
        )
        data = json.loads(response.choices[0].message.content)
    except Exception:
        data = {
            "words": [
                {"word": "leverage", "part": "verb", "meaning": "활용하다", "example": "We can leverage our network."},
                {"word": "streamline", "part": "verb", "meaning": "간소화하다", "example": "Let's streamline the process."},
                {"word": "deliverable", "part": "noun", "meaning": "산출물", "example": "What are the key deliverables?"},
                {"word": "alignment", "part": "noun", "meaning": "합의 / 일치", "example": "We need alignment on the strategy."},
                {"word": "bandwidth", "part": "noun", "meaning": "여유 시간 / 처리 능력", "example": "Do you have the bandwidth for this?"},
            ],
            "expressions": [
                {"phrase": "touch base", "meaning": "잠깐 연락하다", "example": "Let's touch base next week."},
                {"phrase": "circle back", "meaning": "나중에 다시 돌아오다", "example": "I'll circle back once I have the data."},
            ]
        }

    # 오늘 나온 단어 누적 저장 (중복 방지용)
    today_words = [w["word"] for w in data.get("words", [])]
    save_shown_words(today_words)

    # 캐시 저장
    if db:
        db.set_cache("daily_words_date", today)
        db.set_cache("daily_words_json", json.dumps(data, ensure_ascii=False))
    else:
        with open(DAILY_WORDS_FILE, "w", encoding="utf-8") as f:
            json.dump({"date": today, "data": data}, f, ensure_ascii=False)
    return data


def load_recent_context(n=30):
    db = get_db()
    if db:
        return db.load_recent_context(n)
    rows = []
    try:
        with open(CONVERSATION_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        return []
    recent = rows[-n:] if len(rows) > n else rows
    messages = []
    for row in recent:
        user_text = row.get("내 영어 문장", "").strip()
        ai_text = row.get("AI 응답", "").strip()
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if ai_text:
            messages.append({"role": "assistant", "content": ai_text})
    return messages


def get_memo(date_str):
    db = get_db()
    if db:
        return db.get_memo(date_str)
    try:
        with open(MEMO_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["날짜"] == date_str:
                    return row["메모"]
    except FileNotFoundError:
        pass
    return ""


def save_memo(date_str, memo_text):
    db = get_db()
    if db:
        db.save_memo(date_str, memo_text)
        return
    rows = []
    found = False
    try:
        with open(MEMO_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["날짜"] == date_str:
                    row["메모"] = memo_text
                    found = True
                rows.append(dict(row))
    except FileNotFoundError:
        pass
    if not found:
        rows.append({"날짜": date_str, "메모": memo_text})
    with open(MEMO_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=MEMO_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def get_training_status():
    """현재 훈련 day 번호와 오늘 진행 상황 반환"""
    today = datetime.date.today().isoformat()
    db = get_db()
    if db:
        rows = db.get_training_rows()
    else:
        rows = []
        try:
            with open(TRAINING_CSV, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except FileNotFoundError:
            pass

    completed_days = sum(1 for r in rows if int(r.get("완료문장수", 0)) >= 8)
    today_row = next((r for r in rows if r["날짜"] == today), None)

    sentences_done = int(today_row["완료문장수"]) if today_row else 0
    correct_count = int(today_row["정답수"]) if today_row else 0

    # current day: completed days + 1, but if today already completed, stay at completed count
    if today_row and int(today_row["완료문장수"]) >= 8:
        current_day = min(completed_days, 30)
    else:
        current_day = min(completed_days + 1, 30)

    from ai_handler import TRAINING_CURRICULUM
    day_info = TRAINING_CURRICULUM[current_day - 1] if current_day <= 30 else TRAINING_CURRICULUM[29]

    return {
        "current_day": current_day,
        "sentences_done": sentences_done,
        "correct_count": correct_count,
        "day_info": day_info,
        "today": today,
        "today_complete": sentences_done >= 8,
        "program_complete": completed_days >= 30,
    }


def update_training_progress(date_str: str, day_number: int, sentences_done: int, correct_count: int):
    db = get_db()
    if db:
        db.update_training_progress(date_str, day_number, sentences_done, correct_count)
        return
    rows = []
    found = False
    try:
        with open(TRAINING_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["날짜"] == date_str:
                    row["완료문장수"] = sentences_done
                    row["정답수"] = correct_count
                    found = True
                rows.append(dict(row))
    except FileNotFoundError:
        pass
    if not found:
        rows.append({"날짜": date_str, "요일번호": day_number, "완료문장수": sentences_done, "정답수": correct_count})
    with open(TRAINING_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=TRAINING_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def load_correct_sentences(day_number: int) -> list:
    db = get_db()
    if db:
        return db.load_correct_sentences(day_number)
    sentences = []
    try:
        with open(TRAINING_SENTENCES_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row.get("요일번호", 0)) == day_number and row.get("평가결과") == "correct":
                    sentences.append(row["한국어문장"])
    except FileNotFoundError:
        pass
    return sentences


def save_sentence_result(day_number: int, korean_sentence: str, evaluation: str):
    db = get_db()
    if db:
        db.save_sentence_result(day_number, korean_sentence, evaluation)
        return
    if not korean_sentence:
        return
    existing = load_correct_sentences(day_number)
    if korean_sentence in existing:
        return
    with open(TRAINING_SENTENCES_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=TRAINING_SENTENCES_HEADERS)
        writer.writerow({"요일번호": day_number, "한국어문장": korean_sentence, "평가결과": evaluation})


def load_completed_phrases() -> list:
    db = get_db()
    if db:
        return db.load_completed_phrases()
    completed = []
    try:
        with open(PHRASE_PROGRESS_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.append(row["표현"])
    except FileNotFoundError:
        pass
    return completed


def save_phrase_complete(phrase: str, category: str):
    db = get_db()
    if db:
        db.save_phrase_complete(phrase, category)
        return
    completed = load_completed_phrases()
    if phrase in completed:
        return
    with open(PHRASE_PROGRESS_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PHRASE_PROGRESS_HEADERS)
        writer.writerow({
            "표현": phrase,
            "카테고리": category,
            "완료날짜": datetime.date.today().isoformat(),
        })


def get_next_phrase_data():
    """다음에 배울 표현 반환 (완료되지 않은 것 중 첫 번째)"""
    from ai_handler import BUSINESS_PHRASES
    completed = load_completed_phrases()
    for cat in BUSINESS_PHRASES:
        for phrase in cat["phrases"]:
            if phrase["phrase"] not in completed:
                return {
                    "phrase": phrase["phrase"],
                    "meaning": phrase["meaning"],
                    "usage": phrase["usage"],
                    "examples": phrase["examples"],
                    "category": cat["category"],
                    "category_title": cat["title"],
                }
    return None  # 모든 표현 완료


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/debug")
def api_debug():
    db_status = "not configured"
    db_error = None
    if os.environ.get("SPREADSHEET_ID"):
        try:
            db = get_db()
            db_status = "connected" if db else "init failed (check logs)"
        except Exception as e:
            db_error = str(e)
            db_status = "error"
    return jsonify({
        "spreadsheet_id_set": bool(os.environ.get("SPREADSHEET_ID")),
        "credentials_set": bool(os.environ.get("GOOGLE_CREDENTIALS_JSON")),
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "db_status": db_status,
        "db_error": db_error,
    })


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat")
def chat():
    session["session_id"] = str(uuid.uuid4())[:8].upper()
    # 과거 대화를 초기 컨텍스트로 로드
    session["conversation_history"] = load_recent_context(n=30)
    return render_template("chat.html")


@app.route("/api/daily-words")
def api_daily_words():
    try:
        return jsonify(get_daily_words())
    except Exception as e:
        import traceback
        app.logger.error(f"daily-words error: {traceback.format_exc()}")
        return jsonify({"error": str(e), "detail": traceback.format_exc()}), 500


@app.route("/api/calendar")
def api_calendar():
    dates = get_dates_with_conversations()
    return jsonify({"dates": dates})


@app.route("/api/history/<date_str>")
def api_history(date_str):
    rows = get_conversations_for_date(date_str)
    memo = get_memo(date_str)
    return jsonify({"conversations": rows, "memo": memo})


@app.route("/api/memo/<date_str>", methods=["POST"])
def api_save_memo(date_str):
    data = request.json
    save_memo(date_str, data.get("memo", ""))
    return jsonify({"ok": True})


@app.route("/api/today")
def api_today():
    today = datetime.date.today().isoformat()
    rows = get_conversations_for_date(today)
    return jsonify({"conversations": rows, "date": today})


@app.route("/api/business-scenarios")
def api_business_scenarios():
    from ai_handler import AIHandler
    ai = AIHandler(client)
    return jsonify(ai.get_business_scenarios())


@app.route("/api/real-english-situations")
def api_real_english_situations():
    from ai_handler import AIHandler
    ai = AIHandler(client)
    return jsonify(ai.get_real_english_situations())


@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.json
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Translate the following English text to natural Korean:\n\n{text}"}],
            temperature=0.3,
        )
        return jsonify({"translation": response.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/business-phrases")
def api_business_phrases():
    from ai_handler import AIHandler
    ai = AIHandler(client)
    return jsonify(ai.get_business_phrases())


@app.route("/api/training-status")
def api_training_status():
    status = get_training_status()
    return jsonify(status)


@app.route("/api/training-progress", methods=["POST"])
def api_training_progress():
    data = request.json
    update_training_progress(
        data["date"], data["day_number"], data["sentences_done"], data["correct_count"]
    )
    return jsonify({"ok": True})


@app.route("/api/greeting")
def api_greeting():
    from ai_handler import AIHandler
    ai = AIHandler(client)
    past_history = session.get("conversation_history", [])
    mode = request.args.get("mode", "free")
    scenario_id = request.args.get("scenario", "")

    scenario = None
    if mode == "phrase":
        next_phrase = get_next_phrase_data()
        if not next_phrase:
            return jsonify({
                "response": "Congratulations! You've learned all 25 business expressions! 🎉",
                "hint": "", "current_phrase": "", "phrase_meaning": "",
                "all_complete": True, "next_phrase": None
            })
        greeting = ai.get_initial_greeting(
            mode="phrase",
            phrase_data=next_phrase,
        )
        return jsonify({
            "response": greeting["response"],
            "hint": greeting.get("hint", ""),
            "current_phrase": next_phrase["phrase"],
            "phrase_meaning": next_phrase["meaning"],
            "phrase_data": next_phrase,
            "all_complete": False,
        })

    phrase_category = request.args.get("phrase_category", "")
    if mode == "training":
        status = get_training_status()
        day_info = status["day_info"]
        sentences_done = status["sentences_done"]
        used_sentences = load_correct_sentences(day_info["day"])
        greeting = ai.get_initial_greeting(
            mode="training",
            day_info=day_info,
            sentences_done=sentences_done,
            used_sentences=used_sentences,
        )
        return jsonify({
            "response": greeting["response"],
            "hint": greeting.get("hint", ""),
            "evaluation": greeting.get("evaluation", "start"),
            "feedback_kr": greeting.get("feedback_kr", ""),
            "correct_answer": greeting.get("correct_answer", ""),
            "next_prompt": greeting.get("next_prompt", {"korean": "", "hint": ""}),
            "progress": greeting.get("progress", {"current": 0, "total": 8}),
            "session_complete": greeting.get("session_complete", False),
            "day_info": day_info,
            "sentences_done": sentences_done,
        })

    if mode == "business" and scenario_id:
        scenarios = ai.get_business_scenarios()
        scenario = next((s for s in scenarios if s["id"] == scenario_id), None)
    elif mode == "real_english" and scenario_id:
        situations = ai.get_real_english_situations()
        scenario = next((s for s in situations if s["id"] == scenario_id), None)

    greeting = ai.get_initial_greeting(
        past_history if past_history else None,
        mode=mode,
        scenario=scenario,
    )
    return jsonify({
        "response": greeting["response"],
        "hint": greeting.get("hint", ""),
        "current_phrase": greeting.get("current_phrase", ""),
        "phrase_meaning": greeting.get("phrase_meaning", ""),
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    session_id = session.get("session_id", str(uuid.uuid4())[:8].upper())
    conversation_history = session.get("conversation_history", [])

    mode = data.get("mode", "free")
    scenario_data = data.get("scenario", None)
    phrase_category = data.get("phrase_category", None)
    phrase_data = data.get("phrase_data", None)
    training_day_info = data.get("day_info", None)
    training_sentences_done = data.get("sentences_done", 0)
    training_correct_count = data.get("correct_count", 0)
    current_korean_prompt = data.get("current_korean_prompt", "")

    from ai_handler import AIHandler

    ai = AIHandler(client)
    try:
        used_sentences = load_correct_sentences(training_day_info["day"]) if mode == "training" and training_day_info else []
        result = ai.chat(user_message, conversation_history, mode=mode, scenario=scenario_data, phrase_category=phrase_category, phrase_data=phrase_data, day_info=training_day_info, sentences_done=training_sentences_done, used_sentences=used_sentences)
        # Server-side phrase confirmation: if user message contains the phrase, always mark confirmed
        if mode == "phrase" and phrase_data:
            phrase_text = phrase_data.get("phrase", "").lower()
            if phrase_text and phrase_text in user_message.lower():
                result["phrase_confirmed"] = True
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": result["response"]})
    session["conversation_history"] = conversation_history
    session.modified = True

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db = get_db()
    if db:
        db.append_conversation(
            session_id=session_id,
            timestamp=timestamp,
            user_sentence=user_message,
            corrected_sentence=result.get("corrected", ""),
            correction_explanation=result.get("explanation", ""),
            ai_response=result["response"],
            pronunciation_tip=result.get("pronunciation_tip", ""),
            has_correction=result["has_correction"],
        )
    else:
        from local_saver import LocalSaver
        saver = LocalSaver(directory=BASE_DIR)
        saver.append_row(
            session_id=session_id,
            timestamp=timestamp,
            user_sentence=user_message,
            corrected_sentence=result.get("corrected", ""),
            correction_explanation=result.get("explanation", ""),
            ai_response=result["response"],
            pronunciation_tip=result.get("pronunciation_tip", ""),
            has_correction=result["has_correction"],
        )

    if mode == "training" and training_day_info:
        new_sentences_done = training_sentences_done + 1
        evaluation = result.get("evaluation", "")
        is_correct = evaluation in ("correct", "partial")
        new_correct_count = training_correct_count + (1 if is_correct else 0)
        today = datetime.date.today().isoformat()
        update_training_progress(today, training_day_info["day"], new_sentences_done, new_correct_count)
        # 정답인 경우 해당 한국어 문장을 저장 (재출제 방지)
        if evaluation == "correct" and current_korean_prompt:
            save_sentence_result(training_day_info["day"], current_korean_prompt, "correct")

    return jsonify({
        "response": result["response"],
        "has_correction": result["has_correction"],
        "original": result.get("original", ""),
        "corrected": result.get("corrected", ""),
        "explanation": result.get("explanation", ""),
        "pronunciation_tip": result.get("pronunciation_tip", ""),
        "hint": result.get("hint", ""),
        "current_phrase": result.get("current_phrase", ""),
        "phrase_meaning": result.get("phrase_meaning", ""),
        "phrase_confirmed": result.get("phrase_confirmed", False),
        "phrase_data": phrase_data,
        "timestamp": timestamp,
        "evaluation": result.get("evaluation", ""),
        "feedback_kr": result.get("feedback_kr", ""),
        "correct_answer": result.get("correct_answer", ""),
        "next_prompt": result.get("next_prompt", {"korean": "", "hint": ""}),
        "progress": result.get("progress", {"current": 0, "total": 8}),
        "session_complete": result.get("session_complete", False),
    })


@app.route("/api/phrase-list")
def api_phrase_list():
    """모든 표현 목록 + 완료 여부"""
    from ai_handler import BUSINESS_PHRASES
    completed = load_completed_phrases()
    result = []
    for cat in BUSINESS_PHRASES:
        phrases = []
        for p in cat["phrases"]:
            phrases.append({
                "phrase": p["phrase"],
                "meaning": p["meaning"],
                "usage": p["usage"],
                "examples": p["examples"],
                "completed": p["phrase"] in completed,
            })
        result.append({
            "category": cat["category"],
            "title": cat["title"],
            "emoji": cat["emoji"],
            "phrases": phrases,
        })
    total = sum(len(cat["phrases"]) for cat in BUSINESS_PHRASES)
    return jsonify({"categories": result, "completed_count": len(completed), "total": total})


@app.route("/api/phrase-complete", methods=["POST"])
def api_phrase_complete():
    data = request.json
    save_phrase_complete(data["phrase"], data.get("category", ""))
    next_phrase = get_next_phrase_data()
    return jsonify({"ok": True, "next_phrase": next_phrase})


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio"}), 400

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_file.save(f.name)
        temp_path = f.name

    try:
        with open(temp_path, "rb") as af:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=("recording.wav", af, "audio/wav"),
                language="en",
            )
        return jsonify({"text": transcript.text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass


@app.route("/api/tts", methods=["POST"])
def api_tts():
    data = request.json
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_path = f.name

    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
        )
        with open(temp_path, "wb") as f:
            f.write(response.content)
        return send_file(temp_path, mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🌐 http://localhost:5000 에서 실행 중...")
    app.run(debug=True, port=5000)
