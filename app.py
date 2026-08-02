"""영어 회화 웹 앱 - Flask 백엔드"""

import os
import io
import csv
import json
import hmac
import uuid
import datetime
import tempfile

from flask import (
    Flask, render_template, request, jsonify, session, send_file,
    redirect, render_template_string,
)
from dotenv import load_dotenv

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

# OpenAI 클라이언트는 지연 생성한다(아래 _LazyClient). openai 패키지 임포트는
# 콜드스타트 비용이 커서, AI를 쓰지 않는 엔드포인트는 아예 로드하지 않게 한다.

# 하루 API 비용 상한 (USD). 환경변수 DAILY_BUDGET_USD 로 조정 가능.
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "1.0"))


def init_csvs():
    if not os.path.exists(CONVERSATION_CSV):
        with open(CONVERSATION_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(CONV_HEADERS)
    if not os.path.exists(MEMO_CSV):
        with open(MEMO_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(MEMO_HEADERS)


try:
    init_csvs()
except OSError:
    pass  # read-only filesystem (Vercel) — Sheets DB used instead


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


def get_recent_conversations(days: int) -> list:
    """최근 days 일치 대화를 한 번의 읽기로 가져온다(최신순)."""
    since = (datetime.date.today() - datetime.timedelta(days=days - 1)).isoformat()
    db = get_db()
    if db:
        return db.get_conversations_since(since)
    rows = []
    try:
        with open(CONVERSATION_CSV, "r", encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f) if r["날짜/시간"][:10] >= since]
    except FileNotFoundError:
        pass
    return sorted(rows, key=lambda r: r.get("날짜/시간", ""), reverse=True)


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
    combined = list(dict.fromkeys(existing + [w.lower() for w in new_words]))
    # 프롬프트 토큰 비용이 무한히 커지지 않도록 최근 300개만 유지
    combined = combined[-300:]
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


# ── 접근 코드 게이트 (선택) ────────────────────────────────────────────────────
# ACCESS_CODE 환경변수가 설정된 경우에만 활성화됩니다. 미설정 시 기존과 동일하게
# 누구나 접근 가능(무해). 공개 배포 시 OpenAI 크레딧 남용을 막기 위한 최소 장치.
ACCESS_CODE = os.environ.get("ACCESS_CODE")

_UNLOCK_PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>잠금</title>
<style>
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       display:flex;min-height:100vh;align-items:center;justify-content:center;
       margin:0;background:#0f172a;color:#e2e8f0}
  form{background:#1e293b;padding:32px;border-radius:16px;width:280px;text-align:center}
  h1{font-size:18px;margin:0 0 16px}
  input{width:100%;box-sizing:border-box;padding:12px;border-radius:8px;border:1px solid #334155;
        background:#0f172a;color:#e2e8f0;font-size:16px;margin-bottom:12px}
  button{width:100%;padding:12px;border:0;border-radius:8px;background:#6366f1;color:#fff;
         font-size:16px;cursor:pointer}
  .err{color:#f87171;font-size:13px;margin-bottom:8px;min-height:16px}
</style></head>
<body><form method="post">
  <h1>🔒 접근 코드를 입력하세요</h1>
  <div class="err">{{ error }}</div>
  <input type="password" name="code" autofocus autocomplete="current-password">
  <button type="submit">입장</button>
</form></body></html>"""


@app.before_request
def _require_access_code():
    if not ACCESS_CODE:
        return  # 게이트 비활성화
    if session.get("authed"):
        return
    path = request.path
    if path == "/unlock" or path.startswith("/static"):
        return
    if path.startswith("/api/"):
        return jsonify({"error": "unauthorized"}), 401
    return redirect("/unlock")


@app.route("/unlock", methods=["GET", "POST"])
def unlock():
    if not ACCESS_CODE:
        return redirect("/")
    error = ""
    if request.method == "POST":
        code = (request.form.get("code") or "").strip()
        if hmac.compare_digest(code, ACCESS_CODE):
            session["authed"] = True
            session.permanent = True
            return redirect("/")
        error = "코드가 올바르지 않습니다."
    return render_template_string(_UNLOCK_PAGE, error=error)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/debug")
def api_debug():
    # 비밀값은 노출하지 않고, Sheets 초기화 실패의 '원인'만 안전하게 반환하는 진단용.
    info = {
        "spreadsheet_id_set": bool(os.environ.get("SPREADSHEET_ID")),
        "credentials_set": bool(os.environ.get("GOOGLE_CREDENTIALS_JSON")),
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "access_gate_on": bool(ACCESS_CODE),
    }

    # 1) 자격증명 JSON 파싱 점검
    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    client_email = None
    if creds_json:
        try:
            parsed = json.loads(creds_json)
            info["creds_json_parseable"] = True
            client_email = parsed.get("client_email")
            info["client_email"] = client_email  # 시트 공유 대상 확인용(비밀 아님)
            pk = parsed.get("private_key", "")
            info["private_key_has_real_newlines"] = "\n" in pk
            info["private_key_has_markers"] = pk.startswith("-----BEGIN") and pk.strip().endswith("PRIVATE KEY-----")
        except Exception as e:
            info["creds_json_parseable"] = False
            info["creds_parse_error"] = f"{type(e).__name__}: {str(e)[:200]}"

    # 2) 실제 SheetsDB 초기화 시도 (메모이즈 무시하고 신선하게)
    if os.environ.get("SPREADSHEET_ID"):
        try:
            from sheets_db import SheetsDB
            SheetsDB(os.environ["SPREADSHEET_ID"])
            info["sheets_init"] = "OK"
        except Exception as e:
            import traceback
            info["sheets_init"] = "FAILED"
            info["sheets_error_type"] = type(e).__name__
            info["sheets_error"] = str(e)[:500]
            info["sheets_trace_tail"] = traceback.format_exc()[-600:]
    return jsonify(info)


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
    data = request.get_json(silent=True) or {}
    save_memo(date_str, data.get("memo", ""))
    return jsonify({"ok": True})


@app.route("/api/today")
def api_today():
    today = datetime.date.today().isoformat()
    rows = get_conversations_for_date(today)
    return jsonify({"conversations": rows, "date": today})


def _static_json(payload, hours: int = 6):
    """내용이 바뀌지 않는 목록 응답. 브라우저/CDN이 캐시하게 해 함수 호출 자체를 줄인다."""
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = f"public, max-age={hours*3600}, s-maxage=86400"
    return resp


@app.route("/api/business-scenarios")
def api_business_scenarios():
    from ai_handler import AIHandler
    ai = AIHandler(client)
    return _static_json(ai.get_business_scenarios())


@app.route("/api/real-english-situations")
def api_real_english_situations():
    from ai_handler import AIHandler
    ai = AIHandler(client)
    return _static_json(ai.get_real_english_situations())


@app.route("/api/translate", methods=["POST"])
def api_translate():
    data = request.get_json(silent=True) or {}
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
    except BudgetExceededError:
        raise
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/greeting")
def api_greeting():
    from ai_handler import AIHandler
    ai = AIHandler(client)
    past_history = session.get("conversation_history", [])
    mode = request.args.get("mode", "free")
    scenario_id = request.args.get("scenario", "")

    scenario = None
    if mode == "opic":
        from opic_curriculum import OPIC_CURRICULUM, TOTAL_DAYS
        progress = get_opic_progress()
        day = min(progress["day"], TOTAL_DAYS)
        content = OPIC_CURRICULUM[day - 1]
        greeting_text = (
            f"Hi! I'm Ava, your OPIc coach. Welcome to Day {day}! 🎯 "
            f"Today's topic is \"{content['topic']}\". "
            f"Check today's vocabulary and expressions above, then answer this question:\n\n"
            f"Q. {content['question']}"
        )
        return jsonify({
            "response": greeting_text,
            "has_correction": False,
            "hint": content["mission"],
            "opic_day": content,
            "day": day,
            "total": TOTAL_DAYS,
        })

    if mode == "speaking":
        return jsonify({
            "response": "Let's get started! 🎯 I'll show you a Korean sentence — type it in English. Don't worry about being perfect; natural variations are accepted!",
            "has_correction": False,
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
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    session_id = session.get("session_id", str(uuid.uuid4())[:8].upper())
    conversation_history = session.get("conversation_history", [])

    mode = data.get("mode", "free")
    scenario_data = data.get("scenario", None)
    opic_day = data.get("opic_day", None)

    from ai_handler import AIHandler

    ai = AIHandler(client)
    try:
        result = ai.chat(user_message, conversation_history, mode=mode, scenario=scenario_data, opic_day=opic_day)
    except BudgetExceededError:
        raise
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

    return jsonify({
        "response": result["response"],
        "has_correction": result["has_correction"],
        "original": result.get("original", ""),
        "corrected": result.get("corrected", ""),
        "explanation": result.get("explanation", ""),
        "pronunciation_tip": result.get("pronunciation_tip", ""),
        "hint": result.get("hint", ""),
        "opic_feedback": result.get("opic_feedback", ""),
        "better_expression": result.get("better_expression", ""),
        "better_explanation": result.get("better_explanation", ""),
        "timestamp": timestamp,
    })


@app.route("/api/chunk-text", methods=["POST"])
def api_chunk_text():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    prompt = f"""You are an English pronunciation coach for Korean speakers.
Analyze the following English text and mark where to pause for natural reading/speaking.

Use these markers:
- | for a short pause (between thought groups, at commas)
- || for a longer pause (at sentence boundaries, major clause breaks)

Then list each chunk with a brief Korean explanation of WHY to pause there (grammar reason, breath point, meaning group).

Text: "{text}"

Respond in this EXACT JSON format:
{{
    "chunked": "the text with | and || markers inserted naturally",
    "chunks": [
        {{"text": "chunk text (no markers)", "reason": "왜 여기서 끊는지 한국어로 짧게"}}
    ],
    "tip": "이 문장을 읽을 때의 전체적인 팁 (한국어, 1-2문장)"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return jsonify(result)
    except BudgetExceededError:
        raise
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 오픽 90일 커리큘럼 ────────────────────────────────────────

OPIC_PROGRESS_JSON = os.path.join(BASE_DIR, "opic_progress.json")

def get_opic_progress(raw=None):
    """{'day': 현재 학습일, 'completed_days': [...], 'last_completed': 'YYYY-MM-DD'}

    raw 를 주면 캐시 시트를 다시 읽지 않는다(호출부에서 이미 읽어온 경우).
    """
    default = {"day": 1, "completed_days": [], "last_completed": ""}
    db = get_db()
    if db:
        try:
            if raw is None:
                raw = db.get_cache("opic_progress")
            if raw:
                return {**default, **json.loads(raw)}
        except Exception as e:
            app.logger.error(f"opic progress read failed: {e}")
        return default
    if os.path.exists(OPIC_PROGRESS_JSON):
        try:
            with open(OPIC_PROGRESS_JSON, encoding="utf-8") as f:
                return {**default, **json.load(f)}
        except Exception:
            pass
    return default


def save_opic_progress(progress):
    db = get_db()
    if db:
        try:
            db.set_cache("opic_progress", json.dumps(progress, ensure_ascii=False))
            return
        except Exception as e:
            app.logger.error(f"opic progress save failed: {e}")
    try:
        with open(OPIC_PROGRESS_JSON, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False)
    except OSError:
        pass


@app.route("/api/home")
def api_home():
    """홈 화면이 필요한 데이터를 한 번에 반환한다.

    예전에는 홈에서 5개 엔드포인트를 각각 호출해 서버리스 콜드스타트가 동시에
    여러 번 일어났다. 느리고 실패 위험이 큰 오늘의 단어(첫 호출 시 GPT 생성)만
    분리해 두고, 나머지는 이 요청 하나로 처리한다.
    """
    from opic_curriculum import OPIC_CURRICULUM, TOTAL_DAYS, PHASE_INFO
    from level_sentences import LEVELS, LEVEL_SENTENCES

    out = {}

    # 캐시 시트 1회 읽기로 오픽 진행률 + 드릴 오답을 함께 확보
    db = get_db()
    cached = {}
    if db:
        try:
            cached = db.get_cache_many(["opic_progress", "drill_mistakes"])
        except Exception as e:
            app.logger.error(f"home cache read failed: {e}")

    # 오픽 오늘의 커리큘럼
    progress = get_opic_progress(raw=cached.get("opic_progress"))
    day = min(progress["day"], TOTAL_DAYS)
    content = OPIC_CURRICULUM[day - 1]
    out["opic"] = {
        "day": day,
        "total": TOTAL_DAYS,
        "completed_count": len(progress["completed_days"]),
        "completed_today": progress["last_completed"] == datetime.date.today().isoformat(),
        "all_complete": len(progress["completed_days"]) >= TOTAL_DAYS,
        "phase_info": PHASE_INFO[content["phase"]],
        "content": content,
    }

    # 복습 대기 건수 (홈은 개수만 필요하므로 목록은 만들지 않는다)
    vocab_n = sum(
        len(OPIC_CURRICULUM[d - 1]["vocab"])
        for d in progress["completed_days"]
        if (progress["day"] - d) in REVIEW_INTERVALS and 1 <= d <= len(OPIC_CURRICULUM)
    )
    try:
        recent = get_recent_conversations(7)
    except Exception:
        recent = []
    seen = set()
    for row in recent:
        if len(seen) >= 5:
            break
        orig = row.get("내 영어 문장", "")
        if row.get("수정 필요") == "O" and row.get("수정된 문장") and orig:
            seen.add(orig)
    drill_n = len(_load_json_store("drill_mistakes", [], raw=cached.get("drill_mistakes"))[:5])
    out["review"] = {
        "counts": {"vocab": vocab_n, "correction": len(seen), "drill": drill_n},
        "total": vocab_n + len(seen) + drill_n,
    }

    # 단계 목록 (정적)
    out["levels"] = [
        {**lv, "count": sum(1 for x in LEVEL_SENTENCES if x["level"] == lv["id"])}
        for lv in LEVELS
    ]

    # 달력
    try:
        out["dates"] = get_dates_with_conversations()
    except Exception:
        out["dates"] = []

    return jsonify(out)


@app.route("/api/opic-today")
def api_opic_today():
    """오늘의 커리큘럼 (현재 Day 기준)"""
    from opic_curriculum import OPIC_CURRICULUM, TOTAL_DAYS, PHASE_INFO
    progress = get_opic_progress()
    day = min(progress["day"], TOTAL_DAYS)
    content = OPIC_CURRICULUM[day - 1]
    today = datetime.date.today().isoformat()
    return jsonify({
        "day": day,
        "total": TOTAL_DAYS,
        "completed_count": len(progress["completed_days"]),
        "completed_today": progress["last_completed"] == today,
        "all_complete": len(progress["completed_days"]) >= TOTAL_DAYS,
        "phase_info": PHASE_INFO[content["phase"]],
        "content": content,
    })


@app.route("/api/opic-complete", methods=["POST"])
def api_opic_complete():
    """오늘의 미션 완료 → 다음 Day로 진행 (하루 1 Day 제한)"""
    from opic_curriculum import TOTAL_DAYS
    progress = get_opic_progress()
    today = datetime.date.today().isoformat()
    day = progress["day"]

    # 하루에 한 Day만 완료 가능 (간격 반복 설계 유지)
    if progress["last_completed"] == today:
        return jsonify({
            "ok": False,
            "already_completed_today": True,
            "next_day": progress["day"],
            "all_complete": len(progress["completed_days"]) >= TOTAL_DAYS,
        })

    if day not in progress["completed_days"]:
        progress["completed_days"].append(day)
    progress["last_completed"] = today
    if day < TOTAL_DAYS:
        progress["day"] = day + 1
    save_opic_progress(progress)

    return jsonify({
        "ok": True,
        "completed_day": day,
        "next_day": progress["day"],
        "all_complete": len(progress["completed_days"]) >= TOTAL_DAYS,
    })


# ── 복습 시스템 (간격 반복) ───────────────────────────────────

REVIEW_INTERVALS = (1, 3, 7, 21)  # 에빙하우스 간격: 완료 후 1, 3, 7, 21일(커리큘럼 Day 기준)


def _load_json_store(key, default, raw=None):
    """Sheets 캐시 우선, 로컬 JSON 파일 폴백. raw 를 주면 재조회하지 않는다."""
    db = get_db()
    if db:
        try:
            if raw is None:
                raw = db.get_cache(key)
            if raw:
                return json.loads(raw)
        except Exception as e:
            app.logger.error(f"{key} read failed: {e}")
        return default
    path = os.path.join(BASE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _save_json_store(key, data):
    db = get_db()
    if db:
        try:
            db.set_cache(key, json.dumps(data, ensure_ascii=False))
            return
        except Exception as e:
            app.logger.error(f"{key} save failed: {e}")
    try:
        with open(os.path.join(BASE_DIR, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except OSError:
        pass


# ── 일일 API 비용 가드 ────────────────────────────────────────
# 오늘 누적 비용을 저장소(Sheets 캐시 우선, 로컬 JSON 폴백)에 날짜별로 기록하고,
# DAILY_BUDGET_USD 를 넘으면 이후 OpenAI 호출을 차단한다.

def _spend_key():
    return "api_spend_" + datetime.date.today().isoformat()


def _spend_tmp_path():
    # Vercel 등 앱 디렉토리가 읽기 전용인 환경 대비: 쓰기 가능한 임시 디렉토리 사용.
    return os.path.join(tempfile.gettempdir(), _spend_key() + ".json")


def get_today_spend():
    db = get_db()
    if db:
        try:
            raw = db.get_cache(_spend_key())
            return float(json.loads(raw)["usd"]) if raw else 0.0
        except Exception:
            return 0.0
    # 로컬/서버리스: 쓰기 가능한 임시 파일에 기록
    try:
        with open(_spend_tmp_path(), encoding="utf-8") as f:
            return float(json.load(f).get("usd", 0.0))
    except (FileNotFoundError, ValueError, TypeError, KeyError):
        return 0.0


def add_today_spend(amount):
    current = get_today_spend()
    new_total = current + float(amount)
    db = get_db()
    if db:
        try:
            db.set_cache(_spend_key(), json.dumps({"usd": new_total}))
            return
        except Exception as e:
            app.logger.error(f"spend save failed: {e}")
    try:
        with open(_spend_tmp_path(), "w", encoding="utf-8") as f:
            json.dump({"usd": new_total}, f)
    except OSError:
        pass


# 원본 client 를 비용 가드로 감싼다(app 전역 client 재할당).
from cost_guard import GuardedOpenAI, BudgetExceededError


class _LazyClient:
    """첫 사용 시점에 openai 를 임포트하고 비용 가드로 감싼 클라이언트를 만든다."""

    _real = None

    def __getattr__(self, name):
        if _LazyClient._real is None:
            from openai import OpenAI
            _LazyClient._real = GuardedOpenAI(
                OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
                get_today_spend, add_today_spend, DAILY_BUDGET_USD,
            )
        return getattr(_LazyClient._real, name)


client = _LazyClient()


@app.route("/api/budget")
def api_budget():
    """오늘 사용액/한도 조회 (모니터링용)"""
    spent = get_today_spend()
    return jsonify({
        "date": datetime.date.today().isoformat(),
        "spent_usd": round(spent, 4),
        "budget_usd": DAILY_BUDGET_USD,
        "remaining_usd": round(max(0.0, DAILY_BUDGET_USD - spent), 4),
        "exceeded": spent >= DAILY_BUDGET_USD,
    })


@app.errorhandler(BudgetExceededError)
def _handle_budget_exceeded(e):
    return jsonify({"error": str(e), "budget_exceeded": True}), 429


@app.route("/api/review-today")
def api_review_today():
    """오늘의 복습 큐: 오픽 단어(간격 반복) + 최근 교정 문장 + 드릴 오답

    Sheets 읽기는 캐시 1회 + 대화기록 1회로 끝낸다(예전엔 최대 9회 전체 읽기).
    """
    from opic_curriculum import OPIC_CURRICULUM
    items = []

    # 캐시 시트를 한 번만 읽어 진행률과 드릴 오답을 함께 가져온다
    db = get_db()
    cached = {}
    if db:
        try:
            cached = db.get_cache_many(["opic_progress", "drill_mistakes"])
        except Exception as e:
            app.logger.error(f"review cache read failed: {e}")

    # 1) 오픽 단어 — 완료한 Day 중 1/3/7/21일 전 것 재출제
    progress = get_opic_progress(raw=cached.get("opic_progress"))
    current_day = progress["day"]
    for d in progress["completed_days"]:
        if (current_day - d) in REVIEW_INTERVALS and 1 <= d <= len(OPIC_CURRICULUM):
            for v in OPIC_CURRICULUM[d - 1]["vocab"]:
                items.append({
                    "type": "vocab", "day": d,
                    "word": v["word"], "meaning": v["meaning"], "example": v["example"],
                })

    # 2) 최근 7일 회화 교정 문장 (대화기록 1회 읽기, 최신순, 중복 제거, 최대 5개)
    corrections, seen = [], set()
    try:
        recent = get_recent_conversations(7)
    except Exception as e:
        app.logger.error(f"review conversations read failed: {e}")
        recent = []
    for row in recent:
        if len(corrections) >= 5:
            break
        orig = row.get("내 영어 문장", "")
        if row.get("수정 필요") == "O" and row.get("수정된 문장") and orig and orig not in seen:
            seen.add(orig)
            corrections.append({
                "type": "correction",
                "original": orig,
                "corrected": row["수정된 문장"],
                "explanation": row.get("수정 설명", ""),
            })
    items.extend(corrections)

    # 3) 스피킹 드릴 오답 (최대 5개)
    mistakes = _load_json_store("drill_mistakes", [], raw=cached.get("drill_mistakes"))
    for m in mistakes[:5]:
        items.append({"type": "drill", "id": m["id"], "korean": m["korean"], "english": m["english"]})

    counts = {
        "vocab": sum(1 for i in items if i["type"] == "vocab"),
        "correction": sum(1 for i in items if i["type"] == "correction"),
        "drill": sum(1 for i in items if i["type"] == "drill"),
    }
    return jsonify({"items": items, "counts": counts, "total": len(items)})


@app.route("/api/drill-mistake", methods=["POST"])
def api_drill_mistake():
    """스피킹 드릴 오답 저장 (복습용)"""
    data = request.get_json(silent=True) or {}
    s = data.get("sentence")
    if not s or "id" not in s:
        return jsonify({"error": "No sentence"}), 400
    mistakes = _load_json_store("drill_mistakes", [])
    for m in mistakes:
        if m["id"] == s["id"]:
            m["count"] = m.get("count", 1) + 1
            break
    else:
        mistakes.append({"id": s["id"], "korean": s["korean"], "english": s["english"], "count": 1})
    _save_json_store("drill_mistakes", mistakes[:100])
    return jsonify({"ok": True})


@app.route("/api/review-result", methods=["POST"])
def api_review_result():
    """복습 결과 반영 — 드릴 오답을 맞히면 오답 목록에서 제거"""
    data = request.get_json(silent=True) or {}
    if data.get("type") == "drill" and data.get("correct") and data.get("id") is not None:
        mistakes = _load_json_store("drill_mistakes", [])
        mistakes = [m for m in mistakes if m["id"] != data["id"]]
        _save_json_store("drill_mistakes", mistakes)
    return jsonify({"ok": True})


# ── 단계별 말하기 (한→영, 힌트 제공) ──────────────────────────

@app.route("/api/level-list")
def api_level_list():
    """단계 목록 + 문장 수"""
    from level_sentences import LEVELS, LEVEL_SENTENCES
    result = []
    for lv in LEVELS:
        count = sum(1 for s in LEVEL_SENTENCES if s["level"] == lv["id"])
        result.append({**lv, "count": count})
    return _static_json(result)


@app.route("/api/level-next", methods=["POST"])
def api_level_next():
    """선택한 단계에서 아직 안 본 문장 하나 반환"""
    import random
    from level_sentences import LEVEL_SENTENCES
    data = request.json
    level = data.get("level", "")
    exclude_ids = set(data.get("exclude_ids", []))

    pool = [s for s in LEVEL_SENTENCES if s["level"] == level and s["id"] not in exclude_ids]
    if not pool:
        return jsonify({"done": True, "remaining": 0})

    # 순서대로 진행 (쉬운 문장부터) — 단계 안에서는 난이도 순 배열
    sentence = pool[0] if data.get("in_order") else random.choice(pool)
    return jsonify({"sentence": sentence, "remaining": len(pool)})


# ── 스피킹 드릴 ──────────────────────────────────────────────

@app.route("/api/speaking-categories")
def api_speaking_categories():
    """카테고리 목록 반환"""
    from speaking_sentences import SPEAKING_CATEGORIES, SPEAKING_SENTENCES
    cats = []
    for cat in SPEAKING_CATEGORIES:
        count = len(SPEAKING_SENTENCES) if cat["id"] == "전체" else sum(1 for s in SPEAKING_SENTENCES if s["category"] == cat["id"])
        cats.append({**cat, "count": count})
    return _static_json(cats)


@app.route("/api/speaking-next", methods=["POST"])
def api_speaking_next():
    """다음 문장 반환 (이미 본 문장 제외, 랜덤)"""
    import random
    from speaking_sentences import SPEAKING_SENTENCES
    data = request.get_json(silent=True) or {}
    category = data.get("category", "전체")
    exclude_ids = set(data.get("exclude_ids", []))

    pool = SPEAKING_SENTENCES if category == "전체" else [s for s in SPEAKING_SENTENCES if s["category"] == category]
    pool = [s for s in pool if s["id"] not in exclude_ids]

    if not pool:
        return jsonify({"done": True, "remaining": 0})

    sentence = random.choice(pool)
    return jsonify({"sentence": sentence, "remaining": len(pool)})


@app.route("/api/speaking-check", methods=["POST"])
def api_speaking_check():
    """사용자 답변 평가"""
    data = request.get_json(silent=True) or {}
    user_answer = data.get("user_answer", "").strip()
    correct_answer = data.get("correct_answer", "").strip()
    korean = data.get("korean", "").strip()

    if not user_answer:
        return jsonify({"error": "No answer provided"}), 400

    prompt = f"""You are an English speaking drill evaluator for Korean learners.

Korean sentence: {korean}
Reference answer: {correct_answer}
Student's answer: {user_answer}

Evaluate the student's answer. Be FLEXIBLE — accept natural variations that convey the same meaning.
- "correct": meaning is right, expression is natural (minor grammar slips OK)
- "partial": meaning is mostly right but expression has noticeable issues
- "incorrect": wrong meaning or completely off

Respond ONLY in this JSON format:
{{
    "evaluation": "correct" or "partial" or "incorrect",
    "feedback_kr": "2-3문장 한국어 피드백. 왜 맞는지/틀린지 설명하고, partial/incorrect면 개선 방법 제시.",
    "correct_answer": "{correct_answer}",
    "natural_alternatives": ["자연스러운 대안 표현 1-2개 (correct인 경우도 추가 표현 제시)"]
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return jsonify(result)
    except BudgetExceededError:
        raise
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    except BudgetExceededError:
        raise
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass


@app.route("/api/tts", methods=["POST"])
def api_tts():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400

    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
        )
        return send_file(io.BytesIO(response.content), mimetype="audio/mpeg")
    except BudgetExceededError:
        raise
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🌐 http://localhost:5000 에서 실행 중...")
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
