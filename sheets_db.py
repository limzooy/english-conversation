"""Google Sheets 데이터베이스 - 모든 CSV 데이터를 Sheets로 관리"""

import json
import os
import datetime
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

SHEET_CONV      = "대화기록"
SHEET_MEMO      = "메모"
SHEET_TRAIN     = "훈련진행"
SHEET_SENTENCES = "훈련문장"
SHEET_PHRASES   = "표현완료"
SHEET_CACHE     = "캐시"

ALL_SHEETS = {
    SHEET_CONV:      ["날짜/시간", "세션 ID", "내 영어 문장", "수정된 문장", "수정 필요", "수정 설명", "AI 응답", "발음 팁"],
    SHEET_MEMO:      ["날짜", "메모"],
    SHEET_TRAIN:     ["날짜", "요일번호", "완료문장수", "정답수"],
    SHEET_SENTENCES: ["요일번호", "한국어문장", "평가결과"],
    SHEET_PHRASES:   ["표현", "카테고리", "완료날짜"],
    SHEET_CACHE:     ["키", "값"],
}


def _build_service():
    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if creds_json:
        info = json.loads(creds_json)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    else:
        creds_file = os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")
        creds = Credentials.from_service_account_file(creds_file, scopes=SCOPES)
    return build("sheets", "v4", credentials=creds)


class SheetsDB:
    def __init__(self, spreadsheet_id: str):
        self.spreadsheet_id = spreadsheet_id
        self._svc = _build_service()
        self._sheet = self._svc.spreadsheets()
        # 헤더 보정은 best-effort: 시트가 이미 존재하면 검증이 실패(타임아웃 등)해도
        # 읽기/쓰기는 가능하므로 초기화 전체를 막지 않는다.
        try:
            self._ensure_sheets()
        except Exception:
            pass

    def _ensure_sheets(self):
        meta = self._sheet.get(spreadsheetId=self.spreadsheet_id).execute()
        existing = {s["properties"]["title"] for s in meta["sheets"]}
        requests = [
            {"addSheet": {"properties": {"title": name}}}
            for name in ALL_SHEETS if name not in existing
        ]
        if requests:
            self._sheet.batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body={"requests": requests},
            ).execute()
        # 모든 시트의 헤더 행을 한 번의 batchGet 으로 확인(콜드스타트 왕복 최소화)
        names = list(ALL_SHEETS.keys())
        resp = self._sheet.values().batchGet(
            spreadsheetId=self.spreadsheet_id,
            ranges=[f"'{name}'!A1:Z1" for name in names],
        ).execute()
        value_ranges = resp.get("valueRanges", [])
        for name, vr in zip(names, value_ranges):
            if not vr.get("values"):
                self._append(name, ALL_SHEETS[name])

    def _append(self, sheet: str, row: list):
        self._sheet.values().append(
            spreadsheetId=self.spreadsheet_id,
            range=f"'{sheet}'!A1",
            valueInputOption="USER_ENTERED",
            body={"values": [row]},
        ).execute()

    def _read_all(self, sheet: str) -> list:
        result = self._sheet.values().get(
            spreadsheetId=self.spreadsheet_id,
            range=f"'{sheet}'!A1:Z",
        ).execute()
        rows = result.get("values", [])
        if len(rows) < 2:
            return []
        headers = rows[0]
        return [
            dict(zip(headers, row + [""] * max(0, len(headers) - len(row))))
            for row in rows[1:]
        ]

    def _find_row_index(self, sheet: str, col_index: int, match_value: str) -> int:
        """해당 컬럼에서 값이 일치하는 행의 인덱스 반환 (없으면 -1), 1-based (헤더 제외)"""
        result = self._sheet.values().get(
            spreadsheetId=self.spreadsheet_id,
            range=f"'{sheet}'!A1:Z",
        ).execute()
        rows = result.get("values", [])
        for i, row in enumerate(rows[1:], start=2):  # 2 = 1-indexed + skip header
            if len(row) > col_index and row[col_index] == match_value:
                return i
        return -1

    def _update_cell_range(self, sheet: str, range_a1: str, values: list):
        self._sheet.values().update(
            spreadsheetId=self.spreadsheet_id,
            range=f"'{sheet}'!{range_a1}",
            valueInputOption="USER_ENTERED",
            body={"values": [values]},
        ).execute()

    # ── Conversations ──────────────────────────────────────────────────

    def append_conversation(self, session_id, timestamp, user_sentence,
                            corrected_sentence, correction_explanation,
                            ai_response, pronunciation_tip, has_correction):
        self._append(SHEET_CONV, [
            timestamp, session_id, user_sentence,
            corrected_sentence if has_correction else "",
            "O" if has_correction else "",
            correction_explanation if has_correction else "",
            ai_response, pronunciation_tip,
        ])

    def get_conversations_for_date(self, date_str: str) -> list:
        rows = self._read_all(SHEET_CONV)
        return [r for r in rows if r.get("날짜/시간", "").startswith(date_str)]

    def get_dates_with_conversations(self) -> list:
        rows = self._read_all(SHEET_CONV)
        dates = {r["날짜/시간"][:10] for r in rows if r.get("날짜/시간")}
        return sorted(dates)

    def load_recent_context(self, n: int = 30) -> list:
        rows = self._read_all(SHEET_CONV)
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

    # ── Memos ──────────────────────────────────────────────────────────

    def get_memo(self, date_str: str) -> str:
        rows = self._read_all(SHEET_MEMO)
        for row in rows:
            if row.get("날짜") == date_str:
                return row.get("메모", "")
        return ""

    def save_memo(self, date_str: str, memo_text: str):
        row_idx = self._find_row_index(SHEET_MEMO, 0, date_str)
        if row_idx >= 0:
            self._update_cell_range(SHEET_MEMO, f"B{row_idx}", [memo_text])
        else:
            self._append(SHEET_MEMO, [date_str, memo_text])

    # ── Training Progress ───────────────────────────────────────────────

    def get_training_rows(self) -> list:
        return self._read_all(SHEET_TRAIN)

    def update_training_progress(self, date_str: str, day_number: int,
                                 sentences_done: int, correct_count: int):
        row_idx = self._find_row_index(SHEET_TRAIN, 0, date_str)
        if row_idx >= 0:
            self._update_cell_range(
                SHEET_TRAIN, f"A{row_idx}:D{row_idx}",
                [date_str, day_number, sentences_done, correct_count],
            )
        else:
            self._append(SHEET_TRAIN, [date_str, day_number, sentences_done, correct_count])

    # ── Training Sentences ──────────────────────────────────────────────

    def load_correct_sentences(self, day_number: int) -> list:
        rows = self._read_all(SHEET_SENTENCES)
        return [
            r["한국어문장"] for r in rows
            if str(r.get("요일번호", "")) == str(day_number)
            and r.get("평가결과") == "correct"
        ]

    def save_sentence_result(self, day_number: int, korean_sentence: str, evaluation: str):
        if not korean_sentence:
            return
        if evaluation == "correct" and korean_sentence in self.load_correct_sentences(day_number):
            return
        self._append(SHEET_SENTENCES, [day_number, korean_sentence, evaluation])

    # ── Phrase Progress ─────────────────────────────────────────────────

    def load_completed_phrases(self) -> list:
        rows = self._read_all(SHEET_PHRASES)
        return [r["표현"] for r in rows if r.get("표현")]

    def save_phrase_complete(self, phrase: str, category: str):
        if phrase in self.load_completed_phrases():
            return
        self._append(SHEET_PHRASES, [phrase, category, datetime.date.today().isoformat()])

    # ── Cache (daily words / shown words) ────────────────────────────────

    def get_cache(self, key: str):
        rows = self._read_all(SHEET_CACHE)
        for row in rows:
            if row.get("키") == key:
                return row.get("값", "")
        return None

    def set_cache(self, key: str, value: str):
        row_idx = self._find_row_index(SHEET_CACHE, 0, key)
        if row_idx >= 0:
            self._update_cell_range(SHEET_CACHE, f"B{row_idx}", [value])
        else:
            self._append(SHEET_CACHE, [key, value])
