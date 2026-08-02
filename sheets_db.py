"""Google Sheets 데이터베이스 - 모든 CSV 데이터를 Sheets로 관리"""

import json
import os
import datetime
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

SHEET_CONV      = "대화기록"
SHEET_MEMO      = "메모"
SHEET_CACHE     = "캐시"

# 현재 앱이 사용하는 시트만 관리한다. 훈련진행/훈련문장/표현완료는 해당 모드가
# 제거되어 더 이상 생성·검증하지 않는다(기존 데이터는 스프레드시트에 그대로 남음).
ALL_SHEETS = {
    SHEET_CONV:      ["날짜/시간", "세션 ID", "내 영어 문장", "수정된 문장", "수정 필요", "수정 설명", "AI 응답", "발음 팁"],
    SHEET_MEMO:      ["날짜", "메모"],
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
    # 스키마 검증(시트 생성 + 헤더 확인)은 Google API 왕복이 2~3회 필요해 콜드스타트를
    # 크게 느리게 만든다. 읽기에는 필요 없으므로 '첫 쓰기 직전'에 프로세스당 한 번만 수행한다.
    _schema_ready = False

    def __init__(self, spreadsheet_id: str):
        self.spreadsheet_id = spreadsheet_id
        self._svc = _build_service()
        self._sheet = self._svc.spreadsheets()

    def _ensure_ready(self):
        """쓰기 전에 시트/헤더 존재를 보장한다(프로세스당 1회, best-effort)."""
        if SheetsDB._schema_ready:
            return
        # 플래그를 먼저 세워 _ensure_sheets 내부의 _append 재진입을 막는다.
        SheetsDB._schema_ready = True
        try:
            self._ensure_sheets()
        except Exception:
            pass  # 시트가 이미 있으면 검증 실패해도 읽기/쓰기는 가능

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
        self._ensure_ready()
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
        self._ensure_ready()
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

    # ── Cache ─────────────────────────────────────────────────────────

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
