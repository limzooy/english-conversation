"""Google Sheets 연동 모듈"""

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SHEET_NAME = "영어회화"

# 스프레드시트 헤더 정의
HEADERS = [
    "날짜/시간",
    "세션 ID",
    "내 영어 문장",
    "수정된 문장",
    "수정 필요",
    "수정 설명",
    "AI 응답",
    "발음 팁",
]


class SheetsHandler:
    def __init__(self, credentials_file: str, spreadsheet_id: str):
        self.spreadsheet_id = spreadsheet_id
        creds = Credentials.from_service_account_file(credentials_file, scopes=SCOPES)
        self.service = build("sheets", "v4", credentials=creds)
        self.sheet = self.service.spreadsheets()
        self._ensure_sheet_exists()

    def _ensure_sheet_exists(self):
        """시트가 없으면 생성하고 헤더 추가"""
        try:
            # 시트 목록 확인
            spreadsheet = self.sheet.get(spreadsheetId=self.spreadsheet_id).execute()
            sheet_names = [s["properties"]["title"] for s in spreadsheet["sheets"]]

            if SHEET_NAME not in sheet_names:
                # 새 시트 생성
                body = {
                    "requests": [
                        {
                            "addSheet": {
                                "properties": {"title": SHEET_NAME}
                            }
                        }
                    ]
                }
                self.sheet.batchUpdate(
                    spreadsheetId=self.spreadsheet_id, body=body
                ).execute()

                # 헤더 추가
                self._append_row(HEADERS)
                self._format_header()
            else:
                # 헤더가 없으면 추가
                result = self.sheet.values().get(
                    spreadsheetId=self.spreadsheet_id,
                    range=f"{SHEET_NAME}!A1:H1",
                ).execute()
                if not result.get("values"):
                    self._append_row(HEADERS)
                    self._format_header()
        except Exception as e:
            raise RuntimeError(f"시트 초기화 실패: {e}")

    def _format_header(self):
        """헤더 행 볼드 처리"""
        try:
            # 시트 ID 가져오기
            spreadsheet = self.sheet.get(spreadsheetId=self.spreadsheet_id).execute()
            sheet_id = None
            for s in spreadsheet["sheets"]:
                if s["properties"]["title"] == SHEET_NAME:
                    sheet_id = s["properties"]["sheetId"]
                    break

            if sheet_id is None:
                return

            body = {
                "requests": [
                    {
                        "repeatCell": {
                            "range": {
                                "sheetId": sheet_id,
                                "startRowIndex": 0,
                                "endRowIndex": 1,
                            },
                            "cell": {
                                "userEnteredFormat": {
                                    "textFormat": {"bold": True},
                                    "backgroundColor": {
                                        "red": 0.2,
                                        "green": 0.5,
                                        "blue": 0.8,
                                    },
                                }
                            },
                            "fields": "userEnteredFormat(textFormat,backgroundColor)",
                        }
                    }
                ]
            }
            self.sheet.batchUpdate(
                spreadsheetId=self.spreadsheet_id, body=body
            ).execute()
        except Exception:
            pass  # 포맷 실패는 무시

    def _append_row(self, values: list):
        body = {"values": [values]}
        self.sheet.values().append(
            spreadsheetId=self.spreadsheet_id,
            range=f"{SHEET_NAME}!A1",
            valueInputOption="USER_ENTERED",
            body=body,
        ).execute()

    def append_row(
        self,
        session_id: str,
        timestamp: str,
        user_sentence: str,
        corrected_sentence: str,
        correction_explanation: str,
        ai_response: str,
        pronunciation_tip: str,
        has_correction: bool,
    ):
        """대화 한 줄을 스프레드시트에 추가"""
        row = [
            timestamp,
            session_id,
            user_sentence,
            corrected_sentence if has_correction else "",
            "O" if has_correction else "",
            correction_explanation if has_correction else "",
            ai_response,
            pronunciation_tip,
        ]
        self._append_row(row)
