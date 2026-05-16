"""Google Sheets 미연결 시 로컬 CSV 저장 모듈"""

import csv
import os
from datetime import datetime

CSV_FILE = "conversation_history.csv"
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


class LocalSaver:
    def __init__(self, directory: str = "."):
        self.filepath = os.path.join(directory, CSV_FILE)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(HEADERS)

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
        with open(self.filepath, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                session_id,
                user_sentence,
                corrected_sentence if has_correction else "",
                "O" if has_correction else "",
                correction_explanation if has_correction else "",
                ai_response,
                pronunciation_tip,
            ])
