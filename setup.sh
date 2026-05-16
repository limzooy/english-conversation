#!/bin/bash
# 영어 회화 프로그램 설치 스크립트

echo "======================================"
echo "  영어 회화 프로그램 설치"
echo "======================================"

# Homebrew 확인
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew가 필요합니다: https://brew.sh"
    exit 1
fi

# portaudio 설치 (pyaudio 의존성)
echo ""
echo "▶ portaudio 설치 중..."
brew install portaudio

# Python 가상환경 생성
echo ""
echo "▶ Python 가상환경 생성 중..."
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
echo ""
echo "▶ Python 패키지 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "======================================"
echo "  설치 완료!"
echo "======================================"
echo ""
echo "다음 단계:"
echo "1. .env.example을 복사해서 .env 파일 생성:"
echo "   cp .env.example .env"
echo ""
echo "2. .env 파일을 열어 API 키 입력:"
echo "   OPENAI_API_KEY=sk-..."
echo ""
echo "3. (선택) Google Sheets 연동:"
echo "   - Google Cloud Console에서 서비스 계정 생성"
echo "   - credentials.json 다운로드 후 이 폴더에 저장"
echo "   - 스프레드시트 ID를 .env에 입력"
echo "   - 스프레드시트를 서비스 계정 이메일과 공유"
echo ""
echo "4. 프로그램 실행:"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
