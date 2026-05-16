"""음성 입출력 모듈 (OpenAI Whisper + TTS)"""

import os
import subprocess
import tempfile
import threading
import wave
import pyaudio
from openai import OpenAI


class SpeechHandler:
    def __init__(self, client: OpenAI):
        self.client = client
        self.pa = pyaudio.PyAudio()
        print("✅ 마이크 준비 완료")

    def listen(self, timeout: int = 10, phrase_limit: int = 30) -> str:
        """마이크로 음성을 녹음하고 Whisper로 텍스트 변환 (Enter로 종료)"""
        print("\n🎤 녹음 시작... (말씀하신 후 Enter를 누르세요)")

        frames = []
        stop_event = threading.Event()
        rate = 16000
        chunk = 1024

        def record():
            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                frames_per_buffer=chunk,
            )
            while not stop_event.is_set():
                frames.append(stream.read(chunk, exception_on_overflow=False))
            stream.stop_stream()
            stream.close()

        record_thread = threading.Thread(target=record, daemon=True)
        record_thread.start()

        input()  # Enter 입력 대기
        stop_event.set()
        record_thread.join()

        if not frames:
            return ""

        print("🔄 음성 인식 중...")

        # WAV 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

        try:
            with open(temp_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                )
            text = transcript.text.strip()
            return text
        except Exception as e:
            print(f"❌ 음성 인식 실패: {e}")
            return ""
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def speak(self, text: str, voice: str = "nova"):
        """OpenAI TTS로 텍스트를 음성으로 출력"""
        if not text.strip():
            return

        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice,  # alloy, echo, fable, onyx, nova, shimmer
                input=text,
            )

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(response.content)
                temp_path = f.name

            # macOS: afplay로 재생
            subprocess.run(
                ["afplay", temp_path],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            pass  # 오디오 재생 실패 시 무시 (텍스트는 이미 출력됨)
        except Exception as e:
            print(f"⚠️  음성 출력 실패: {e}")
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
