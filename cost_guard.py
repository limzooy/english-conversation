"""OpenAI 일일 비용 상한 가드.

app.py의 단일 OpenAI 클라이언트를 이 래퍼로 감싸면, 앱의 모든 OpenAI 호출
(채팅·번역·TTS·Whisper 등)에 하루 누적 비용 한도가 적용된다. 한도를 넘으면
BudgetExceededError 를 던져 추가 과금을 막는다.

정확한 회계가 아니라 '안전 상한'이 목적이므로, 토큰 사용량(usage)이 있으면
정확히 계산하고 없으면 보수적으로(약간 크게) 추정한다.
"""

# 모델별 단가 (USD, 1토큰당) — (input, output)
_CHAT_PRICING = {
    "gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
}
_DEFAULT_CHAT_PRICING = _CHAT_PRICING["gpt-4o"]  # 알 수 없는 모델은 비싸게 잡아 안전측

_TTS_PER_CHAR = 15.00 / 1_000_000          # tts-1: $15 / 1M characters
_WHISPER_FLAT = 0.006                       # whisper-1: 호출당 보수적 추정($0.006 ≈ 1분)


class BudgetExceededError(Exception):
    """오늘 사용 한도를 초과했을 때 발생."""


def _chat_cost(model, usage):
    in_rate, out_rate = _DEFAULT_CHAT_PRICING
    for name, rates in _CHAT_PRICING.items():
        if model and model.startswith(name):
            in_rate, out_rate = rates
            break
    if usage is None:
        return 0.0
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    return prompt * in_rate + completion * out_rate


class _Create:
    """실제 create 메서드를 감싸 호출 전 한도 확인 + 호출 후 비용 기록."""

    def __init__(self, guard, real_create, cost_fn):
        self._guard = guard
        self._real = real_create
        self._cost_fn = cost_fn

    def create(self, *args, **kwargs):
        self._guard._check_budget()
        response = self._real(*args, **kwargs)
        try:
            self._guard._record(self._cost_fn(response, kwargs))
        except Exception:
            pass  # 회계 실패가 실제 기능을 막지 않도록
        return response


class _Namespace:
    def __init__(self, **children):
        self.__dict__.update(children)


class GuardedOpenAI:
    """OpenAI 클라이언트를 투명하게 감싼다. 앱이 실제로 쓰는 경로만 노출:
    - chat.completions.create
    - audio.transcriptions.create
    - audio.speech.create
    그 외 속성은 원본 클라이언트로 위임한다.
    """

    def __init__(self, client, get_spent, add_spent, daily_budget=1.0):
        self._client = client
        self._get_spent = get_spent
        self._add_spent = add_spent
        self.daily_budget = daily_budget

        self.chat = _Namespace(completions=_Create(
            self, client.chat.completions.create,
            lambda resp, kw: _chat_cost(kw.get("model"), getattr(resp, "usage", None)),
        ))
        self.audio = _Namespace(
            transcriptions=_Create(
                self, client.audio.transcriptions.create,
                lambda resp, kw: _WHISPER_FLAT,
            ),
            speech=_Create(
                self, client.audio.speech.create,
                lambda resp, kw: len(kw.get("input", "")) * _TTS_PER_CHAR,
            ),
        )

    def _check_budget(self):
        if self._get_spent() >= self.daily_budget:
            raise BudgetExceededError(
                f"오늘 API 사용 한도(${self.daily_budget:.2f})를 초과했어요. "
                f"내일 다시 이용해 주세요."
            )

    def _record(self, amount):
        if amount:
            self._add_spent(float(amount))

    def __getattr__(self, name):
        # 감싸지 않은 속성은 원본 클라이언트로 위임
        return getattr(self._client, name)
