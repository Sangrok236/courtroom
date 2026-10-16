"""공개 배포용 설정과 AI 호출 예산.

인터넷에 공개하면 링크를 아는 누구나 내 API 키 비용을 쓰게 된다. 그래서 공개 모드(`COURTROOM_PUBLIC=1`)에서는
  - 세션(접속 1회)마다 AI 호출 수를 제한하고,
  - 서버 전체의 하루 AI 호출 수도 제한한다.
한도를 넘으면 AI 대신 키 없이 쓰는 키워드 기반 간이 모드로 이어서 진행한다(`ai.has_api_key()`가 False가 된다).
로컬 실행(공개 모드가 아님)에서는 아무 제한도 없다.
"""
from __future__ import annotations

import contextvars
import datetime
import os
import threading

SESSION_CALLS = 120   # 세션당 AI 호출 수(질문 한 번 1~2회, 증거 제시 한 번 3~6회)
DAILY_CALLS = 1500    # 서버 전체 하루 AI 호출 수
_TRUE = {'1', 'true', 'yes', 'on'}


def public_mode() -> bool:
    return os.getenv('COURTROOM_PUBLIC', '').strip().lower() in _TRUE


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, '').strip())
    except ValueError:
        return default
    return value if value > 0 else default


class DailyCounter:
    """서버 프로세스 전체가 공유하는 하루 호출 수. 날짜가 바뀌면 0으로 돌아간다(서버가 재시작되어도 0으로 돌아간다)."""

    def __init__(self, today=datetime.date.today):
        self._today = today
        self._lock = threading.Lock()
        self._day = None
        self.used = 0

    def _roll(self) -> None:
        day = self._today()
        if day != self._day:
            self._day, self.used = day, 0

    def available(self, limit: int) -> bool:
        with self._lock:
            self._roll()
            return self.used < limit

    def spend(self) -> None:
        with self._lock:
            self._roll()
            self.used += 1


DAILY = DailyCounter()


class CallBudget:
    """세션 하나의 AI 호출 예산. 세션 한도와 서버 전체 하루 한도를 함께 본다."""

    def __init__(self, session_limit: int, daily_limit: int, daily: DailyCounter = DAILY):
        self.session_limit = session_limit
        self.daily_limit = daily_limit
        self.daily = daily
        self.used = 0

    @classmethod
    def from_env(cls) -> 'CallBudget':
        return cls(_int_env('COURTROOM_SESSION_CALLS', SESSION_CALLS), _int_env('COURTROOM_DAILY_CALLS', DAILY_CALLS))

    def available(self) -> bool:
        return self.used < self.session_limit and self.daily.available(self.daily_limit)

    def spend(self) -> None:
        self.used += 1
        self.daily.spend()

    def reason(self) -> str:
        """한도에 닿았으면 'session' 또는 'daily', 아니면 ''."""
        if self.used >= self.session_limit:
            return 'session'
        if not self.daily.available(self.daily_limit):
            return 'daily'
        return ''


# 지금 실행 중인 스크립트(세션)의 예산. Streamlit은 세션마다 스레드에서 스크립트를 돌리므로 매 실행 시작에 설정한다.
_current: contextvars.ContextVar[CallBudget | None] = contextvars.ContextVar('courtroom_budget', default=None)


def activate(budget: CallBudget | None) -> None:
    _current.set(budget)


def available() -> bool:
    budget = _current.get()
    return True if budget is None else budget.available()


def spend() -> None:
    budget = _current.get()
    if budget is not None:
        budget.spend()


def notice() -> str:
    """한도에 닿았을 때 화면에 보여 줄 안내. 아니면 빈 문자열."""
    budget = _current.get()
    reason = budget.reason() if budget is not None else ''
    if reason == 'session':
        return '이번 접속의 AI 사용 한도에 도달해 이후에는 키워드 기반 간이 모드로 진행됩니다.'
    if reason == 'daily':
        return '오늘 이 게임의 AI 사용 한도에 도달해 이후에는 키워드 기반 간이 모드로 진행됩니다. 내일 다시 이용해 주세요.'
    return ''
