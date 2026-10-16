"""테스트는 절대 실제 API를 호출하지 않는다: .env 로딩을 끄고, 키 환경 변수를 매 테스트마다 비운다."""
import os
import sys
from pathlib import Path

os.environ['COURTROOM_NO_DOTENV'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

API_ENV = ('GEMINI_API_KEY', 'GOOGLE_API_KEY', 'OPENAI_API_KEY', 'AI_PROVIDER',
           'AI_MODEL', 'GEMINI_MODEL', 'OPENAI_MODEL')


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    for name in API_ENV:
        monkeypatch.delenv(name, raising=False)
