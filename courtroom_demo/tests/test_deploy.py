"""인터넷 공개 배포용 보호 장치: 호출 예산, 공개 모드의 오류 숨김, 암호, Secrets 연동, 의존성 범위."""
import datetime
import importlib.metadata
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ai
import budget
import engine
import flow
from cases.case01 import CASE

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / 'app.py')


# ---------- 공개 모드 / 예산 ----------

def test_public_mode_is_off_by_default_and_reads_common_truthy_values(monkeypatch):
    assert budget.public_mode() is False
    for value in ('1', 'true', 'TRUE', ' yes ', 'on'):
        monkeypatch.setenv('COURTROOM_PUBLIC', value)
        assert budget.public_mode() is True
    for value in ('0', 'false', '', 'no'):
        monkeypatch.setenv('COURTROOM_PUBLIC', value)
        assert budget.public_mode() is False


def test_session_budget_runs_out_after_its_limit():
    b = budget.CallBudget(2, 100, daily=budget.DailyCounter())
    assert b.available() and b.reason() == ''
    b.spend()
    b.spend()
    assert not b.available() and b.reason() == 'session'


def test_daily_budget_is_shared_by_every_session_and_resets_the_next_day():
    day = [datetime.date(2026, 10, 16)]
    daily = budget.DailyCounter(today=lambda: day[0])
    a, b = budget.CallBudget(50, 3, daily=daily), budget.CallBudget(50, 3, daily=daily)
    a.spend()
    b.spend()
    a.spend()
    assert not a.available() and not b.available() and b.reason() == 'daily'   # 다른 세션이 써도 같이 바닥난다
    day[0] = datetime.date(2026, 10, 17)
    assert a.available() and b.available() and daily.used == 0                  # 날짜가 바뀌면 다시 열린다


def test_budget_limits_come_from_the_environment_with_safe_defaults(monkeypatch):
    b = budget.CallBudget.from_env()
    assert (b.session_limit, b.daily_limit) == (budget.SESSION_CALLS, budget.DAILY_CALLS)
    monkeypatch.setenv('COURTROOM_SESSION_CALLS', '7')
    monkeypatch.setenv('COURTROOM_DAILY_CALLS', '99')
    b = budget.CallBudget.from_env()
    assert (b.session_limit, b.daily_limit) == (7, 99)
    for bad in ('abc', '', '0', '-5'):
        monkeypatch.setenv('COURTROOM_SESSION_CALLS', bad)
        assert budget.CallBudget.from_env().session_limit == budget.SESSION_CALLS   # 잘못된 값은 기본값으로


def test_without_an_active_budget_nothing_is_limited(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    budget.activate(None)
    for _ in range(1000):
        budget.spend()
    assert budget.available() and ai.has_api_key() and budget.notice() == ''


def test_has_api_key_turns_false_when_the_budget_is_exhausted(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    b = budget.CallBudget(1, 100, daily=budget.DailyCounter())
    budget.activate(b)
    assert ai.has_api_key()
    b.spend()
    assert not ai.has_api_key()                      # 키가 있어도 예산이 없으면 간이 모드
    monkeypatch.delenv('GEMINI_API_KEY')
    budget.activate(budget.CallBudget(5, 5, daily=budget.DailyCounter()))
    assert not ai.has_api_key()                      # 예산이 있어도 키가 없으면 간이 모드


def test_every_ai_call_is_counted_even_when_it_fails(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    b = budget.CallBudget(10, 100, daily=budget.DailyCounter())
    budget.activate(b)
    monkeypatch.setattr(ai, '_call_gemini', lambda *a, **k: {'ok': True})
    ai._call_json('i', 'u', 'x', {}, 10, 0)
    monkeypatch.setattr(ai, '_call_gemini', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('x')))
    with pytest.raises(RuntimeError):
        ai._call_json('i', 'u', 'x', {}, 10, 0)
    assert b.used == 2 and b.daily.used >= 2


def test_when_the_budget_is_gone_the_game_keeps_working_without_calling_the_ai(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    monkeypatch.setattr(ai, '_call_gemini', lambda *a, **k: pytest.fail('예산이 없으면 AI를 부르면 안 된다'))
    b = budget.CallBudget(0 + 1, 100, daily=budget.DailyCounter())
    b.used = 1
    budget.activate(b)
    state = engine.new_state(CASE)
    new_id = flow.ask_question(CASE, state, 1, '복도 조명이 켜져 있었나요?')
    assert '밝' in engine.get(state, new_id).text                              # 키워드 기반 간이 답변


def test_notice_tells_the_player_which_limit_was_hit():
    assert budget.notice() == ''
    session = budget.CallBudget(1, 100, daily=budget.DailyCounter())
    session.used = 1
    budget.activate(session)
    assert '이번 접속' in budget.notice() and '간이 모드' in budget.notice()
    daily = budget.CallBudget(100, 1, daily=budget.DailyCounter())
    daily.spend()
    budget.activate(daily)
    assert '오늘' in budget.notice() and '내일 다시' in budget.notice()


# ---------- 공개 모드의 오류 숨김 / 캐시 상한 ----------

def test_public_mode_hides_error_details_from_players(monkeypatch):
    monkeypatch.setattr(ai, 'last_error', "[witness_reply] ClientError 403: Vertex API has not been used in project 1010815321808")
    assert 'project' in ai.visible_error()                                      # 로컬에서는 진단용으로 그대로 보인다
    monkeypatch.setenv('COURTROOM_PUBLIC', '1')
    shown = ai.visible_error()
    assert '1010815321808' not in shown and 'ClientError' not in shown and '잠시 후' in shown
    monkeypatch.setattr(ai, 'last_error', '')
    assert ai.visible_error() == ''                                              # 오류가 없으면 아무것도 보이지 않는다


def test_public_mode_ask_error_does_not_leak_details(monkeypatch):
    monkeypatch.setenv('COURTROOM_PUBLIC', '1')
    monkeypatch.setattr(ai, 'last_error', '[witness_reply] ClientError 403: project 1010815321808')
    monkeypatch.setattr(ai, 'witness_reply', lambda *a, **k: None)
    with pytest.raises(flow.AskError) as error:
        flow.ask_question(CASE, engine.new_state(CASE), 1, '질문')
    assert '1010815321808' not in str(error.value) and '잠시 후' in str(error.value)


def test_the_judge_cache_is_bounded(monkeypatch):
    ai._contradiction_cache.clear()
    for i in range(ai._CACHE_MAX):
        ai._contradiction_cache[(f's{i}',)] = {}
    ai.judge_argument(CASE, ['복도 조명은 밝았다.'], '증인은 조명이 밝았다고 했는데 정전입니다.', 'blackout')
    assert len(ai._contradiction_cache) <= 2


# ---------- 앱: 암호 / Secrets / 예산 연결 ----------

def run_app(**secrets):
    at = AppTest.from_file(APP, default_timeout=30)
    for key, value in secrets.items():
        at.secrets[key] = value
    return at.run()


def test_without_a_password_the_game_is_open():
    at = run_app()
    assert not at.exception and any('법정 상황' in s.value for s in at.subheader)
    assert 'budget' not in at.session_state                                      # 로컬 실행은 예산이 없다


def test_password_gate_hides_the_game_until_the_right_password(monkeypatch):
    monkeypatch.setenv('COURTROOM_PASSWORD', 'open-sesame')
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    from streamlit.proto.TextInput_pb2 import TextInput
    assert [t.label for t in at.text_input] == ['접속 암호']
    assert at.text_input[0].proto.type == TextInput.PASSWORD                  # 입력한 암호가 화면에 보이지 않는다
    assert not list(at.subheader) and not [b for b in at.button if b.label == '질문하기']   # 게임 화면은 보이지 않는다

    at.text_input[0].set_value('wrong').run()
    [b for b in at.button if b.label == '입장'][0].click().run()
    assert any('맞지 않습니다' in e.value for e in at.error) and not list(at.subheader)

    at.text_input[0].set_value('open-sesame')
    [b for b in at.button if b.label == '입장'][0].click().run()
    assert not at.exception and any('법정 상황' in s.value for s in at.subheader)
    assert at.session_state['authed'] is True


def test_streamlit_secrets_are_copied_into_the_environment_without_overriding_it(monkeypatch):
    for name in ('COURTROOM_PASSWORD', 'SOME_FROM_SECRETS'):
        monkeypatch.setenv(name, 'tmp')
        monkeypatch.delenv(name)                      # 테스트가 끝나면 앱이 넣은 값도 지워지도록 원래 상태를 기록해 둔다
    at = run_app(COURTROOM_PASSWORD='from-secrets', SOME_FROM_SECRETS='1')
    assert not at.exception
    import os
    assert os.environ['SOME_FROM_SECRETS'] == '1'
    assert [t.label for t in at.text_input] == ['접속 암호']        # Secrets의 암호가 실제로 적용되었다
    os.environ.pop('COURTROOM_PASSWORD', None)

    monkeypatch.setenv('COURTROOM_PASSWORD', 'from-env')
    at = run_app(COURTROOM_PASSWORD='from-secrets')
    assert os.environ['COURTROOM_PASSWORD'] == 'from-env'            # 이미 있는 환경 변수가 우선한다


def test_public_mode_gives_each_session_a_budget_that_survives_a_restart_of_the_game(monkeypatch):
    monkeypatch.setenv('COURTROOM_PUBLIC', '1')
    monkeypatch.setenv('COURTROOM_SESSION_CALLS', '5')
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    session_budget = at.session_state['budget']
    assert session_budget.session_limit == 5
    session_budget.used = 3
    at.run()
    assert at.session_state['budget'] is session_budget and session_budget.used == 3   # 다시 그려도 예산은 그대로


def test_the_player_sees_a_notice_when_the_budget_is_gone(monkeypatch):
    monkeypatch.setenv('COURTROOM_PUBLIC', '1')
    monkeypatch.setenv('COURTROOM_SESSION_CALLS', '5')
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not list(at.warning)
    at.session_state['budget'].used = 5
    at.run()
    assert not at.exception and any('간이 모드' in w.value for w in at.warning)


# ---------- 배포 파일 ----------

def lower_bound(line):
    match = re.match(r'([A-Za-z0-9_.-]+)\s*>=\s*([\d.]+)', line)
    return match.group(1), tuple(int(part) for part in match.group(2).split('.'))


def test_requirements_are_pinned_to_ranges_that_were_actually_tested():
    """Streamlit Cloud는 requirements.txt로 설치한다. 시험하지 않은 낮은 버전이 깔려 앱이 깨지는 일을 막는다."""
    lines = [l.strip() for l in (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines() if l.strip()]
    bounds = dict(lower_bound(line) for line in lines)
    assert set(bounds) == {'streamlit', 'google-genai', 'openai', 'python-dotenv'}
    for name, minimum in bounds.items():
        installed = tuple(int(p) for p in re.findall(r'\d+', importlib.metadata.version(name))[:3])
        assert installed >= minimum, f'{name} {installed}가 requirements의 하한 {minimum}보다 낮다'
    assert bounds['google-genai'] >= (2, 31)      # ThinkingLevel(Gemini 3 생각 설정)을 쓴다
    assert all('<' in line for line in lines)      # 다음 메이저 버전으로 자동 업그레이드되지 않게 상한도 둔다


def test_secrets_and_env_files_are_ignored_by_git_and_an_example_exists():
    ignored = (ROOT / '.gitignore').read_text(encoding='utf-8').splitlines()
    assert '.env' in ignored and '.streamlit/secrets.toml' in ignored
    example = (ROOT / '.streamlit' / 'secrets.toml.example').read_text(encoding='utf-8')
    assert 'COURTROOM_PUBLIC' in example and 'GEMINI_API_KEY' in example
    for line in example.splitlines():
        assert not re.search(r'(AIza|AQ\.)[A-Za-z0-9_-]{10,}', line), '예시 파일에 실제 키처럼 보이는 값이 있다'
