import hmac
import os

import streamlit as st

import ai
import budget
import engine
import flow
from cases.case01 import CASE

BADGES = {'added': '🆕 신규', 'revised': '✏️ 수정됨'}

st.set_page_config(page_title='법정의 모순', page_icon='⚖️', layout='centered')


def load_deploy_secrets():
    """Streamlit Cloud의 Secrets(.streamlit/secrets.toml)를 환경 변수로 옮긴다. 로컬 .env나 이미 설정된 환경 변수가 우선한다."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ.setdefault(key, str(value))
    except Exception:  # 로컬 실행처럼 secrets가 없으면 건너뛴다.
        pass


def require_password():
    """COURTROOM_PASSWORD가 설정되어 있으면 암호를 맞춰야 게임이 보인다(공개 링크를 아는 사람 중 일부만 들이고 싶을 때)."""
    password = os.getenv('COURTROOM_PASSWORD', '')
    if not password or st.session_state.get('authed'):
        return
    st.title('⚖️ 법정의 모순')
    with st.form('login'):
        entered = st.text_input('접속 암호', type='password')
        submitted = st.form_submit_button('입장')
    if submitted and hmac.compare_digest(entered.encode(), password.encode()):
        st.session_state.authed = True
        st.rerun()
    if submitted:
        st.error('암호가 맞지 않습니다.')
    st.stop()


load_deploy_secrets()
require_password()

# 공개 배포에서는 세션마다 AI 호출 예산을 두고(서버 전체 하루 한도도 함께), 바닥나면 간이 모드로 이어 간다.
# '처음부터 다시'로 예산이 새로 생기지 않도록 예산은 세션에 한 번만 만든다.
if budget.public_mode():
    if 'budget' not in st.session_state:
        st.session_state.budget = budget.CallBudget.from_env()
    budget.activate(st.session_state.budget)
else:
    budget.activate(None)

st.title('⚖️ 법정의 모순')
st.caption(f'{CASE["title"]} | AI 법정 추리 프로토타입')
if budget.notice():
    st.warning(budget.notice())


def reset_game():
    for key in ('state', 'current_id', 'ask_error', 'last_result', 'celebrated'):
        st.session_state.pop(key, None)


def go_to(statement_id):
    st.session_state.current_id = statement_id
    st.session_state.pop('ask_error', None)
    st.rerun()


if 'state' not in st.session_state:
    st.session_state.state = engine.new_state(CASE)
state = st.session_state.state
over = state.solved or state.game_over  # 클리어했거나 재판관이 심문을 종결했으면 더 이상 질문·증거 제시를 받지 않는다.
statements = engine.active(state)
# 증언이 철회되어도 보던 증언을 잃지 않도록 위치가 아니라 id로 현재 증언을 추적한다.
current_id = engine.successor(state, st.session_state.get('current_id', statements[0].id)) or statements[0].id
st.session_state.current_id = current_id
index = next(i for i, s in enumerate(statements) if s.id == current_id)
current = statements[index]

st.subheader('법정 상황')
st.info(CASE['briefing'])
st.markdown(f'**증인 · {CASE["witness"]["name"]}**')

badges = [BADGES[current.status]] if current.status in BADGES else []
st.caption(f'증언 {index + 1}/{len(statements)} · 증언 한도 {state.limit} ' + ' '.join(badges))
st.success(current.text)
if current.history:
    with st.expander('이전에 한 말'):
        for old in current.history:
            st.write(f'~~{old}~~')

c1, c2 = st.columns(2)
with c1:
    if st.button('◀ 이전 증언', use_container_width=True):
        go_to(statements[(index - 1) % len(statements)].id)
with c2:
    if st.button('다음 증언 ▶', use_container_width=True):
        go_to(statements[(index + 1) % len(statements)].id)

tab1, tab2, tab3 = st.tabs(['💬 질문', '📁 증거 제시', '📜 법정 기록'])
with tab1:
    st.caption(
        '지금 보고 있는 증언에 대해 질문하세요.'
        '답변은 이 증언 바로 뒤에 새 증언으로 들어갑니다. Enter로 제출됩니다.'
    )
    with st.form('ask', clear_on_submit=True):
        question = st.text_input('증인에게 질문', placeholder='질문을 입력', disabled=over)
        submitted = st.form_submit_button('질문하기', disabled=over)
    if submitted and question.strip():
        try:
            with st.spinner('증인이 답변하는 중...'):
                st.session_state.current_id = flow.ask_question(CASE, state, current.id, question.strip())
            st.session_state.pop('ask_error', None)
        except flow.AskError as exc:
            st.session_state.ask_error = str(exc)
        st.rerun()
    if st.session_state.get('ask_error'):
        st.error(st.session_state.ask_error)
    elif ai.visible_error():
        st.caption(f'⚠️ 최근 AI 오류: {ai.visible_error()}')
with tab2:
    evidence_id = st.selectbox(
        '제시할 증거를 선택하세요',
        list(CASE['evidence']),
        format_func=lambda eid: CASE['evidence'][eid]['name'],
    )
    st.caption(CASE['evidence'][evidence_id]['description'])
    st.markdown('**현재 증언**')
    for statement in statements:
        st.markdown(f'- {statement.text}')
    st.caption(
        '증인이 한 말을 짚어서, 이 증거와 왜 모순인지 대사로 말하세요. '
        'Enter로 제출됩니다.'
    )
    with st.form('present'):
        claim = st.text_input('대사', placeholder='증인이 한 말을 짚어서, 증거를 토대로 모순을 지적하세요', disabled=over)
        presented = st.form_submit_button('증거 제시', disabled=over)
    if presented:
        with st.spinner('재판부가 판단하는 중...'):
            st.session_state.last_result = flow.present_evidence(CASE, state, claim, evidence_id)
        st.rerun()
    if 'last_result' in st.session_state:
        kind, message = st.session_state.last_result
        getattr(st, kind)(message)
with tab3:
    st.markdown(f'**사건 개요**\n\n{CASE["summary"]}')
    for evidence in CASE['evidence'].values():
        with st.expander(evidence['name']):
            st.write(evidence['description'])
    st.markdown('**반박된 증언** (증인은 이 주장을 다시 하지 못합니다)')
    if not state.refuted:
        st.caption('아직 반박된 증언이 없습니다.')
    for text in state.refuted:
        st.write(f'~~{text}~~')
    st.markdown('**증언 변경 기록**')
    if not state.events:
        st.caption('아직 기록이 없습니다.')
    for event in state.events:
        st.write(event)

if state.solved:
    if not st.session_state.get('celebrated'):
        st.balloons()
        st.session_state.celebrated = True
    st.divider()
    st.header('🎉 1차 심문 성공')
    st.write(CASE['ending'])
    st.caption('다만 이것만으로 무죄가 확정되지는 않습니다. 데모는 여기까지입니다.')
    if st.button('처음부터 다시'):
        reset_game()
        st.rerun()
elif state.game_over:
    st.divider()
    st.header('❌ 게임 오버')
    st.error(CASE['game_over'])
    if st.button('처음부터 다시'):
        reset_game()
        st.rerun()
