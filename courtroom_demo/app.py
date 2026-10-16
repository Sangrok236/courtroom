import streamlit as st

import ai
import engine
import flow
from cases.case01 import CASE

BADGES = {'added': '🆕 신규', 'revised': '✏️ 수정됨'}

st.set_page_config(page_title='법정의 모순', page_icon='⚖️', layout='centered')
st.title('⚖️ 법정의 모순')
st.caption(f'{CASE["title"]} | AI 법정 추리 프로토타입')


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
        '지금 보고 있는 증언에 대해 질문하세요. 증인은 한 번에 하나의 사실만 말하고, '
        '그 답변은 이 증언 바로 뒤에 새 증언으로 들어갑니다. 질문만 가능하며 Enter로 제출됩니다.'
    )
    with st.form('ask', clear_on_submit=True):
        question = st.text_input('증인에게 질문', placeholder='예: 복도 조명이 켜져 있었나요?', disabled=over)
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
    elif ai.last_error:
        st.caption(f'⚠️ 최근 AI 오류: {ai.last_error}')
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
        '증인이 한 말을 짚어서, 이 증거와 왜 모순인지 대사로 말하세요. 증언 번호는 말하지 않아도 됩니다. '
        '예: "증인은 9시쯤 복도에서 사람을 마주쳤다고 했고 그때 조명이 켜져 있었다고 했는데, 정전 기록은 그 시각 3층이 정전이었다고 합니다." '
        'Enter로 제출됩니다.'
    )
    with st.form('present'):
        claim = st.text_input('대사', placeholder='증인이 한 말을 짚어서, 이 증거와 왜 모순인지 말하세요', disabled=over)
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
