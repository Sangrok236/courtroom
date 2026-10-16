import pytest

import ai
import engine
import flow
from cases.case01 import CASE


def fresh(limit=None):
    case = CASE if limit is None else {**CASE, 'statement_limit': limit}
    return engine.new_state(case)


def ids(state):
    return [s.id for s in engine.active(state)]


def texts(state):
    return [s.text for s in engine.active(state)]


# ---------- 초기 상태: 3층 구조 ----------

def test_game_starts_with_only_the_introduction():
    state = fresh()
    assert texts(state) == [CASE['witness']['introduction']]
    assert state.view == CASE['witness']['view'] and state.refuted == []
    assert not state.solved and not state.game_over


def test_state_view_is_a_deep_copy_so_refutation_does_not_mutate_case_data():
    original = [dict(item) for item in CASE['witness']['view']]
    state = fresh()
    state.view[0]['stance'] = '애매'           # 항목 자체를 고쳐도
    state.view.append({'text': 'x', 'stance': '확신'})
    engine.set_view(state, [{'text': '바뀐 믿음', 'stance': '애매'}])
    assert CASE['witness']['view'] == original  # 사건 데이터는 그대로다


def test_statements_carry_no_truth_values_and_no_merge_leftovers():
    fields = set(engine.Statement.__dataclass_fields__)
    assert not fields & {'is_lie', 'key', 'exposed', 'true', 'false', 'merged_into'}
    assert not hasattr(engine, 'merge_into') and not hasattr(engine, 'discard')
    assert not hasattr(flow, 'resolve') and not hasattr(ai, 'merge_statements')
    assert 'pending' not in engine.GameState.__dataclass_fields__


# ---------- view 항목의 태도(stance) ----------

def test_normalize_view_accepts_strings_and_dicts():
    items = ['문자열', {'text': '딕셔너리', 'stance': '숨김'}, {'text': '추측함', 'stance': '추측'}]
    assert engine.normalize_view(items) == [
        {'text': '문자열', 'stance': '확신'},
        {'text': '딕셔너리', 'stance': '숨김'},
        {'text': '추측함', 'stance': '추측'},
    ]


def test_normalize_view_replaces_unknown_stance_and_drops_garbage():
    items = [{'text': 'a', 'stance': '모름'}, {'text': 'b'}, {'text': '  '}, {'stance': '확신'}, 3, None, '']
    assert engine.normalize_view(items) == [{'text': 'a', 'stance': '확신'}, {'text': 'b', 'stance': '확신'}]
    assert engine.normalize_view(items, fallback='애매')[0]['stance'] == '애매'
    assert engine.normalize_view('x') == [] and engine.normalize_view(None) == []


def test_normalize_view_clips_text_and_caps_length():
    out = engine.normalize_view([{'text': '가' * 500, 'stance': '확신'}] + ['a'] * 30)
    assert len(out) == engine.MAX_VIEW and len(out[0]['text']) == engine.MAX_TEXT


def test_stances_are_the_keys_the_author_uses():
    assert set(engine.STANCES) == {'확신', '애매', '숨김', '추측', '미언급'}
    assert engine.DEFAULT_STANCE in engine.STANCES


def test_non_volunteering_stance_is_accepted_and_kept():
    items = engine.normalize_view([{'text': '말 안 한 사실', 'stance': '미언급'}])
    assert items == [{'text': '말 안 한 사실', 'stance': '미언급'}]
    state = fresh()
    assert engine.set_view(state, items) and state.view[0]['stance'] == '미언급'   # AI가 돌려줘도 애매로 바뀌지 않는다


# ---------- 한 질문에 하나의 사실: first_sentence / one_fact ----------

def test_first_sentence_keeps_only_the_first_sentence():
    assert engine.first_sentence('9시쯤이었습니다. 시계는 확인하지 못했습니다.') == '9시쯤이었습니다.'
    assert engine.first_sentence('정말입니까? 네, 그렇습니다.') == '정말입니까?'
    assert engine.first_sentence('봤습니다! 확실합니다.') == '봤습니다!'
    assert engine.first_sentence('봤습니다。 확실합니다。') == '봤습니다。'


def test_first_sentence_handles_edge_cases():
    assert engine.first_sentence('문장부호가 없는 답변') == '문장부호가 없는 답변'
    assert engine.first_sentence('3.5층에서 봤습니다.') == '3.5층에서 봤습니다.'   # 숫자 안의 점은 문장 끝이 아니다
    assert engine.first_sentence('  줄바꿈이\n있어도   한 문장입니다.  다음 문장.') == '줄바꿈이 있어도 한 문장입니다.'
    assert engine.first_sentence('') == '' and engine.first_sentence('   ') == ''


def test_one_fact_is_first_sentence_within_length_limit():
    assert engine.one_fact('가' * 500 + '. 다음.') == engine.clip('가' * 500 + '.')
    assert engine.one_fact('하나. 둘.') == '하나.'


# ---------- add_after ----------

def test_add_after_inserts_right_after_the_questioned_statement():
    state = fresh()
    a = engine.add_after(state, 1, '첫째')
    b = engine.add_after(state, 1, '둘째')  # 1번에서 다시 질문하면 1번 바로 뒤로 들어간다
    assert ids(state) == [1, b, a]
    assert engine.get(state, a).status == 'added'


def test_add_after_stores_only_the_first_sentence():
    state = fresh()
    new_id = engine.add_after(state, 1, '9시쯤이었습니다. 그리고 조명도 켜져 있었습니다.')
    assert engine.get(state, new_id).text == '9시쯤이었습니다.'
    assert '새 증언' in state.events[-1] and '조명' not in state.events[-1]


def test_add_after_skips_over_retracted_entries_between():
    state = fresh()
    a = engine.add_after(state, 1, '하나')
    engine.retract(state, a)
    b = engine.add_after(state, 1, '둘')
    assert ids(state) == [1, b]


def test_add_after_rejects_bad_input():
    state = fresh()
    assert engine.add_after(state, 99, 'x') is None
    assert engine.add_after(state, 1, '   ') is None
    assert ids(state) == [1]


def test_add_after_clips_long_text():
    state = fresh()
    text = engine.get(state, engine.add_after(state, 1, '가' * 500)).text
    assert len(text) == engine.MAX_TEXT and text.endswith('…')


# ---------- 증언 상한과 게임오버 ----------

def test_limit_comes_from_the_case_and_defaults_otherwise():
    assert fresh().limit == CASE['statement_limit']
    assert fresh(limit=5).limit == 5
    no_limit = {k: v for k, v in CASE.items() if k != 'statement_limit'}
    assert engine.new_state(no_limit).limit == engine.DEFAULT_LIMIT


def test_game_over_when_the_statement_count_reaches_the_limit():
    state = fresh(limit=3)
    engine.add_after(state, 1, '둘째')
    assert not state.game_over and len(engine.active(state)) == 2   # 상한 직전에는 아니다
    engine.add_after(state, 1, '셋째')
    assert state.game_over and len(engine.active(state)) == 3


def test_nothing_can_be_added_after_game_over():
    state = fresh(limit=2)
    engine.add_after(state, 1, '둘째')
    assert state.game_over
    assert engine.add_after(state, 1, '셋째') is None and len(engine.active(state)) == 2


def test_no_game_over_if_the_game_is_already_solved():
    state = fresh(limit=2)
    state.solved = True
    assert engine.add_after(state, 1, '둘째') is None and not state.game_over


def test_retraction_makes_room_but_does_not_undo_game_over():
    state = fresh(limit=3)
    a = engine.add_after(state, 1, '둘째')
    engine.retract(state, a)                       # 반박으로 줄어든 뒤에는 다시 여유가 생긴다
    engine.add_after(state, 1, '셋째')
    assert not state.game_over and len(engine.active(state)) == 2
    engine.add_after(state, 1, '넷째')
    assert state.game_over
    engine.retract(state, engine.active(state)[-1].id)
    assert state.game_over                         # 이미 종결된 심문은 되살아나지 않는다


# ---------- retract / revise / refuted / view / successor ----------

def test_revise_keeps_history_marks_revised_and_keeps_one_fact():
    state = fresh()
    new_id = engine.add_after(state, 1, '복도는 환했습니다.')
    assert engine.revise(state, new_id, '복도가 어두웠던 것 같습니다. 그리고 조용했습니다.')
    s = engine.get(state, new_id)
    assert s.text == '복도가 어두웠던 것 같습니다.' and s.status == 'revised' and s.history == ['복도는 환했습니다.']
    assert not engine.revise(state, new_id, s.text)  # 같은 문장으로는 수정되지 않는다
    assert not engine.revise(state, new_id, '  ')


def test_retract_removes_statement_but_never_the_last_one():
    state = fresh()
    new_id = engine.add_after(state, 1, '추가')
    assert engine.retract(state, new_id) and ids(state) == [1]
    assert not engine.retract(state, 1)  # 마지막 증언은 철회할 수 없다


def test_record_refuted_has_no_duplicates():
    state = fresh()
    engine.record_refuted(state, '주장')
    engine.record_refuted(state, '주장')
    assert state.refuted == ['주장']


def test_successor_moves_to_the_next_statement_after_a_retraction():
    state = fresh()
    a = engine.add_after(state, 1, 'a')
    b = engine.add_after(state, a, 'b')            # [1, a, b]
    assert engine.successor(state, a) == a
    engine.retract(state, a)
    assert engine.successor(state, a) == b          # 다음 증언
    engine.retract(state, b)
    assert engine.successor(state, b) == 1          # 다음이 없으면 앞 증언
    assert engine.successor(state, 999) == 1


def test_set_view_validates_input():
    state = fresh()
    before = list(state.view)
    for bad in (None, 'text', [], ['  '], [1, 2]):
        assert not engine.set_view(state, bad) and state.view == before
    assert engine.set_view(state, ['a', '', 'b'] + ['c'] * (engine.MAX_VIEW + 10))   # 상한보다 길면 잘린다
    assert state.view[:2] == [{'text': 'a', 'stance': '애매'}, {'text': 'b', 'stance': '애매'}]  # AI가 준 문자열은 단정하지 않는 쪽
    assert len(state.view) == engine.MAX_VIEW


def test_set_view_keeps_stances_and_softens_unknown_ones():
    state = fresh()
    engine.set_view(state, [{'text': 'a', 'stance': '확신'}, {'text': 'b', 'stance': '숨김'}, {'text': 'c', 'stance': '이상함'}])
    assert [i['stance'] for i in state.view] == ['확신', '숨김', '애매']


# ---------- 질문 흐름 (AI는 monkeypatch) ----------

def stub_reply(monkeypatch, *answers):
    queue = list(answers)
    calls = []

    def fake(case, state, question, base, note=''):
        calls.append(note)
        return queue.pop(0) if queue else None

    monkeypatch.setattr(ai, 'witness_reply', fake)
    return calls


def test_ask_question_inserts_the_answer_after_the_questioned_statement(monkeypatch):
    state = fresh()
    stub_reply(monkeypatch, '복도에서 사람을 봤습니다.')
    new_id = flow.ask_question(CASE, state, 1, '무엇을 보셨나요?')
    assert ids(state) == [1, new_id]
    assert state.log[-2:] == [('변호인', '무엇을 보셨나요?'), ('증인', '복도에서 사람을 봤습니다.')]
    assert state.questions == 1


def test_a_two_sentence_answer_becomes_one_statement_with_one_fact(monkeypatch):
    state = fresh()
    stub_reply(monkeypatch, '9시쯤이었습니다. 시계는 확인하지 못했습니다.')
    new_id = flow.ask_question(CASE, state, 1, '몇 시였나요?')
    assert engine.get(state, new_id).text == '9시쯤이었습니다.'
    assert state.log[-1] == ('증인', '9시쯤이었습니다.')      # 증인 대화 기록에도 저장된 문장만 남긴다


def test_every_question_adds_a_statement_that_stays(monkeypatch):
    state = fresh()
    stub_reply(monkeypatch, '첫째', '둘째', '셋째')
    first = flow.ask_question(CASE, state, 1, 'q1')
    second = flow.ask_question(CASE, state, first, 'q2')
    third = flow.ask_question(CASE, state, 1, 'q3')
    assert ids(state) == [1, third, first, second]  # 질문한 증언 바로 뒤에 들어간다


def test_ask_question_reports_ai_failure_without_adding(monkeypatch):
    state = fresh()
    stub_reply(monkeypatch)  # 항상 None
    monkeypatch.setattr(ai, 'last_error', '[witness_reply] boom')
    with pytest.raises(flow.AskError, match='boom'):
        flow.ask_question(CASE, state, 1, '질문')
    assert ids(state) == [1] and state.log == [] and state.questions == 0


def test_ask_question_retries_when_answer_repeats_a_refuted_claim(monkeypatch):
    state = fresh()
    state.refuted.append('복도는 환하게 밝혀져 있었습니다.')
    calls = stub_reply(monkeypatch, '복도는 환하게 밝혀져 있었습니다.', '복도가 어두웠을 수도 있겠습니다.')
    new_id = flow.ask_question(CASE, state, 1, '조명은요?')
    assert texts(state)[-1] == '복도가 어두웠을 수도 있겠습니다.' and new_id in ids(state)
    assert calls[0] == '' and calls[1] == flow.REPEAT_NOTE  # 두 번째 시도에는 되풀이하지 말라는 주의가 붙는다


def test_ask_question_gives_up_when_witness_keeps_repeating(monkeypatch):
    state = fresh()
    state.refuted.append('복도는 환하게 밝혀져 있었습니다.')
    stub_reply(monkeypatch, *['복도는 환하게 밝혀져 있었습니다.'] * (flow.MAX_RETRIES + 1))
    with pytest.raises(flow.AskError, match='피했습니다'):
        flow.ask_question(CASE, state, 1, '조명은요?')
    assert ids(state) == [1] and state.log == []


def test_reaching_the_limit_ends_the_questioning_with_game_over(monkeypatch):
    state = fresh(limit=3)
    stub_reply(monkeypatch, '둘째', '셋째')
    flow.ask_question(CASE, state, 1, 'q1')
    assert not state.game_over
    flow.ask_question(CASE, state, 1, 'q2')
    assert state.game_over and not state.solved and len(engine.active(state)) == 3


def test_no_more_questions_after_game_over_and_the_ai_is_not_called(monkeypatch):
    state = fresh(limit=2)
    stub_reply(monkeypatch, '둘째')
    flow.ask_question(CASE, state, 1, 'q1')
    assert state.game_over
    monkeypatch.setattr(ai, 'witness_reply', lambda *a, **k: pytest.fail('심문이 끝났으면 AI를 부르면 안 된다'))
    with pytest.raises(flow.AskError, match='이미 끝났습니다'):
        flow.ask_question(CASE, state, 1, 'q2')


def test_no_more_questions_after_the_game_is_solved(monkeypatch):
    state = fresh()
    state.solved = True
    monkeypatch.setattr(ai, 'witness_reply', lambda *a, **k: pytest.fail('클리어 뒤에는 AI를 부르면 안 된다'))
    with pytest.raises(flow.AskError, match='이미 끝났습니다'):
        flow.ask_question(CASE, state, 1, 'q')


# ---------- 증거 + 대사 (판정 AI와 증인 AI는 monkeypatch) ----------

# 증언 번호를 말하지 않는 대사(변호인은 번호를 입력하지 않는다).
CLAIM = '증인은 9시에 복도에서 사람을 봤다고 했고 그때 조명이 켜져 있었다고 했는데 정전 기록은 그 시각 정전이었다고 합니다. 모순입니다.'


def verdict(contradicts=True, grounded=True, cited=(2, 3), confidence=1.0, reason='모순', unsupported='',
            supported=None, conflicts=None):
    """판정 AI의 결과. 기본은 ②증거가 뒷받침하고 ③모순이 성립하는 성공 판정이다."""
    supported = contradicts if supported is None else supported
    conflicts = contradicts if conflicts is None else conflicts
    return {
        'grounded': grounded, 'unsupported': unsupported, 'cited': list(cited), 'supported': supported,
        'conflicts': conflicts, 'contradicts': supported and conflicts, 'confidence': confidence, 'reason': reason,
    }


def setup_two_statements(monkeypatch, judge=None):
    """화면 번호: 1=자기소개, 2='9시에 봤다'(first), 3='조명이 켜져 있었다'(second)."""
    state = fresh()
    stub_reply(monkeypatch, '저는 어젯밤 9시에 복도에서 사람을 봤습니다.', '그때 복도 조명은 켜져 있었습니다.')
    first = flow.ask_question(CASE, state, 1, 'q1')
    second = flow.ask_question(CASE, state, first, 'q2')
    monkeypatch.setattr(ai, 'judge_argument', lambda case, statements, claim, eid: judge or verdict())
    return state, first, second


def stub_react(monkeypatch, *reactions):
    queue = list(reactions)
    seen = []

    def fake(case, state, cited, claim, evidence_id, note=''):
        seen.append({'cited': [s.id for s in cited], 'claim': claim, 'evidence': evidence_id, 'note': note})
        return queue.pop(0) if queue else None

    monkeypatch.setattr(ai, 'witness_react', fake)
    return seen


def react(action='retract', target_id=None, speech='', text=None, view=None):
    return {'action': action, 'target_id': target_id, 'speech': speech, 'text': text, 'view': view}


def test_empty_claim_is_rejected_without_calling_the_judge(monkeypatch):
    state, *_ = setup_two_statements(monkeypatch)
    monkeypatch.setattr(ai, 'judge_argument', lambda *a: pytest.fail('판정 AI를 부르면 안 된다'))
    kind, message = flow.present_evidence(CASE, state, '   ', 'blackout')
    assert kind == 'info' and '대사' in message and not state.solved


def test_claim_built_on_made_up_statements_is_rejected_and_changes_nothing(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch, verdict(grounded=False))
    before = texts(state)
    kind, message = flow.present_evidence(CASE, state, '증인이 아까 밤 10시라고 했잖습니까', 'blackout')
    assert kind == 'warning' and '말하지 않은' in message
    assert texts(state) == before and state.refuted == [] and not state.solved


def test_claim_that_cites_no_statement_asks_the_player_to_be_specific(monkeypatch):
    state, *_ = setup_two_statements(monkeypatch, verdict(cited=()))
    kind, message = flow.present_evidence(CASE, state, '이건 모순입니다', 'blackout')
    assert kind == 'warning' and '어떤 증언' in message and state.refuted == []


def test_evidence_that_does_not_support_the_contradiction_changes_nothing(monkeypatch):
    state, *_ = setup_two_statements(monkeypatch, verdict(contradicts=False))
    stub_react(monkeypatch)
    kind, _ = flow.present_evidence(CASE, state, CLAIM, 'access')
    assert kind == 'warning' and state.refuted == [] and not state.solved


# ---------- 단계별 확인: ①증언에 있는가 ②증거가 뒷받침 ③모순 ----------

def run_stage(monkeypatch, judged):
    state, *_ = setup_two_statements(monkeypatch, judged)
    before = (texts(state), list(state.refuted), list(state.view), state.solved)
    seen = stub_react(monkeypatch, react('retract', target_id=3))
    kind, message = flow.present_evidence(CASE, state, CLAIM, 'blackout')
    assert (texts(state), state.refuted, state.view, state.solved) == before  # 실패하면 상태는 그대로
    assert seen == []                                                          # 증인은 반응하지 않는다
    return kind, message


def test_each_failed_stage_gives_its_own_message_without_changing_anything(monkeypatch):
    stage1 = run_stage(monkeypatch, verdict(grounded=False, unsupported='피고인이 범인이라고'))
    stage1b = run_stage(monkeypatch, verdict(cited=()))
    stage2 = run_stage(monkeypatch, verdict(supported=False, conflicts=True))
    stage3 = run_stage(monkeypatch, verdict(supported=True, conflicts=False))
    stage4 = run_stage(monkeypatch, verdict(confidence=0.2))
    messages = [stage1[1], stage1b[1], stage2[1], stage3[1], stage4[1]]
    assert len(set(messages)) == 5                                   # 단계마다 다른 안내
    assert '"피고인이 범인이라고"은(는) 증인이 말한 적이 없습니다' in stage1[1]
    assert '어떤 증언을 가리키는지' in stage1b[1]
    assert '뒷받침하지 않습니다' in stage2[1]
    assert '모순되지 않습니다' in stage3[1]
    assert stage4[0] == 'info' and '불분명' in stage4[1]
    assert all(kind in ('warning', 'info') for kind, _ in (stage1, stage1b, stage2, stage3, stage4))


def test_failure_messages_never_include_the_judge_reason(monkeypatch):
    secret = '진실: 실제 범행은 10시였다'
    for judged in (verdict(grounded=False, reason=secret), verdict(cited=(), reason=secret),
                   verdict(supported=False, reason=secret), verdict(conflicts=False, reason=secret),
                   verdict(confidence=0.1, reason=secret)):
        assert '10시' not in run_stage(monkeypatch, judged)[1]


def test_unsupported_side_phrase_is_ignored_but_reported_on_success(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch, verdict(unsupported='피고인을 마주쳤다'))
    stub_react(monkeypatch, react('retract', target_id=second, speech='인정합니다.'))
    kind, message = flow.present_evidence(CASE, state, CLAIM, 'access')
    assert kind == 'success' and second not in ids(state)            # 곁가지가 있어도 핵심 전제가 있으면 성공
    assert '※ 대사 중 "피고인을 마주쳤다"은(는) 증인이 말한 적이 없어 근거에서 제외했습니다.' in message


def test_unsupported_side_phrase_is_also_reported_when_the_argument_fails(monkeypatch):
    state, *_ = setup_two_statements(monkeypatch, verdict(conflicts=False, unsupported='피고인을 마주쳤다'))
    kind, message = flow.present_evidence(CASE, state, CLAIM, 'blackout')
    assert kind == 'warning' and '모순되지 않습니다' in message and '근거에서 제외했습니다' in message


def test_no_note_when_nothing_is_unsupported(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch)
    stub_react(monkeypatch, react('retract', target_id=second))
    assert '근거에서 제외' not in flow.present_evidence(CASE, state, CLAIM, 'access')[1]


def test_player_never_needs_to_type_a_statement_number(monkeypatch):
    """번호 없는 대사만으로 성공하고, 어떤 메시지에도 번호 인용을 요구하는 문구가 없다."""
    assert not any(ch.isdigit() and ch != '9' for ch in CLAIM)
    state, first, second = setup_two_statements(monkeypatch)
    stub_react(monkeypatch, react('retract', target_id=second, speech='인정합니다.'))
    kind, message = flow.present_evidence(CASE, state, CLAIM, 'blackout')
    assert kind == 'success' and state.solved
    source = open(flow.__file__, encoding='utf-8').read()
    assert '번 증언' not in source and '번호로 인용' not in source


def test_a_claim_that_happens_to_contain_a_number_is_not_an_error(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch)
    stub_react(monkeypatch, react('retract', target_id=second))
    kind, _ = flow.present_evidence(CASE, state, '2번 증언과 3번 증언은 정전 기록과 모순입니다.', 'access')
    assert kind == 'success'


def test_the_original_player_claim_succeeds_when_only_a_side_phrase_is_unsupported(monkeypatch):
    """실제 플레이 사례: '피고인'은 증인이 쓰지 않은 말이지만, 핵심 전제(9시쯤, 조명 켜짐)는 증언에 있다."""
    state = fresh()
    first = engine.add_after(state, 1, '어젯밤 9시쯤 3층 복도에서 선글라스와 마스크로 얼굴을 가린 사람이 저를 지나쳐 갔습니다.')
    engine.add_after(state, first, '그 사람을 마주쳤을 당시 3층 복도 조명은 분명히 켜져 있었습니다.')
    claim = ('증인은 2번 증언에서 9시에 피고인을 마주쳤고 3번 증언을 통해 이 때 조명이 켜져있다 증언했습니다. '
             '하지만 그당시 3층은 정전이었습니다')
    monkeypatch.setattr(ai, 'judge_argument', lambda *a: verdict(unsupported='피고인을 마주쳤고', cited=(2, 3)))
    stub_react(monkeypatch, react('revise', target_id=3, speech='제가 시간을 착각했나 봅니다.',
                                  text='그때 복도 조명이 켜져 있었는지 확실하지 않습니다.'))
    kind, message = flow.present_evidence(CASE, state, claim, 'blackout')
    assert kind == 'success' and state.solved and '결정적' in message
    assert '"피고인을 마주쳤고"은(는) 증인이 말한 적이 없어 근거에서 제외했습니다' in message


def test_low_confidence_is_inconclusive(monkeypatch):
    state, *_ = setup_two_statements(monkeypatch, verdict(confidence=0.2))
    kind, _ = flow.present_evidence(CASE, state, CLAIM, 'blackout')
    assert kind == 'info' and state.refuted == [] and not state.solved


def test_success_lets_the_witness_retract_one_of_the_cited_statements(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch)
    new_view = [{'text': '복도는 어두웠다.', 'stance': '애매'}]
    seen = stub_react(monkeypatch, react('retract', target_id=second, speech='제 기억이 틀렸군요.', view=new_view))
    kind, message = flow.present_evidence(CASE, state, CLAIM, 'access')  # 결정적 증거가 아니면 클리어 아님
    assert kind == 'success' and second not in ids(state) and first in ids(state)
    assert state.refuted == ['그때 복도 조명은 켜져 있었습니다.'] and state.view == new_view
    assert seen[0]['cited'] == [first, second] and seen[0]['claim'] == CLAIM  # 증인에게는 인용된 증언과 대사가 간다
    # 번호가 아니라 인용된 증언의 문장으로 무엇을 짚었는지 보여 준다.
    assert '"저는 어젯밤 9시에 복도에서 사람을 봤습니다." / "그때 복도 조명은 켜져 있었습니다."' in message
    assert '번 증언' not in message and '제 기억이 틀렸군요' in message and not state.solved


def test_witness_may_revise_the_other_cited_statement_instead(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch)
    stub_react(monkeypatch, react('revise', target_id=first, speech='음…', text='시계로는 9시쯤이었던 것 같습니다.'))
    flow.present_evidence(CASE, state, CLAIM, 'access')
    s = engine.get(state, first)
    assert s.text == '시계로는 9시쯤이었던 것 같습니다.' and s.status == 'revised'
    assert s.history == ['저는 어젯밤 9시에 복도에서 사람을 봤습니다.'] and second in ids(state)
    assert state.refuted == ['저는 어젯밤 9시에 복도에서 사람을 봤습니다.']


def test_revised_statement_is_kept_to_one_sentence(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch)
    stub_react(monkeypatch, react('revise', target_id=first, text='시계로는 9시쯤이었습니다. 하지만 확실치 않습니다.'))
    flow.present_evidence(CASE, state, CLAIM, 'access')
    assert engine.get(state, first).text == '시계로는 9시쯤이었습니다.'


def test_target_outside_the_cited_statements_falls_back_to_the_last_cited(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch)
    stub_react(monkeypatch, react('retract', target_id=999, speech='...'))
    flow.present_evidence(CASE, state, CLAIM, 'access')
    assert second not in ids(state) and first in ids(state)


def test_revision_that_repeats_the_refuted_claim_is_retried_then_retracted(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch)
    repeat = react('revise', target_id=second, speech='아니요, 켜져 있었습니다.', text='그때 복도 조명은 켜져 있었습니다.')
    seen = stub_react(monkeypatch, repeat, repeat, repeat)
    flow.present_evidence(CASE, state, CLAIM, 'access')
    assert second not in ids(state)  # 끝내 되풀이하면 철회로 처리된다
    assert len(seen) == flow.MAX_RETRIES + 1 and seen[1]['note'] == flow.REPEAT_NOTE


def test_revision_is_accepted_after_a_repeat_is_corrected(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch)
    bad = react('revise', target_id=second, text='그때 복도 조명은 켜져 있었습니다.')
    good = react('revise', target_id=second, text='복도에 불빛이 희미했습니다.')
    stub_react(monkeypatch, bad, good)
    flow.present_evidence(CASE, state, CLAIM, 'access')
    assert engine.get(state, second).text == '복도에 불빛이 희미했습니다.'


def test_decisive_evidence_clears_only_when_the_argument_succeeds(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch, verdict(contradicts=False))
    assert flow.present_evidence(CASE, state, CLAIM, 'blackout')[0] == 'warning' and not state.solved
    monkeypatch.setattr(ai, 'judge_argument', lambda *a: verdict())
    stub_react(monkeypatch, react('retract', target_id=second, speech='인정합니다.'))
    kind, message = flow.present_evidence(CASE, state, CLAIM, 'blackout')
    assert kind == 'success' and state.solved and '결정적' in message


# ---------- 클리어 판정은 재판관 AI가 한다 (실패/오프라인일 때만 결정적 증거 규칙) ----------

def stub_clear(monkeypatch, cleared=None, error=False):
    calls = []

    def fake(case, state, evidence_id, claim, pointed, changes):
        calls.append({'evidence': evidence_id, 'claim': claim, 'pointed': pointed, 'changes': changes, 'refuted': list(state.refuted)})
        return None if error else {'cleared': cleared, 'reason': '재판관의 판단'}

    monkeypatch.setattr(ai, 'judge_clear', fake)
    return calls


def test_the_judge_ai_can_refuse_to_clear_even_with_the_decisive_evidence(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch)
    stub_react(monkeypatch, react('retract', target_id=second, speech='인정합니다.'))
    calls = stub_clear(monkeypatch, cleared=False)
    kind, message = flow.present_evidence(CASE, state, CLAIM, 'blackout')   # 결정적 증거여도 재판관이 아니라고 하면 클리어가 아니다
    assert kind == 'success' and not state.solved and '결정적' not in message
    assert len(calls) == 1 and calls[0]['evidence'] == 'blackout'


def test_the_judge_ai_can_clear_with_any_evidence(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch)
    stub_react(monkeypatch, react('retract', target_id=second))
    stub_clear(monkeypatch, cleared=True)
    kind, message = flow.present_evidence(CASE, state, CLAIM, 'access')     # 결정적 증거가 아니어도 AI가 핵심 모순이라 판단하면 클리어
    assert kind == 'success' and state.solved and '결정적' in message


def test_judge_receives_what_was_pointed_what_changed_and_the_refuted_history(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch)
    stub_react(monkeypatch, react('revise', target_id=second, text='조명이 켜져 있었는지는 확실치 않습니다.'))
    calls = stub_clear(monkeypatch, cleared=False)
    flow.present_evidence(CASE, state, CLAIM, 'blackout')
    call = calls[0]
    assert call['claim'] == CLAIM
    assert call['pointed'] == ['저는 어젯밤 9시에 복도에서 사람을 봤습니다.', '그때 복도 조명은 켜져 있었습니다.']   # 고치기 전의 문장
    assert any('조명이 켜져 있었는지는 확실치 않습니다' in line for line in call['changes'])
    assert '그때 복도 조명은 켜져 있었습니다.' in call['refuted']


def test_the_judge_is_asked_only_after_a_successful_refutation(monkeypatch):
    state, *_ = setup_two_statements(monkeypatch, verdict(conflicts=False))
    monkeypatch.setattr(ai, 'judge_clear', lambda *a: pytest.fail('반박에 실패하면 재판관 판정을 부르면 안 된다'))
    assert flow.present_evidence(CASE, state, CLAIM, 'blackout')[0] == 'warning'


def test_when_the_judge_ai_fails_the_old_decisive_evidence_rule_applies(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch)
    stub_react(monkeypatch, react('retract', target_id=second))
    stub_clear(monkeypatch, error=True)
    assert flow.present_evidence(CASE, state, CLAIM, 'access')[0] == 'success' and not state.solved   # 결정적 증거가 아니면 아님
    state, first, second = setup_two_statements(monkeypatch)
    stub_react(monkeypatch, react('retract', target_id=second))
    stub_clear(monkeypatch, error=True)
    flow.present_evidence(CASE, state, CLAIM, 'blackout')
    assert state.solved                                                                                 # 결정적 증거면 클리어


# ---------- 연쇄 반박 ----------

def setup_dependents(monkeypatch):
    """증언: 자기소개, '9시에 봤다'(a), '조명이 켜져 있었다'(b), '그 사람은 마스크를 썼다'(c), '그 사람은 봉투를 들었다'(d)."""
    state = fresh()
    stub_reply(monkeypatch, '저는 어젯밤 9시에 복도에서 사람을 봤습니다.', '그때 복도 조명은 켜져 있었습니다.',
               '그 사람은 마스크를 썼습니다.', '그 사람은 봉투를 들고 있었습니다.')
    a = flow.ask_question(CASE, state, 1, 'q1')
    b = flow.ask_question(CASE, state, a, 'q2')
    c = flow.ask_question(CASE, state, b, 'q3')
    d = flow.ask_question(CASE, state, c, 'q4')
    monkeypatch.setattr(ai, 'judge_argument', lambda *args: verdict(cited=(2, 3)))
    stub_clear(monkeypatch, cleared=False)
    return state, a, b, c, d


def cascade_item(statement_id, action='retract', text=None):
    return {'id': statement_id, 'action': action, 'text': text}


def test_dependent_statements_are_withdrawn_together_with_the_refuted_one(monkeypatch):
    state, a, b, c, d = setup_dependents(monkeypatch)
    stub_react(monkeypatch, {**react('retract', target_id=a, speech='시간이 틀렸군요.'),
                             'cascade': [cascade_item(c), cascade_item(d)]})
    kind, message = flow.present_evidence(CASE, state, CLAIM, 'access')
    assert kind == 'success' and ids(state) == [1, b]                      # a뿐 아니라 그 사람에 대한 증언 c, d도 거둔다
    assert '연쇄로 "그 사람은 마스크를 썼습니다." 증언도 함께 철회되었습니다.' in message
    assert '연쇄로 "그 사람은 봉투를 들고 있었습니다." 증언도 함께 철회되었습니다.' in message
    assert state.refuted == ['저는 어젯밤 9시에 복도에서 사람을 봤습니다.', '그 사람은 마스크를 썼습니다.', '그 사람은 봉투를 들고 있었습니다.']


def test_a_dependent_statement_can_be_revised_instead_of_withdrawn(monkeypatch):
    state, a, b, c, d = setup_dependents(monkeypatch)
    stub_react(monkeypatch, {**react('retract', target_id=a),
                             'cascade': [cascade_item(c, 'revise', '그 사람이 마스크를 썼는지는 확실치 않습니다.')]})
    _, message = flow.present_evidence(CASE, state, CLAIM, 'access')
    assert engine.get(state, c).text == '그 사람이 마스크를 썼는지는 확실치 않습니다.' and engine.get(state, c).status == 'revised'
    assert '"그 사람이 마스크를 썼는지는 확실치 않습니다."(으)로 수정되었습니다' in message
    assert '그 사람은 마스크를 썼습니다.' in state.refuted


def test_a_dependent_revision_that_repeats_a_refuted_claim_becomes_a_withdrawal(monkeypatch):
    state, a, b, c, d = setup_dependents(monkeypatch)
    stub_react(monkeypatch, {**react('retract', target_id=a),
                             'cascade': [cascade_item(c, 'revise', '저는 어젯밤 9시에 복도에서 사람을 봤습니다.')]})
    flow.present_evidence(CASE, state, CLAIM, 'access')
    assert c not in ids(state)


def test_malformed_or_invalid_cascade_items_are_ignored_by_form_checks_only(monkeypatch):
    state, a, b, c, d = setup_dependents(monkeypatch)
    junk = ['x', None, 3, {}, {'id': 'c'}, {'id': True}, {'id': 999, 'action': 'retract'}, {'id': a, 'action': 'retract'},
            {'id': c, 'action': 'retract'}, {'id': c, 'action': 'retract'}]   # 마지막 둘은 중복
    stub_react(monkeypatch, {**react('retract', target_id=a), 'cascade': junk})
    flow.present_evidence(CASE, state, CLAIM, 'access')
    assert ids(state) == [1, b, d]                                          # 유효한 항목(c)만, 한 번만 처리된다


def test_the_last_remaining_statement_is_never_withdrawn_by_a_cascade(monkeypatch):
    state, a, b, c, d = setup_dependents(monkeypatch)
    everything = [cascade_item(s) for s in (1, b, c, d)]
    stub_react(monkeypatch, {**react('retract', target_id=a), 'cascade': everything})
    flow.present_evidence(CASE, state, CLAIM, 'access')
    assert len(ids(state)) == 1                                              # 마지막 하나는 남는다


def test_without_a_cascade_only_the_cited_statement_changes(monkeypatch):
    state, a, b, c, d = setup_dependents(monkeypatch)
    stub_react(monkeypatch, react('retract', target_id=a))
    flow.present_evidence(CASE, state, CLAIM, 'access')
    assert ids(state) == [1, b, c, d] and '연쇄' not in ''.join(state.events)


def test_the_judge_sees_the_cascade_as_part_of_what_changed(monkeypatch):
    state, a, b, c, d = setup_dependents(monkeypatch)
    stub_react(monkeypatch, {**react('retract', target_id=a), 'cascade': [cascade_item(c)]})
    calls = stub_clear(monkeypatch, cleared=False)
    flow.present_evidence(CASE, state, CLAIM, 'access')
    assert any('연쇄로 "그 사람은 마스크를 썼습니다." 증언도 함께 철회' in line for line in calls[0]['changes'])


def test_ai_failure_during_reaction_falls_back_to_retracting_the_last_cited(monkeypatch):
    state, first, second = setup_two_statements(monkeypatch)
    stub_react(monkeypatch)  # 항상 None
    kind, message = flow.present_evidence(CASE, state, CLAIM, 'access')
    assert kind == 'success' and second not in ids(state) and '철회' in message


def test_reaction_inputs_do_not_include_judge_reason(monkeypatch):
    """증인에게 넘어가는 정보에는 판정 AI의 reason(진실이 섞일 수 있음)이 없다."""
    state, *_ = setup_two_statements(monkeypatch, verdict(reason='진실: 정전이라 얼굴을 못 봄'))
    seen = stub_react(monkeypatch, react('retract', target_id=3))
    flow.present_evidence(CASE, state, CLAIM, 'access')
    assert all('진실' not in str(item) for item in seen)


def test_judge_gets_statements_in_screen_order(monkeypatch):
    """증언 번호(1부터)는 화면 순서이고, 판정 AI에게도 같은 순서로 전달된다."""
    state, first, second = setup_two_statements(monkeypatch)
    captured = {}

    def judge(case, statements, claim, eid):
        captured.update(statements=statements, claim=claim, eid=eid)
        return verdict(contradicts=False)

    monkeypatch.setattr(ai, 'judge_argument', judge)
    flow.present_evidence(CASE, state, CLAIM, 'blackout')
    assert captured['statements'] == texts(state) and captured['claim'] == CLAIM and captured['eid'] == 'blackout'


def test_evidence_is_refused_after_game_over_without_calling_the_ai(monkeypatch):
    state, *_ = setup_two_statements(monkeypatch)
    state.game_over = True
    monkeypatch.setattr(ai, 'judge_argument', lambda *a: pytest.fail('심문이 끝났으면 AI를 부르면 안 된다'))
    kind, message = flow.present_evidence(CASE, state, CLAIM, 'blackout')
    assert kind == 'info' and '종결' in message and not state.solved


# ---------- 오프라인 폴백(키 없음) ----------

def test_offline_answers_are_single_sentences():
    for question in ('조명은요?', '몇 시였나요?', '봉투는요?', '안녕하세요'):
        answer = ai._offline_reply(question)
        assert engine.first_sentence(answer) == answer


def test_offline_unrelated_evidence_asks_which_statement_is_meant_not_made_up():
    state = fresh()
    flow.ask_question(CASE, state, 1, '복도 조명이 켜져 있었나요?')
    kind, message = flow.present_evidence(CASE, state, '증인의 말은 이 기록과 모순입니다.', 'access')
    assert kind == 'warning' and '어떤 증언을 가리키는지' in message and '말하지 않은' not in message


def test_offline_light_question_then_blackout_evidence_clears_the_game():
    state = fresh()
    new_id = flow.ask_question(CASE, state, 1, '복도 조명이 켜져 있었나요?')
    assert '밝' in engine.get(state, new_id).text
    claim = '증인은 조명이 밝게 켜져 있었다고 했는데 이 기록과 모순입니다.'   # 번호 없음
    kind, _ = flow.present_evidence(CASE, state, claim, 'access')
    assert kind == 'warning' and not state.solved
    kind, message = flow.present_evidence(CASE, state, claim, 'blackout')
    assert kind == 'success' and state.solved and new_id not in ids(state)
    assert '번 증언' not in message


def test_offline_claim_without_a_number_still_finds_the_statement_by_keyword():
    state = fresh()
    flow.ask_question(CASE, state, 1, '복도 조명이 켜져 있었나요?')
    kind, _ = flow.present_evidence(CASE, state, '불이 밝았다는 건 정전과 모순입니다', 'blackout')
    assert kind == 'success' and state.solved


def test_offline_witness_cannot_repeat_a_refuted_claim():
    state = fresh()
    flow.ask_question(CASE, state, 1, '복도 조명이 켜져 있었나요?')
    flow.present_evidence(CASE, state, '증인은 조명이 밝았다고 했는데 정전 기록과 모순입니다.', 'blackout')
    state.solved = False  # 이어서 질문할 수 있게 클리어 상태만 풀고 반복 방지 규칙을 확인한다
    with pytest.raises(flow.AskError):  # 같은 키워드 답변은 이미 반박된 주장이라 막힌다
        flow.ask_question(CASE, state, 1, '조명은 어땠나요?')


def test_offline_game_over_after_enough_questions():
    state = fresh(limit=4)
    for question in ('안녕하세요', '몇 시였나요?', '봉투는요?'):
        flow.ask_question(CASE, state, 1, question)
    assert state.game_over and not state.solved
    with pytest.raises(flow.AskError):
        flow.ask_question(CASE, state, 1, '더 질문')
