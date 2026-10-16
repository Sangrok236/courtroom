"""사건 데이터가 3층 구조 규칙을 지키는지 검사한다. 작가가 사건을 고치다 진실을 유출하는 실수를 막는다."""
import ai
import engine
import flow
from cases.case01 import CASE

# 증인 AI에게 전달되는 필드(persona, introduction, view, goal.detail)에 들어가면 안 되는 진실 키워드.
# '범인'만으로는 막지 않는다: 경비의 추측("피고인이 범인이다")은 증인의 믿음이지 진실 유출이 아니다. 진짜 범인을 가리키는 말만 막는다.
SECRET_WORDS = [
    '10시', '열 시', '조작', '느리게', '1시간', '한 시간', '제3자', '제 3자', '진범', '실제 범인',
    '8시 30분', '8시 40분', '30분 빠르', '20분 빠르', '빠르게', '수면제', '시계를 조작', '쓰레기통', '맞춰 고장',
]
# 증인이 알아도 되는 것(믿음): ① 벽시계가 9시 10분에 멈춰 고장난 것(조작·오차 없이 결과만), ② 컴퓨터 퇴근 기록에 8시 35분에 나간
# 사람이 있었다는 것(기록이므로 안다. 다만 그 사람이 진범이라는 것은 모른다 → '진범', '실제 범인'은 계속 금지).


def witness_visible_texts():
    witness = CASE['witness']
    return {
        'persona': witness['persona'],
        'introduction': witness['introduction'],
        'goal.detail': witness['goal']['detail'],
        # view 항목은 딕셔너리라 text를 꺼내 검사해야 한다(딕셔너리에 `in`을 쓰면 키만 보게 되어 아무것도 못 잡는다).
        **{f'view[{i}]': item['text'] for i, item in enumerate(witness['view'])},
    }


def test_witness_visible_case_data_does_not_leak_truth():
    texts = witness_visible_texts()
    assert all(isinstance(text, str) and text for text in texts.values())
    for field, text in texts.items():
        for word in SECRET_WORDS:
            assert word not in text, f'{field}에 진실 키워드 {word!r}가 있습니다: {text}'


def test_leak_check_really_catches_a_leak(monkeypatch):
    """유출 검사가 형식이 바뀌어도 여전히 잡아내는지(과거에 딕셔너리를 검사해 무의미하게 통과한 적이 있다)."""
    leaky = {**CASE, 'witness': {**CASE['witness'], 'view': CASE['witness']['view'] + [
        {'text': '실제로는 10시였다.', 'stance': '확신'}]}}
    monkeypatch.setattr(__import__(__name__), 'CASE', leaky)
    assert any('10시' in text for text in witness_visible_texts().values())


def test_witness_prompt_contains_no_secret_words(monkeypatch):
    """실제 프롬프트에서도 확인한다(필드 외에 코드가 덧붙인 문장 포함)."""
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    prompts = []
    monkeypatch.setattr(ai, '_call_json', lambda instructions, *a, **k: prompts.append(instructions) or {'answer': 'a'})
    state = engine.new_state(CASE)
    ai.witness_reply(CASE, state, '질문', state.statements[0])
    for word in SECRET_WORDS:
        assert word not in prompts[0], word


def test_clear_condition_exists_and_stays_with_the_judge(monkeypatch):
    """클리어 조건은 재판관 AI만 안다. 증인 AI의 어떤 프롬프트에도 들어가면 안 된다."""
    condition = CASE['clear_condition']
    assert condition.strip() and CASE['decisive']['evidence'] in CASE['evidence']   # 폴백용 결정적 증거도 유효해야 한다
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    prompts = []
    monkeypatch.setattr(ai, '_call_json', lambda instructions, *a, **k: prompts.append(instructions) or
                        {'answer': 'a', 'action': 'retract', 'target_id': 1, 'speech': '', 'text': None, 'view': None, 'cascade': []})
    state = engine.new_state(CASE)
    ai.witness_reply(CASE, state, '질문', state.statements[0])
    ai.witness_react(CASE, state, [state.statements[0]], '지적', 'blackout')
    assert prompts and all(condition not in prompt for prompt in prompts)
    assert all(fact not in prompt for prompt in prompts for fact in CASE['truth'])


def test_no_view_item_is_silently_dropped():
    """view가 상한보다 길면 뒤쪽 항목이 조용히 사라진다(과거에 13개 중 마지막 항목이 잘린 적이 있다)."""
    view = CASE['witness']['view']
    state = engine.new_state(CASE)
    assert len(engine.normalize_view(view)) == len(view) == len(state.view)
    assert state.view[-1]['text'] == view[-1]['text']


def _minutes(hour: int, minute: int) -> int:
    return hour * 60 + minute


def test_story_times_are_consistent():
    """범행 시각 + 시계 오차 = 경비가 믿는 시각(9시), 그리고 범행은 정전이 시작되기 전이다."""
    import re

    truth = ' '.join(CASE['truth'])
    crime = re.search(r'실제 범행은 밤 (\d+)시 (\d+)분', truth)
    offset = re.search(r'(\d+)분 빠르게', truth)
    blackout = re.search(r'(\d+)시 (\d+)분부터', CASE['evidence']['blackout']['description'])
    culprit_tag = re.search(r'진범\(제3자\)도 오후 (\d+)시 (\d+)분에', truth)
    defendant_tag = re.search(r'피고인은 오후 (\d+)시 (\d+)분에 1층 로비', truth)
    assert crime and offset and blackout and culprit_tag and defendant_tag, '시각 문장의 형식이 바뀌었다'
    crime_at = _minutes(int(crime.group(1)), int(crime.group(2)))
    blackout_at = _minutes(int(blackout.group(1)), int(blackout.group(2)))
    culprit_out = _minutes(int(culprit_tag.group(1)), int(culprit_tag.group(2)))
    defendant_out = _minutes(int(defendant_tag.group(1)), int(defendant_tag.group(2)))
    believed = crime_at + int(offset.group(1))
    assert believed == _minutes(9, 0), f'범행 {crime.group(0)} + 시계 {offset.group(1)}분 = {believed // 60}시 {believed % 60}분(9시여야 한다)'
    assert crime_at < blackout_at                          # 범행은 정전 전
    assert crime_at < culprit_out < blackout_at            # 범인은 범행 뒤, 정전 전에 퇴근했다
    assert defendant_out > blackout_at                     # 피고인의 기록은 정전 시간대(1층은 정상이라 기록됨)
    assert '8시 50분 이전에' in truth and '1층 경비실로 돌아왔다' in truth       # 경비는 정전 전에 3층을 떠났다
    assert '우연한 설비 고장' in truth                      # 정전은 우연


def test_the_briefing_does_not_contradict_the_culprits_tag_record():
    """검사의 '9시 전후 퇴근 기록은 피고인뿐'은 범인의 8시 35분 기록과 충돌하지 않아야 한다(범위 밖)."""
    assert '9시 전후' in CASE['briefing'] and '그 무렵' not in CASE['briefing']


def test_truth_list_has_no_duplicates():
    assert len(set(CASE['truth'])) == len(CASE['truth'])


def test_view_items_are_valid():
    view = CASE['witness']['view']
    assert 0 < len(view) <= engine.MAX_VIEW
    for item in view:
        assert set(item) == {'text', 'stance'}, item
        assert item['stance'] in engine.STANCES, item
        assert item['text'].strip() and len(item['text']) <= engine.MAX_TEXT, item
    texts = [item['text'] for item in view]
    assert len(set(texts)) == len(texts)
    assert engine.normalize_view(view) == view  # 정규화해도 달라지지 않는다


def view_item(fragment):
    matches = [item for item in CASE['witness']['view'] if fragment in item['text']]
    assert len(matches) == 1, f'{fragment!r}를 담은 view 항목이 {len(matches)}개'
    return matches[0]


def test_the_guard_saw_both_tag_records_but_thinks_the_other_person_left_30_minutes_before_his_patrol():
    other = view_item('8시 35분에 퇴근한 사람')
    assert '30분쯤 전' in other['text'] and other['stance'] == '미언급'          # 묻지 않으면 먼저 말하지 않는다
    assert '진범' not in other['text'] and '범인' not in other['text']           # 그 사람이 범인이라는 것은 모른다
    defendant = view_item('피고인은 오후 9시 12분')
    assert defendant['stance'] == '확신'
    truth = ' '.join(CASE['truth'])
    assert '피고인의 9시 12분 기록과 진범의 8시 35분 기록을 모두 보았다' in truth      # 판정 AI는 사실을 안다


def test_the_guard_thinks_the_wall_clock_broke_while_he_was_upstairs_and_does_not_volunteer_it():
    clock = view_item('9시 10분에 멈춰 고장나')
    assert '올라가기 전에' in clock['text'] and '순찰하는 동안 멈춘 것' in clock['text']
    assert clock['stance'] == '미언급'
    for secret in ('조작', '빠르게', '30분 빠르'):                                # 고장난 줄만 알고 조작된 줄은 모른다
        assert secret not in clock['text']


def test_the_witness_data_uses_the_non_volunteering_stance_for_both_clues():
    quiet = [item['text'] for item in CASE['witness']['view'] if item['stance'] == '미언급']
    assert len(quiet) == 2


def test_view_has_a_hidden_and_a_vague_belief_to_exercise_the_stances():
    stances = {item['stance'] for item in CASE['witness']['view']}
    assert {'확신', '숨김'} <= stances


def test_statement_limit_and_game_over_line_are_set():
    limit = CASE['statement_limit']
    assert isinstance(limit, int) and not isinstance(limit, bool) and limit >= 3
    assert CASE['game_over'].strip() and '심문' in CASE['game_over'] and '무의미' in CASE['game_over']
    assert engine.new_state(CASE).limit == limit


def test_game_over_line_is_not_visible_to_the_witness_ai(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    prompts = []
    monkeypatch.setattr(ai, '_call_json', lambda instructions, *a, **k: prompts.append(instructions) or {'answer': 'a'})
    state = engine.new_state(CASE)
    ai.witness_reply(CASE, state, '질문', state.statements[0])
    assert CASE['game_over'] not in prompts[0] and CASE['ending'] not in prompts[0]


def test_truth_and_evidence_times_agree():
    truth = ' '.join(CASE['truth'])
    assert '9시 12분' in truth and '1층' in truth
    assert '9시 12분' in CASE['evidence']['access']['description'] and '1층' in CASE['evidence']['access']['description']
    blackout = CASE['evidence']['blackout']['description']
    assert '8시 50분' in blackout and '9시 20분' in blackout and '3층' in blackout
    assert '8시 50분' in truth and '9시 20분' in truth


def test_witness_believes_the_wall_clock_said_nine():
    view = ' '.join(item['text'] for item in CASE['witness']['view'])
    assert '9시' in view and '조명' in view and '켜져' in view


def test_decisive_evidence_exists_and_ending_mentions_the_contradiction():
    assert CASE['decisive']['evidence'] in CASE['evidence']
    assert '정전' in CASE['ending'] and '휴정' in CASE['ending']
    assert '느리게' not in CASE['ending']  # 시계 조작의 진상은 이번 판에서 밝히지 않는다(경비실을 찾았다는 떡밥만)


def test_judge_prompt_tells_the_judge_to_follow_the_time_the_witness_stated(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    prompts = []
    monkeypatch.setattr(ai, '_call_json', lambda instructions, *a, **k: prompts.append(instructions) or
                        {'grounded': True, 'cited': [1], 'contradicts': True, 'confidence': 0.9, 'reason': 'r'})
    ai._contradiction_cache.clear()
    ai.judge_argument(CASE, ['9시쯤 복도 조명이 켜져 있었습니다.'], '증인의 말은 정전 기록과 모순입니다.', 'blackout')
    assert '증언이 말하는 시각' in prompts[0]
    assert all(fact in prompts[0] for fact in CASE['truth'])


def test_story_example_two_statements_combined_by_the_player_clear_the_game_offline():
    """사용자의 예: '9시에 봤다'와 '불이 켜져 있었다'를 엮어 정전 기록과 모순이라고 지적(증언 번호는 말하지 않는다)."""
    state = engine.new_state(CASE)
    a = engine.add_after(state, 1, '저는 어젯밤 9시에 복도에서 사람을 봤습니다.')
    engine.add_after(state, a, '그때 복도 조명은 환하게 켜져 있었습니다.')
    claim = '증인은 9시에 복도에서 사람을 봤고 그때 불이 켜져 있었다고 했는데, 정전 기록은 그 시각 정전이었다고 합니다. 모순입니다.'
    kind, _ = flow.present_evidence(CASE, state, claim, 'access')
    assert kind == 'warning' and not state.solved
    kind, message = flow.present_evidence(CASE, state, claim, 'blackout')
    assert kind == 'success' and state.solved and '결정적' in message
