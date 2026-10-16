import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import ai
import engine
from cases.case01 import CASE

ENV_NAMES = ('GEMINI_API_KEY', 'GOOGLE_API_KEY', 'OPENAI_API_KEY', 'AI_PROVIDER',
             'AI_MODEL', 'GEMINI_MODEL', 'OPENAI_MODEL')


def clear_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def fake_gemini(monkeypatch, generate_content):
    import google.genai as genai

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.models = types.SimpleNamespace(generate_content=generate_content)

    monkeypatch.setattr(genai, 'Client', FakeClient)


# ---------- 스키마 변환 ----------

def test_to_gemini_schema_strips_unsupported_keywords():
    schema = {
        'type': 'object', 'additionalProperties': False, 'required': ['id'],
        'properties': {'id': {'type': ['integer', 'null']}, 'items': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False, 'properties': {'t': {'type': 'string'}}}}},
    }
    out = ai._to_gemini_schema(schema)
    assert 'additionalProperties' not in out
    assert out['properties']['id'] == {'type': 'integer', 'nullable': True}
    assert 'additionalProperties' not in out['properties']['items']['items']


def test_to_gemini_schema_drops_integer_enum_but_keeps_string_enum():
    out = ai._to_gemini_schema({'type': 'object', 'properties': {
        'n': {'type': 'integer', 'enum': [0, 1]}, 's': {'type': 'string', 'enum': ['a', 'b']}}})
    assert 'enum' not in out['properties']['n']
    assert out['properties']['s']['enum'] == ['a', 'b']


# ---------- 공급자/모델/키 ----------

def test_has_api_key_accepts_gemini(monkeypatch):
    clear_env(monkeypatch)
    assert not ai.has_api_key()
    monkeypatch.setenv('GEMINI_API_KEY', 'x')
    assert ai.has_api_key()


def test_provider_and_model_resolution(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv('GEMINI_API_KEY', 'g')
    monkeypatch.setenv('OPENAI_API_KEY', 'o')
    assert ai._provider() == 'gemini'
    monkeypatch.setenv('AI_PROVIDER', 'OpenAI')
    assert ai._provider() == 'openai'
    monkeypatch.setenv('AI_PROVIDER', 'Gemini')
    assert ai._provider() == 'gemini'
    assert ai._model('gemini') == 'gemini-3.8-flash'
    monkeypatch.setenv('AI_MODEL', 'custom')
    assert ai._model('gemini') == 'custom'
    monkeypatch.setenv('GEMINI_MODEL', 'specific')
    assert ai._model('gemini') == 'specific'


def test_describe_error_masks_keys(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv('GEMINI_API_KEY', 'SECRET-KEY-123')
    text = ai._describe_error(ValueError('bad key SECRET-KEY-123 rejected'))
    assert 'SECRET-KEY-123' not in text and '***' in text and text.startswith('ValueError')


# ---------- Gemini 호출 ----------

def test_gemini_client_uses_developer_api_for_all_key_formats(monkeypatch):
    clear_env(monkeypatch)
    created = []
    import google.genai as genai

    class FakeClient:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.models = types.SimpleNamespace(generate_content=lambda **kw: types.SimpleNamespace(text='{"ok": true}'))

    monkeypatch.setattr(genai, 'Client', FakeClient)
    monkeypatch.setenv('GEMINI_API_KEY', 'AQ.fake')
    assert ai._call_gemini('i', 'u', {'type': 'object'}, 10, 0) == {'ok': True}
    monkeypatch.setenv('GEMINI_API_KEY', 'AIzaFake')
    ai._call_gemini('i', 'u', {'type': 'object'}, 10, 0)
    assert created == [{'api_key': 'AQ.fake'}, {'api_key': 'AIzaFake'}]


def test_gemini_3_uses_thinking_level_and_2_uses_budget(monkeypatch):
    clear_env(monkeypatch)
    seen = []
    fake_gemini(monkeypatch, lambda **kw: seen.append(kw['config'].thinking_config) or types.SimpleNamespace(text='{"ok": 1}'))
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    monkeypatch.setenv('AI_MODEL', 'gemini-3.8-flash')
    ai._call_gemini('i', 'u', {'type': 'object'}, 10, 0)
    monkeypatch.setenv('AI_MODEL', 'gemini-2.5-flash')
    ai._call_gemini('i', 'u', {'type': 'object'}, 10, 0)
    assert seen[0].thinking_level is not None and seen[0].thinking_budget is None
    assert seen[1].thinking_budget == 0 and seen[1].thinking_level is None


def test_gemini_retries_without_thinking_config_on_400(monkeypatch):
    clear_env(monkeypatch)
    seen = []

    class ThinkingRejected(Exception):
        code = 400

    def generate_content(**kw):
        seen.append(kw['config'].thinking_config)
        if kw['config'].thinking_config is not None:
            raise ThinkingRejected('thinking_budget is not supported for this model')
        return types.SimpleNamespace(text='{"ok": true}')

    fake_gemini(monkeypatch, generate_content)
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    assert ai._call_gemini('i', 'u', {'type': 'object'}, 10, 0) == {'ok': True}
    assert seen[0] is not None and seen[1] is None


def test_gemini_does_not_retry_other_errors(monkeypatch):
    clear_env(monkeypatch)

    class NotFound(Exception):
        code = 404

    def generate_content(**kw):
        raise NotFound('model gone')

    fake_gemini(monkeypatch, generate_content)
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    with pytest.raises(NotFound):
        ai._call_gemini('i', 'u', {'type': 'object'}, 10, 0)


def test_empty_gemini_response_reports_finish_reason():
    reason = types.SimpleNamespace(name='MAX_TOKENS')
    response = types.SimpleNamespace(text='', candidates=[types.SimpleNamespace(finish_reason=reason)])
    with pytest.raises(RuntimeError, match='MAX_TOKENS'):
        ai._gemini_text(response)
    with pytest.raises(RuntimeError, match='빈 응답'):
        ai._gemini_text(types.SimpleNamespace(text=None))


def test_failed_call_is_recorded_with_call_name_and_witness_returns_none(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv('GEMINI_API_KEY', 'SECRET')
    monkeypatch.setattr(ai, '_call_gemini', lambda *a, **k: (_ for _ in ()).throw(ValueError('boom SECRET')))
    monkeypatch.setattr(ai, 'last_error', '')
    state = engine.new_state(CASE)
    assert ai.witness_reply(CASE, state, '질문', state.statements[0]) is None
    assert ai.last_error.startswith('[witness_reply] ValueError') and 'SECRET' not in ai.last_error


# ---------- 3층 구조: AI별로 보는 범위 ----------

def capture_calls(monkeypatch, response):
    """_call_json을 가로채 프롬프트를 기록한다(실제 API 호출 없음)."""
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    calls = []

    def fake(instructions, user_input, name, schema, max_tokens, temperature):
        calls.append({'instructions': instructions, 'input': user_input, 'name': name, 'schema': schema})
        return response

    monkeypatch.setattr(ai, '_call_json', fake)
    return calls


def test_witness_prompt_has_view_goal_refuted_and_never_the_truth(monkeypatch):
    calls = capture_calls(monkeypatch, {'answer': '네.'})
    state = engine.new_state(CASE)
    engine.record_refuted(state, '복도는 환했습니다.')
    ai.witness_reply(CASE, state, '질문', state.statements[0])
    prompt = calls[0]['instructions']
    for item in CASE['witness']['view']:
        assert f'[{item["stance"]}] {item["text"]}' in prompt   # 2층: 증인이 아는 진실을 태도와 함께
    assert CASE['witness']['goal']['detail'] in prompt          # 3층: 목적
    assert ai.GOAL_GUIDE['honest'] in prompt
    assert '복도는 환했습니다.' in prompt and '다시 하지 마세요' in prompt  # 반박된 주장은 반복 금지
    for fact in CASE['truth']:                                  # 1층: 진실은 절대 넣지 않는다
        assert fact not in prompt


def test_view_prompt_explains_every_stance_in_use_and_puts_stance_above_goal(monkeypatch):
    calls = capture_calls(monkeypatch, {'answer': '네.'})
    state = engine.new_state(CASE)
    state.view = engine.normalize_view([
        {'text': 'a', 'stance': '확신'}, {'text': 'b', 'stance': '애매'},
        {'text': 'c', 'stance': '숨김'}, {'text': 'd', 'stance': '추측'}, {'text': 'e', 'stance': '미언급'},
    ])
    ai.witness_reply(CASE, state, '질문', state.statements[0])
    prompt = calls[0]['instructions']
    for stance in engine.STANCES:
        assert ai.STANCE_GUIDE[stance] in prompt
    assert '목적보다 이 태도가 우선' in prompt                   # 숨김은 목적(honest)보다 태도가 우선
    assert '질문만으로는 털어놓지 말고' in prompt and '증거가 법정에 제시되었을 때에만' in prompt  # 숨김은 증거로만 드러남
    assert '정정하세요' in prompt                              # 추측은 지적받으면 정정
    assert '태도 이름 자체는 입 밖에 내지 마세요' in prompt


def test_non_volunteering_stance_answers_when_asked_but_never_brings_it_up_and_is_not_concealment():
    guide = ai.STANCE_GUIDE['미언급']
    assert '먼저 꺼내지 않는' in guide and '직접 묻는 질문일 때만 사실대로' in guide
    assert '질문이 넓거나 다른 것을 묻는다면' in guide and '말하지 말고' in guide
    assert '숨기거나 둘러대서는 안 됩니다' in guide and '숨김과 다릅니다' in guide      # 숨김과의 차이
    hidden = ai.STANCE_GUIDE['숨김']
    assert '부인하거나' in hidden and '부인하거나' not in guide


def test_non_volunteering_item_is_shown_with_its_stance_in_the_witness_prompt(monkeypatch):
    calls = capture_calls(monkeypatch, {'answer': '네.'})
    state = engine.new_state(CASE)
    ai.witness_reply(CASE, state, '시계는 그 뒤로 어땠나요?', state.statements[0])
    prompt = calls[0]['instructions']
    quiet = [item for item in CASE['witness']['view'] if item['stance'] == '미언급']
    assert quiet and all(f'[미언급] {item["text"]}' in prompt for item in quiet)
    assert ai.STANCE_GUIDE['미언급'] in prompt


def test_view_prompt_only_explains_stances_that_are_used(monkeypatch):
    calls = capture_calls(monkeypatch, {'answer': '네.'})
    state = engine.new_state(CASE)
    state.view = engine.normalize_view([{'text': 'a', 'stance': '확신'}])
    ai.witness_reply(CASE, state, '질문', state.statements[0])
    assert ai.STANCE_GUIDE['확신'] in calls[0]['instructions']
    assert ai.STANCE_GUIDE['숨김'] not in calls[0]['instructions']


def test_stance_guide_covers_exactly_the_engine_stances():
    assert set(ai.STANCE_GUIDE) == set(engine.STANCES)


def test_goal_guide_follows_goal_type(monkeypatch):
    calls = capture_calls(monkeypatch, {'answer': '네.'})
    state = engine.new_state(CASE)
    for goal_type in ('frame', 'protect'):
        case = {**CASE, 'witness': {**CASE['witness'], 'goal': {'type': goal_type, 'detail': 'D'}}}
        ai.witness_reply(case, state, '질문', state.statements[0])
        assert ai.GOAL_GUIDE[goal_type] in calls[-1]['instructions']
    assert set(ai.GOAL_GUIDE) == {'honest', 'frame', 'protect'}


def test_view_updates_are_reflected_in_the_next_prompt(monkeypatch):
    calls = capture_calls(monkeypatch, {'answer': '네.'})
    state = engine.new_state(CASE)
    engine.set_view(state, [{'text': '복도는 사실 어두웠다.', 'stance': '확신'}])
    ai.witness_reply(CASE, state, '질문', state.statements[0])
    assert '[확신] 복도는 사실 어두웠다.' in calls[0]['instructions']
    assert CASE['witness']['view'][0]['text'] not in calls[0]['instructions']


def test_react_prompt_has_cited_statements_claim_and_public_evidence_but_not_truth(monkeypatch):
    reaction = {'action': 'revise', 'target_id': 2, 'speech': '음', 'text': '새 증언',
                'view': [{'text': 'a', 'stance': '확신'}]}
    calls = capture_calls(monkeypatch, reaction)
    state = engine.new_state(CASE)
    first = engine.get(state, engine.add_after(state, 1, '9시에 봤습니다.'))
    second = engine.get(state, engine.add_after(state, first.id, '조명이 켜져 있었습니다.'))
    result = ai.witness_react(CASE, state, [first, second], '9시에 불이 켜져 있었다는 건 모순입니다.', 'blackout')
    prompt = calls[0]['instructions']
    assert CASE['evidence']['blackout']['description'] in prompt  # 법정에 공개된 증거
    assert '9시에 불이 켜져 있었다는 건 모순입니다.' in prompt     # 변호인의 대사
    assert f'[{first.id}] 9시에 봤습니다.' in prompt and f'[{second.id}] 조명이 켜져 있었습니다.' in prompt
    for fact in CASE['truth']:
        assert fact not in prompt
    assert result == {'action': 'revise', 'target_id': 2, 'speech': '음', 'text': '새 증언',
                      'view': [{'text': 'a', 'stance': '확신'}], 'cascade': []}


# ---------- 연쇄 반박: 증인이 무너진 증언에 기댄 다른 증언도 함께 처리 ----------

def test_react_asks_the_witness_to_find_statements_that_depended_on_the_refuted_one(monkeypatch):
    calls = capture_calls(monkeypatch, {'action': 'retract', 'target_id': 1, 'speech': '', 'text': None, 'view': None, 'cascade': []})
    state = engine.new_state(CASE)
    ai.witness_react(CASE, state, [state.statements[0]], '지적', 'blackout')
    prompt = calls[0]['instructions']
    assert '연쇄' in prompt and 'cascade' in prompt and '전제로 하거나 같은 사실에 기대고 있는' in prompt
    assert '당신의 목적에 따라 판단하세요' in prompt            # 정직한 증인과 거짓말하는 증인이 다르게 대응한다
    assert '방금 고른 증언 자체나 상관없는 증언은 넣지 마세요' in prompt
    for fact in CASE['truth']:                                  # 연쇄 판단에도 진실은 증인에게 가지 않는다
        assert fact not in prompt


def test_react_schema_has_a_cascade_list_that_survives_gemini_conversion(monkeypatch):
    calls = capture_calls(monkeypatch, {'action': 'retract', 'target_id': 1, 'speech': '', 'text': None, 'view': None, 'cascade': []})
    state = engine.new_state(CASE)
    ai.witness_react(CASE, state, [state.statements[0]], '지적', 'blackout')
    schema = calls[0]['schema']
    assert 'cascade' in schema['required']
    item = ai._to_gemini_schema(schema)['properties']['cascade']['items']
    assert item['properties']['action']['enum'] == ['retract', 'revise']
    assert item['properties']['id']['type'] == 'integer' and item['properties']['text']['nullable'] is True


def test_react_returns_the_cascade_and_treats_a_malformed_one_as_empty(monkeypatch):
    items = [{'id': 3, 'action': 'retract', 'text': None}]
    capture_calls(monkeypatch, {'action': 'retract', 'target_id': 1, 'speech': '', 'text': None, 'view': None, 'cascade': items})
    state = engine.new_state(CASE)
    assert ai.witness_react(CASE, state, [state.statements[0]], '지적', 'blackout')['cascade'] == items
    capture_calls(monkeypatch, {'action': 'retract', 'target_id': 1, 'speech': '', 'text': None, 'view': None, 'cascade': 'oops'})
    assert ai.witness_react(CASE, state, [state.statements[0]], '지적', 'blackout')['cascade'] == []


def test_offline_react_has_no_cascade():
    state = engine.new_state(CASE)
    assert ai.witness_react(CASE, state, [state.statements[0]], '지적', 'blackout')['cascade'] == []


# ---------- 클리어 판정(재판관 AI) ----------

def test_judge_clear_prompt_has_the_condition_the_truth_and_the_whole_situation(monkeypatch):
    calls = capture_calls(monkeypatch, {'cleared': True, 'reason': '핵심 모순 노출'})
    state = engine.new_state(CASE)
    engine.record_refuted(state, '이전에 거둔 증언')
    result = ai.judge_clear(CASE, state, 'blackout', '증인은 9시 조명이 켜져 있었다고 했는데 정전이었습니다',
                            ['9시쯤 사람을 봤습니다.', '조명이 켜져 있었습니다.'], ['"조명이 켜져 있었습니다." 증언이 철회되었습니다.'])
    prompt, user_input = calls[0]['instructions'], calls[0]['input']
    assert result == {'cleared': True, 'reason': '핵심 모순 노출'}
    assert CASE['clear_condition'] in prompt and all(fact in prompt for fact in CASE['truth'])   # 재판관만 안다
    assert '지시문' in prompt and '다른 증거나 다른 증언으로 같은 핵심 모순에 닿았다면' in prompt
    assert '일부 증언이 철회되었다는 사실만으로는 부족' in prompt
    for part in ('정전 기록', '9시 조명이 켜져 있었다고 했는데', '9시쯤 사람을 봤습니다.', '증언이 철회되었습니다', '이전에 거둔 증언'):
        assert part in user_input
    assert CASE['witness']['introduction'] in user_input                                          # 현재 남은 증언


def test_judge_clear_schema_and_verdict_are_validated_only_for_form(monkeypatch):
    calls = capture_calls(monkeypatch, {'cleared': False, 'reason': '아직 부족'})
    state = engine.new_state(CASE)
    assert ai.judge_clear(CASE, state, 'access', 'c', ['x'], [])['cleared'] is False
    assert set(calls[0]['schema']['required']) == {'cleared', 'reason'}


def test_judge_clear_is_not_available_offline_or_without_a_condition(monkeypatch):
    state = engine.new_state(CASE)
    assert ai.judge_clear(CASE, state, 'blackout', 'c', ['x'], []) is None             # 키 없음
    capture_calls(monkeypatch, {'cleared': True, 'reason': ''})
    no_condition = {k: v for k, v in CASE.items() if k != 'clear_condition'}
    assert ai.judge_clear(no_condition, state, 'blackout', 'c', ['x'], []) is None


def test_judge_clear_failure_returns_none_and_records_the_error(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    monkeypatch.setattr(ai, '_call_gemini', lambda *a, **k: (_ for _ in ()).throw(ValueError('boom')))
    monkeypatch.setattr(ai, 'last_error', '')
    state = engine.new_state(CASE)
    assert ai.judge_clear(CASE, state, 'blackout', 'c', ['x'], []) is None
    assert ai.last_error.startswith('[clear] ValueError')


def test_react_schema_has_target_id_and_stance_enum_that_survive_gemini_conversion(monkeypatch):
    calls = capture_calls(monkeypatch, {'action': 'retract', 'target_id': 1, 'speech': '', 'text': None, 'view': None})
    state = engine.new_state(CASE)
    ai.witness_react(CASE, state, [state.statements[0]], '지적', 'blackout')
    schema = ai._to_gemini_schema(calls[0]['schema'])
    assert schema['properties']['target_id']['type'] == 'integer'
    item = schema['properties']['view']['items']
    assert schema['properties']['view']['nullable'] is True
    assert item['properties']['stance']['enum'] == list(engine.STANCES)


def test_react_unknown_action_becomes_retract(monkeypatch):
    capture_calls(monkeypatch, {'action': 'dance', 'target_id': 1, 'speech': '', 'text': None, 'view': None})
    state = engine.new_state(CASE)
    assert ai.witness_react(CASE, state, [state.statements[0]], '지적', 'blackout')['action'] == 'retract'


# ---------- 변호인의 대사 판정 ----------

STATEMENTS = ['저는 어젯밤 9시에 복도에서 사람을 봤습니다.', '그때 복도 조명은 켜져 있었습니다.']
ARGUMENT = '증인은 9시에 복도에서 사람을 봤다고 했고 그때 조명이 켜져 있었다고 했는데 정전 기록은 9시 정전입니다.'  # 증언 번호 없음


def argument_response(**overrides):
    base = {'grounded': True, 'unsupported': '', 'cited': [1, 2], 'supported': True, 'conflicts': True,
            'confidence': 0.9, 'reason': '9시 정전'}
    return {**base, **overrides}


def test_judge_prompt_has_numbered_statements_evidence_claim_and_the_truth(monkeypatch):
    calls = capture_calls(monkeypatch, argument_response())
    ai._contradiction_cache.clear()
    ai.judge_argument(CASE, STATEMENTS, ARGUMENT, 'blackout')
    prompt, user_input = calls[0]['instructions'], calls[0]['input']
    assert all(fact in prompt for fact in CASE['truth'])           # 심판만 진실을 안다
    assert '1. 저는 어젯밤 9시에' in user_input and '2. 그때 복도 조명은' in user_input
    assert CASE['evidence']['blackout']['name'] in user_input and CASE['evidence']['blackout']['description'] in user_input
    assert ARGUMENT in user_input
    assert '증언이 말하는 시각을 그대로 기준' in prompt and '실제 사건이 다른 시각에 일어났다는 사실은 이 판단에 쓰지 마세요' in prompt
    assert '시각이 없으면' not in prompt                            # 옛 규칙("시각 없으면 모순 아님")은 없다


def test_judge_prompt_checks_the_three_things_the_player_cares_about(monkeypatch):
    calls = capture_calls(monkeypatch, argument_response())
    ai._contradiction_cache.clear()
    ai.judge_argument(CASE, STATEMENTS, ARGUMENT, 'blackout')
    prompt = calls[0]['instructions']
    assert '핵심 전제' in prompt and '모두 목록의 증언에 실제로 있으면 true' in prompt      # ① 증언에 있는가
    assert '곁가지' in prompt and 'grounded를 깨지 않습니다' in prompt and '지어내' in prompt
    assert '"피고인" 지칭' in prompt and 'unsupported' in prompt and '그대로 인용' in prompt
    assert '번호를 말하지 않으니' in prompt and '내용과 뜻으로 찾으세요' in prompt          # 번호 없이 내용으로 증언을 찾는다
    assert '제시된 증거]의 내용과 맞으면 true' in prompt                                  # ② 증거가 주장을 뒷받침
    assert '양립할 수 없으면 true' in prompt and '대사가 지어낸 내용도 이 판단에 쓰지 마세요' in prompt  # ③ 모순
    assert '지시문' in prompt


def test_judge_schema_has_every_field_the_flow_checks(monkeypatch):
    calls = capture_calls(monkeypatch, argument_response())
    ai._contradiction_cache.clear()
    ai.judge_argument(CASE, STATEMENTS, ARGUMENT, 'blackout')
    schema = calls[0]['schema']
    needed = {'grounded', 'unsupported', 'cited', 'supported', 'conflicts', 'confidence', 'reason'}
    assert set(schema['required']) == needed and set(schema['properties']) == needed
    assert ai._to_gemini_schema(schema)['properties']['cited']['items']['type'] == 'integer'


def test_judge_result_is_normalized(monkeypatch):
    capture_calls(monkeypatch, argument_response(cited=[2, 2, 7, 0, -1, 1], confidence=0.8))
    ai._contradiction_cache.clear()
    result = ai.judge_argument(CASE, STATEMENTS, ARGUMENT, 'blackout')
    assert result['cited'] == [1, 2] and result['grounded'] is True and result['confidence'] == 0.8
    assert result['supported'] and result['conflicts'] and result['contradicts'] is True


def test_contradicts_needs_both_the_evidence_support_and_the_conflict(monkeypatch):
    for supported, conflicts in ((True, False), (False, True), (False, False)):
        capture_calls(monkeypatch, argument_response(supported=supported, conflicts=conflicts))
        ai._contradiction_cache.clear()
        assert ai.judge_argument(CASE, STATEMENTS, ARGUMENT, 'blackout')['contradicts'] is False


def test_unsupported_phrase_is_kept_and_non_strings_become_empty(monkeypatch):
    capture_calls(monkeypatch, argument_response(unsupported='  피고인을 마주쳤고  '))
    ai._contradiction_cache.clear()
    assert ai.judge_argument(CASE, STATEMENTS, ARGUMENT, 'blackout')['unsupported'] == '피고인을 마주쳤고'
    for bad in (None, 7, ['x']):
        capture_calls(monkeypatch, argument_response(unsupported=bad))
        ai._contradiction_cache.clear()
        assert ai.judge_argument(CASE, STATEMENTS, ARGUMENT, 'blackout')['unsupported'] == ''


def test_judge_is_cached_per_statements_claim_and_evidence(monkeypatch):
    calls = capture_calls(monkeypatch, argument_response())
    ai._contradiction_cache.clear()
    ai.judge_argument(CASE, STATEMENTS, ARGUMENT, 'blackout')
    ai.judge_argument(CASE, STATEMENTS, ARGUMENT, 'blackout')
    assert len(calls) == 1
    ai.judge_argument(CASE, STATEMENTS + ['새 증언'], ARGUMENT, 'blackout')   # 증언이 바뀌면 다시 판정한다
    ai.judge_argument(CASE, STATEMENTS, ARGUMENT, 'access')
    assert len(calls) == 3


def test_judge_failure_is_safe_and_not_cached(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    monkeypatch.setattr(ai, '_call_json', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('x')))
    ai._contradiction_cache.clear()
    result = ai.judge_argument(CASE, STATEMENTS, ARGUMENT, 'blackout')
    assert result['grounded'] is False and result['cited'] == [] and result['contradicts'] is False
    assert result['supported'] is False and result['conflicts'] is False and result['unsupported'] == ''
    assert not ai._contradiction_cache


def test_offline_judge_needs_no_statement_number():
    blackout = CASE['evidence']['blackout']
    by_keyword = ai._offline_judge(blackout, STATEMENTS, '증인은 조명이 켜져 있었다고 했는데 이건 모순입니다')
    assert by_keyword['cited'] == [2] and by_keyword['contradicts'] is True and by_keyword['grounded'] is True
    assert by_keyword['supported'] and by_keyword['conflicts'] and by_keyword['unsupported'] == ''
    nothing = ai._offline_judge(blackout, ['밤에 봤습니다.'], '이건 모순입니다')
    # 증언을 못 찾으면 '증언에 없는 내용'이 아니라 '어느 증언인지 모름'으로 안내되도록 grounded는 막지 않는다.
    assert nothing['cited'] == [] and nothing['grounded'] is True
    unrelated = ai._offline_judge(CASE['evidence']['access'], STATEMENTS, '증인의 말은 모순입니다')
    assert unrelated['cited'] == [] and unrelated['contradicts'] is False


def test_offline_judge_tolerates_a_number_but_does_not_depend_on_it():
    blackout, access = CASE['evidence']['blackout'], CASE['evidence']['access']
    assert ai._offline_judge(blackout, STATEMENTS, '2번 증언은 모순입니다')['contradicts'] is True
    with_number = ai._offline_judge(access, STATEMENTS, '2번 증언은 모순입니다')
    assert with_number['cited'] == [2] and with_number['contradicts'] is False   # 번호는 보조일 뿐 판정은 내용(키워드)으로
    assert ai._offline_judge(blackout, STATEMENTS, '9번 증언은 모순입니다')['cited'] == [2]  # 범위 밖 번호는 무시


def test_repeats_refuted_skips_the_call_when_nothing_is_refuted(monkeypatch):
    calls = capture_calls(monkeypatch, {'repeats': True})
    assert ai.repeats_refuted(CASE, '아무 말', []) is False and calls == []
    assert ai.repeats_refuted(CASE, '복도는 환했습니다.', ['복도는 밝았습니다.']) is True
    assert '복도는 밝았습니다.' in calls[0]['instructions'] and '복도는 환했습니다.' in calls[0]['input']


def test_repeats_refuted_offline_ignores_whitespace():
    assert ai.repeats_refuted(CASE, '복도는  환하게 밝았습니다.', ['복도는 환하게 밝았습니다.'])
    assert not ai.repeats_refuted(CASE, '복도가 어두웠을 수 있습니다.', ['복도는 환하게 밝았습니다.'])


def test_repeats_refuted_fails_open_on_error(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    monkeypatch.setattr(ai, '_call_json', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('x')))
    assert ai.repeats_refuted(CASE, 'a', ['b']) is False


# ---------- 증인 답변: 한 질문에 하나의 사실 ----------

def test_witness_reply_returns_answer_and_prompt_has_context(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    captured = {}

    def fake_call(instructions, user_input, name, schema, max_tokens, temperature):
        captured.update(instructions=instructions, user_input=user_input, schema=schema)
        return {'answer': '  야간 경비입니다.  '}

    monkeypatch.setattr(ai, '_call_json', fake_call)
    state = engine.new_state(CASE)
    assert ai.witness_reply(CASE, state, '소개해 주세요', state.statements[0]) == '야간 경비입니다.'
    assert captured['user_input'] == '소개해 주세요'
    assert state.statements[0].text in captured['instructions']
    assert '그대로 법정 증언' in captured['instructions']
    for truth in CASE['truth']:  # 증인에게는 진실을 알려 주지 않는다
        assert truth not in captured['instructions']
    assert list(captured['schema']['properties']) == ['answer']


def test_witness_reply_keeps_only_the_first_sentence_even_if_the_model_adds_more(monkeypatch):
    capture_calls(monkeypatch, {'answer': '9시쯤이었습니다. 그리고 복도 조명도 켜져 있었습니다. 시계도 봤습니다.'})
    state = engine.new_state(CASE)
    assert ai.witness_reply(CASE, state, '몇 시였나요?', state.statements[0]) == '9시쯤이었습니다.'


def test_witness_reply_prompt_demands_one_fact_per_answer(monkeypatch):
    calls = capture_calls(monkeypatch, {'answer': '네.'})
    state = engine.new_state(CASE)
    ai.witness_reply(CASE, state, '몇 시였고 조명은 어땠나요?', state.statements[0])
    prompt = calls[0]['instructions']
    assert '한 문장, 하나의 사실만' in prompt
    assert '질문의 핵심에 해당하는 사실 하나만' in prompt and '덧붙이지 마세요' in prompt
    assert '같은 문장 안에서 흐려 말하세요' in prompt            # 애매한 기억도 문장을 나누지 않는다
    assert '1~2문장' not in prompt                                # 옛 규칙은 없다


def test_react_prompt_demands_a_one_sentence_revision(monkeypatch):
    calls = capture_calls(monkeypatch, {'action': 'retract', 'target_id': 1, 'speech': '', 'text': None, 'view': None})
    state = engine.new_state(CASE)
    ai.witness_react(CASE, state, [state.statements[0]], '지적', 'blackout')
    assert '한 문장, 하나의 사실만' in calls[0]['instructions'] and '1~2문장' not in calls[0]['instructions']
