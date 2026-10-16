"""질문/증거 제시 한 번의 흐름을 엔진과 AI로 조립한다."""
from __future__ import annotations

import ai
import engine

MAX_RETRIES = 2  # 반박된 주장을 되풀이했을 때 다시 쓰게 하는 횟수
REPEAT_NOTE = '방금 쓴 내용은 이미 증거로 반박된 주장과 같습니다. 그 주장을 다시 하지 말고 다른 방식으로 쓰세요.'


class AskError(Exception):
    """질문을 증언으로 만들 수 없을 때. 메시지를 화면에 그대로 보여 준다."""


def ask_question(case: dict, state: engine.GameState, base_id: int, question: str) -> int:
    """base_id 증언을 보며 한 질문. 증인의 답변(한 문장, 한 사실)이 base 바로 뒤에 새 증언으로 들어가고 그 id를 돌려준다.
    증언 수가 상한에 닿으면 engine이 게임오버로 만든다."""
    if state.solved or state.game_over:
        raise AskError('심문이 이미 끝났습니다.')
    base = engine.get(state, base_id)
    if base is None:
        raise AskError('질문할 증언을 찾을 수 없습니다.')
    answer, note = None, ''
    for _ in range(MAX_RETRIES + 1):
        candidate = ai.witness_reply(case, state, question, base, note)
        if not candidate:
            raise AskError(f'AI 호출에 실패했습니다: {ai.visible_error()}')
        if not ai.repeats_refuted(case, candidate, state.refuted):
            answer = candidate
            break
        note = REPEAT_NOTE
    if answer is None:
        raise AskError('증인이 반박된 주장을 되풀이할 수 없어 답변을 피했습니다. 다른 질문을 해 보세요.')
    new_id = engine.add_after(state, base_id, answer)
    if new_id is None:
        raise AskError('증인이 의미 있는 답변을 하지 않았습니다. 다시 질문해 주세요.')
    state.questions += 1
    state.log.append(('변호인', question))
    state.log.append(('증인', engine.get(state, new_id).text))
    return new_id


def _react(
    case: dict, state: engine.GameState, cited: list[engine.Statement], claim: str, evidence_id: str,
) -> dict:
    """모순 지적에 대한 증인의 반응. 인용된 증언 중 무엇을 어떻게 고칠지(target, action, text)를 정해 돌려준다.
    수정 문장은 한 문장 한 사실이어야 하고, 반박된 주장을 되풀이하면 다시 쓰게 한다. 끝내 안 되면 철회로 처리한다."""
    note, last = '', None
    for _ in range(MAX_RETRIES + 1):
        reaction = ai.witness_react(case, state, cited, claim, evidence_id, note)
        if reaction is None:
            break
        # 증인이 고른 증언이 인용된 것 중에 없으면 마지막 인용 증언을 대상으로 삼는다.
        target = next((s for s in cited if s.id == reaction.get('target_id')), cited[-1])
        last = {**reaction, 'target': target}
        if reaction['action'] != 'revise':
            return last
        text = engine.one_fact(reaction.get('text') or '')
        forbidden = state.refuted + [target.text]
        if text and text != target.text and not ai.repeats_refuted(case, text, forbidden):
            return {**last, 'text': text}
        note = REPEAT_NOTE
    speech = (last or {}).get('speech') or '그 부분은 철회하겠습니다.'
    return {
        'action': 'retract', 'target': (last or {}).get('target', cited[-1]),
        'speech': speech, 'text': None, 'view': (last or {}).get('view'), 'cascade': (last or {}).get('cascade') or [],
    }


def _cascade(case: dict, state: engine.GameState, target: engine.Statement, items: object) -> list[str]:
    """무너진 증언에 기대고 있던 다른 증언들을 증인이 정한 대로 함께 철회·수정한다. 처리한 내용을 한 줄씩 돌려준다.
    AI가 정한 항목은 형식만 검증한다(존재하는 증언, 방금 고른 증언 자신이 아님, 마지막 남은 증언은 철회 불가).
    수정 문장이 반박된 주장을 되풀이하면 철회로 바꾼다. 거둔 증언은 반박된 주장 목록에 올라 다시 말할 수 없다."""
    lines, done = [], {target.id}
    for item in items if isinstance(items, list) else []:
        sid = item.get('id') if isinstance(item, dict) else None
        statement = engine.get(state, sid) if isinstance(sid, int) and not isinstance(sid, bool) else None
        if statement is None or statement.id in done:
            continue
        done.add(statement.id)
        original = statement.text
        revised = None
        if item.get('action') == 'revise':
            text = engine.one_fact(item.get('text') or '')
            if text and text != original and not ai.repeats_refuted(case, text, state.refuted + [target.text, original]):
                revised = text
        if revised and engine.revise(state, statement.id, revised):
            engine.record_refuted(state, original)
            lines.append(f'연쇄로 "{original}" 증언도 "{revised}"(으)로 수정되었습니다.')
        elif engine.retract(state, statement.id):
            engine.record_refuted(state, original)
            lines.append(f'연쇄로 "{original}" 증언도 함께 철회되었습니다.')
    return lines


def _note(verdict: dict) -> str:
    """대사에 증인이 말한 적 없는 곁가지가 섞여 있으면, 그 구절을 근거에서 뺐다고 알려 준다."""
    if verdict.get('unsupported'):
        return f'\n\n※ 대사 중 "{verdict["unsupported"]}"은(는) 증인이 말한 적이 없어 근거에서 제외했습니다.'
    return ''


def present_evidence(case: dict, state: engine.GameState, claim: str, evidence_id: str) -> tuple[str, str]:
    """증거를 대사와 함께 제시한다. 변호인은 증언 번호를 말하지 않고, 증인이 한 말을 짚어서 증거와의 모순을 지적한다.
    판정 AI가 내용으로 ① 대사가 증언에 있는 내용인지 ② 어느 증언인지 ③ 증거가 주장을 뒷받침하는지 ④ 모순인지를 확인한다.
    (종류, 메시지)를 돌려준다. 종류: success | warning | info
    실패 메시지에는 판정 이유(진실이 섞일 수 있음)를 쓰지 않고 단계별 고정 문구만 쓴다."""
    if state.game_over:
        return 'info', '재판관이 이미 심문을 종결했습니다.'
    if state.solved:
        return 'info', '심문이 이미 끝났습니다.'
    claim = claim.strip()
    if not claim:
        return 'info', '대사를 입력하세요. 증인이 한 말을 짚어서, 이 증거와 왜 모순인지 말해야 합니다.'
    live = engine.active(state)
    verdict = ai.judge_argument(case, [s.text for s in live], claim, evidence_id)
    note = _note(verdict)
    cited = [live[n - 1] for n in verdict['cited'] if 1 <= n <= len(live)]
    if not verdict['grounded']:
        quoted = f'대사 중 "{verdict["unsupported"]}"은(는) 증인이 말한 적이 없습니다.' if verdict['unsupported'] \
            else '대사가 증인이 말하지 않은 내용을 근거로 삼고 있습니다.'
        return 'warning', f'{quoted} 이미 나온 증언만 근거로 지적하세요.'
    if not cited:
        return 'warning', ('대사에서 어떤 증언을 가리키는지 알 수 없습니다. 증인이 한 말을 짚어서 말하세요. '
                           '(예: "증인은 …라고 했는데, 이 증거는 …라고 합니다.")') + note
    if not verdict['supported']:
        return 'warning', '이 증거는 대사의 주장을 뒷받침하지 않습니다. 다른 증거를 고르거나 주장을 확인하세요.' + note
    if not verdict['conflicts']:
        return 'warning', '그 증언들은 이 증거와 함께 놓아도 모순되지 않습니다. 다른 증언이나 증거를 검토하세요.' + note
    if verdict['confidence'] < ai.CONFIDENCE_MIN:
        return 'info', '재판장: 증거와 증언의 관계가 불분명합니다. 더 분명한 근거를 제시하십시오.' + note

    pointed_texts = [s.text for s in cited]  # 증언이 고쳐지기 전의 문장(메시지와 재판관 판정에 쓴다)
    reaction = _react(case, state, cited, claim, evidence_id)
    target = reaction['target']
    original = target.text
    engine.record_refuted(state, original)  # 이후 증인은 이 주장을 다시 할 수 없다.
    if reaction['action'] == 'revise' and engine.revise(state, target.id, reaction['text']):
        outcome = f'증언이 다음과 같이 수정되었습니다: "{engine.get(state, target.id).text}"'
    elif engine.retract(state, target.id):
        outcome = '증언이 철회되었습니다.'
    else:
        outcome = '증인은 이 증언만 남았지만 그 주장은 더 이상 유지할 수 없게 되었습니다.'
    engine.set_view(state, reaction.get('view'))
    chained = _cascade(case, state, target, reaction.get('cascade'))

    # 클리어는 재판관 AI가 판단한다. AI를 쓸 수 없을 때(키 없음/호출 실패)에만 기존 규칙(결정적 증거)으로 되돌아간다.
    ruling = ai.judge_clear(case, state, evidence_id, claim, pointed_texts, [f'"{original}" {outcome}'] + chained)
    if ruling is not None:
        cleared = ruling['cleared']
    else:
        cleared = evidence_id == case['decisive']['evidence']
    if cleared:
        state.solved = True

    pointed = ' / '.join(f'"{text}"' for text in pointed_texts)
    reason = f' ({verdict["reason"]})' if verdict['reason'] else ''
    speech = f'\n\n증인: "{reaction["speech"]}"' if reaction['speech'] else ''
    chain = ''.join(f'\n→ {line}' for line in chained)
    head = '이의 있음! 결정적인 모순입니다!' if state.solved else '이의 있음!'
    return 'success', (
        f'{head} {pointed} 증언은 이 증거와 함께 사실일 수 없습니다.{reason}\n\n→ "{original}" {outcome}{chain}{speech}{note}'
    )
