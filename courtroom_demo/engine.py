"""증언 상태. AI 호출 없이 순수하게 동작한다.

증언에는 참/거짓 값이 없다. 질문 한 번은 증언 하나(한 문장, 한 사실)를 질문한 증언 바로 뒤에 추가하고,
증거로 반박된 증언만 철회되거나 수정된다. 반박된 증언의 문장은 `refuted`에 남아 증인이 다시 말하지 못한다.
증언 수가 상한(`limit`)에 닿으면 재판관이 더 이상의 심문은 무의미하다고 보고 게임오버가 된다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_TEXT = 200
DEFAULT_LIMIT = 20
MAX_VIEW = 30  # 증인이 아는 사실의 개수 상한. 사건의 view가 이보다 길면 뒤쪽 항목이 조용히 잘리므로 넉넉히 둔다.

# 증인이 아는 진실(view) 항목의 태도. 증인 AI가 그 믿음을 어떻게 말할지를 정한다(뜻은 ai.STANCE_GUIDE).
# 키를 늘릴 때는 여기와 ai.STANCE_GUIDE 두 곳만 고치면 된다.
STANCES = ('확신', '애매', '숨김', '추측', '미언급')
DEFAULT_STANCE = '확신'

_SENTENCE_BREAK = re.compile(r'(?<=[.!?。])\s+')


@dataclass
class Statement:
    id: int
    text: str
    # original | added(질문으로 삽입) | revised(반박 뒤 수정) | retracted(반박 뒤 철회)
    status: str = 'original'
    history: list[str] = field(default_factory=list)


@dataclass
class GameState:
    statements: list[Statement]
    view: list[dict] = field(default_factory=list)     # 증인이 아는 진실 [{'text', 'stance'}]. 반박되면 바뀐다.
    refuted: list[str] = field(default_factory=list)   # 반박된 증언 문장들. 증인은 다시 말할 수 없다.
    limit: int = DEFAULT_LIMIT                         # 증언 수 상한. 닿으면 게임오버.
    questions: int = 0
    solved: bool = False
    game_over: bool = False
    log: list[tuple[str, str]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    next_id: int = 1


def first_sentence(text: str) -> str:
    """답변에서 첫 문장만 남긴다(한 질문에 하나의 사실). 문장부호 뒤 공백을 문장의 끝으로 본다."""
    text = ' '.join(text.split())
    return _SENTENCE_BREAK.split(text, maxsplit=1)[0].strip()


def clip(text: str) -> str:
    """증언 길이 제한에 맞춘다. 넘치면 말줄임표로 자른다."""
    text = ' '.join(text.split())
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT - 1].rstrip() + '…'


def one_fact(text: str) -> str:
    """증언으로 저장할 문장: 첫 문장만, 길이 제한 안에서."""
    return clip(first_sentence(text))


def normalize_view(items: object, fallback: str = DEFAULT_STANCE) -> list[dict]:
    """view 항목을 [{'text', 'stance'}]로 맞춘다. 태도가 없는 문자열과 모르는 태도는 fallback으로 바꾸고,
    빈 항목은 버린다. 항상 새 딕셔너리를 만들어 돌려주므로 원본(사건 데이터)은 바뀌지 않는다."""
    if not isinstance(items, list):
        return []
    result = []
    for item in items:
        if isinstance(item, str):
            text, stance = item, fallback
        elif isinstance(item, dict) and isinstance(item.get('text'), str):
            text, stance = item['text'], item.get('stance')
        else:
            continue
        text = clip(text)
        if text:
            result.append({'text': text, 'stance': stance if stance in STANCES else fallback})
    return result[:MAX_VIEW]


def new_state(case: dict) -> GameState:
    """증언은 증인의 자기소개 하나로 시작한다. 나머지는 변호인의 질문으로 쌓인다."""
    witness = case['witness']
    intro = Statement(id=1, text=witness['introduction'])
    return GameState(
        statements=[intro], view=normalize_view(witness['view']), next_id=2,
        limit=case.get('statement_limit', DEFAULT_LIMIT),
    )


def active(state: GameState) -> list[Statement]:
    return [s for s in state.statements if s.status != 'retracted']


def get(state: GameState, statement_id: int) -> Statement | None:
    return next((s for s in active(state) if s.id == statement_id), None)


def successor(state: GameState, statement_id: int) -> int | None:
    """보던 증언이 철회되었을 때 이어서 볼 증언 id(다음 증언, 없으면 앞 증언). 살아 있으면 그대로, 활성 증언이 없으면 None."""
    if get(state, statement_id) is not None:
        return statement_id
    gone = next((s for s in state.statements if s.id == statement_id), None)
    if gone is None:
        live = active(state)
        return live[0].id if live else None
    position = state.statements.index(gone)
    for candidate in state.statements[position + 1:] + state.statements[:position][::-1]:
        if get(state, candidate.id) is not None:
            return candidate.id
    return None


def add_after(state: GameState, base_id: int, text: str) -> int | None:
    """질문한 증언(base_id) 바로 뒤에 새 증언(첫 문장 하나)을 끼워 넣는다. 넣을 수 없으면 None.
    증언 수가 상한에 닿으면 게임오버가 된다(이미 클리어했다면 아니다)."""
    base = get(state, base_id)
    text = one_fact(text)
    if base is None or not text or state.solved or state.game_over or len(active(state)) >= state.limit:
        return None
    statement = Statement(id=state.next_id, text=text, status='added')
    state.next_id += 1
    state.statements.insert(state.statements.index(base) + 1, statement)
    state.events.append(f'🆕 증언 {base.id} 뒤에 새 증언 {statement.id} 추가: "{statement.text}"')
    if len(active(state)) >= state.limit:
        state.game_over = True
    return statement.id


def record_refuted(state: GameState, text: str) -> None:
    """반박된 증언의 문장을 기록한다. 이후 증인은 같은 주장을 다시 할 수 없다."""
    if text not in state.refuted:
        state.refuted.append(text)


def retract(state: GameState, statement_id: int) -> bool:
    """반박된 증언을 철회한다. 마지막 남은 증언은 철회할 수 없다."""
    statement = get(state, statement_id)
    if statement is None or len(active(state)) <= 1:
        return False
    statement.history.append(statement.text)
    statement.status = 'retracted'
    state.events.append(f'🗑️ 증언 {statement.id} 철회: "{statement.text}"')
    return True


def revise(state: GameState, statement_id: int, text: str) -> bool:
    """반박된 증언을 새 문장(첫 문장 하나)으로 고친다. 이전 문장은 history에 남는다."""
    statement = get(state, statement_id)
    text = one_fact(text)
    if statement is None or not text or text == statement.text:
        return False
    state.events.append(f'✏️ 증언 {statement.id} 수정: "{statement.text}" → "{text}"')
    statement.history.append(statement.text)
    statement.text = text
    statement.status = 'revised'
    return True


def set_view(state: GameState, items: object) -> bool:
    """증인이 아는 진실을 갱신한다. 리스트가 아니거나 쓸 만한 항목이 없으면 무시한다.
    AI가 준 목록이므로 모르는 태도는 단정하지 않는 '애매'로 처리한다."""
    cleaned = normalize_view(items, fallback='애매')
    if not cleaned:
        return False
    state.view = cleaned
    return True
