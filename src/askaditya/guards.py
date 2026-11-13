"""Guards for the six false claims that do not depend on the model.

``safety.TRAPS`` recognises a question that tries a false claim and an answer that asserts one. These guards go
further: whenever a question or an answer is about one of the six projects AND about the topic where the false claim
lives (savings, success, reliability, "better than", "stops all"), the answer must carry the caveat that the sources
attach to that claim. If it does not, the caveat is added, or, for a question that is itself phrased as the claim,
the answer is replaced by the true statement when it asserts the claim, and gets the caveat added otherwise. It runs on every answer, whatever the phrasing of the question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .safety import TRAPS, Trap, _normalise, asserts_false_claim, split_follow_up, trap_for_question


@dataclass(frozen=True)
class Guard:
    trap: Trap
    topic: re.Pattern  # the topic words where the false claim lives
    required: re.Pattern  # what a correct answer on that topic says
    caveat: str  # a true sentence from the sources, added when the answer lacks it


def _g(key: str, topic: str, required: str, caveat: str) -> Guard:
    trap = next(t for t in TRAPS if t.key == key)
    return Guard(trap, re.compile(topic, re.I), re.compile(required, re.I), caveat)


GUARDS = [
    _g(
        "promptlab",
        r"(random|better|beat|best|outperform|\bwin|evolution|accura|improv|score|search|compare|baseline)",
        r"(\btie|\bties|\btied|not beat|does not beat|no better|synthetic|equal budget)",
        "On the two real tasks, CommonsenseQA and BoolQ, its evolutionary search ties random search at equal budget; it is ahead only on a synthetic task.",
    ),
    _g(
        "modelrouter",
        r"(sav|cost|cheap|money|production|\blive\b|traffic|budget|expens|price|efficien|reduc|95|37|quality)",
        r"(offline|recorded|not (from )?live|no live|replay)",
        "These figures come from an offline analysis of recorded RouterBench outcomes, not from live traffic.",
    ),
    _g(
        "sftforge",
        r"(better|improv|train|quality|benefit|result|help|effect|\bwork|worth|boost|perform|stronger)",
        r"(no measurable|no practical|not shown|no benefit|did not (find|improve)|not measurably)",
        "Its own experiment found no measurable benefit from the cleaned data at that scale.",
    ),
    _g(
        "sentinel",
        r"(stop|block|catch|\ball\b|accura|effective|detect|protect|secur|safe|\brate|how good|how well|miss|fool|perfect|reliab|prevent)",
        r"(one layer|not a complete|misses|not stop|does not stop|doesn't stop|not catch|not a guarantee|cannot)",
        "It is one layer of defence, not a complete one: at its default setting it misses a large share of unseen attacks.",
    ),
    _g(
        "agentflow",
        r"(success|reliab|accura|\brate|perform|work well|\bgood\b|complete|score|percent|how often|how well|finish)",
        r"(does not claim|not claim|did not work|no high|not high|not a high|small model|measured)",
        "It does not claim a high task success rate; its README reports the measured results, including the parts that did not work.",
    ),
    _g(
        "evalbench",
        r"(judge|reliab|trust|agree|valid|human|rubric|how well|how good)",
        r"(does not claim|not claim|only where no rule|no rule exists|not highly|exact[- ]match)",
        "It does not claim its rubric judge is highly reliable: exact-match and rule checks come first and the judge is used only where no rule exists.",
    ),
]

_RISKY_FOLLOW_UP = re.compile(
    r"(cost|sav|success|reliab|accura|better|stop|block|beat|random|judge|train|perform|effective|quality)",
    re.I,
)
SAFE_FOLLOW_UP = "Ask me about his other projects."


def guards_in_scope(question: str, answer: str) -> list[Guard]:
    """Guards whose project is named in the question or the answer and whose topic words appear in the
    question (or whose false claim the answer itself asserts): a question about EvalBench's checkpoints does not
    get a caveat about its judge."""
    text = _normalise(f"{question} {answer}")
    asked = _normalise(question)
    claimed = asserts_false_claim(answer)
    return [
        g
        for g in GUARDS
        if g.trap.question.search(text)
        and (g.topic.search(asked) or (claimed is not None and claimed.key == g.trap.key))
    ]


def enforce_caveats(question: str, answer: str) -> tuple[str, str | None]:
    """Return the answer with the required caveats, and a note of what was done (or None)."""
    claim = trap_for_question(question)
    body, follow = split_follow_up(answer)
    notes: list[str] = []
    for g in guards_in_scope(question, answer):
        if g.required.search(answer):
            continue
        if claim is not None and claim.key == g.trap.key and asserts_false_claim(answer):
            return claim.safe + " Ask me about his other projects.", f"replaced:{g.trap.key}"
        body = f"{body.rstrip()} {g.caveat}"
        notes.append(f"caveat:{g.trap.key}")
    if not notes:
        return answer, None
    return (f"{body} {follow}".strip() if follow else body), ",".join(notes)


def sanitise_follow_up(answer: str) -> str:
    """Never invite a question about one of the six projects on a topic where the false claims live."""
    body, follow = split_follow_up(answer)
    if not follow:
        return answer
    low = _normalise(follow)
    if _RISKY_FOLLOW_UP.search(low) and any(t.question.search(low) for t in TRAPS):
        return f"{body} {SAFE_FOLLOW_UP}".strip()
    return answer
