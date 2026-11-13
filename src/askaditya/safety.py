"""The stages that wrap the model: input validation, an injection screen, deterministic handling of topics
that are not published, an off-topic check, and the verification of every answer before it is sent.

Each stage is a plain function with no network access, so each is unit-tested on its own. The injection
screen is this project's own rule list (it does not use Sentinel's rules or classifier).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rag_chatbot.text import content_tokens, tokens

from .build import phone_like

PLACEHOLDER = "[ASSISTANT-PLATFORM]"
MAX_QUESTION = 500
MAX_HISTORY = 6
MAX_HISTORY_TEXT = 1000


class BadRequest(ValueError):
    """The request body is not valid; ``str(exc)`` is safe to return to the caller."""


# ------------------------------------------------------------------------------------ 1. validation
def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return re.sub(r"[ \t]+", " ", text).strip()


def validate_request(payload: object) -> tuple[str, list[dict[str, str]]]:
    """Return the cleaned question and history, or raise ``BadRequest``."""
    if not isinstance(payload, dict):
        raise BadRequest("the body must be a JSON object")
    q = payload.get("question")
    if not isinstance(q, str):
        raise BadRequest("question must be a string")
    q = _clean(q)
    if not 1 <= len(q) <= MAX_QUESTION:
        raise BadRequest(f"question must be 1 to {MAX_QUESTION} characters")
    hist = payload.get("history", [])
    if hist is None:
        hist = []
    if not isinstance(hist, list) or len(hist) > MAX_HISTORY:
        raise BadRequest(f"history must be a list of at most {MAX_HISTORY} items")
    out = []
    for item in hist:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            raise BadRequest("each history item needs a role of user or assistant")
        text = item.get("text")
        if not isinstance(text, str) or len(text) > MAX_HISTORY_TEXT:
            raise BadRequest(
                f"each history text must be a string of at most {MAX_HISTORY_TEXT} characters"
            )
        out.append({"role": item["role"], "text": _clean(text)})
    return q, out


# ------------------------------------------------------------------------------- 2. injection screen
_INJECTION = [
    (
        "override",
        r"\b(ignore|disregard|forget|override|bypass|skip|drop)\b.{0,40}\b(previous|prior|above|earlier|all|any|your|the|these|those|system)\b.{0,30}\b(instruction|prompt|rule|guideline|direction|constraint|restriction|filter|safeguard|guardrail|polic)",
    ),
    (
        "override",
        r"\b(ignore|disregard|forget)\b.{0,12}\b(everything|all of that|the above|what you were told)\b",
    ),
    (
        "reveal",
        r"\b(reveal|show|print|repeat|display|output|leak|tell me|give me|what (is|are|was|were))\b.{0,40}\b(system prompt|your (initial |hidden |secret )?(prompt|instruction|rules|guidelines|configuration)|the (prompt|instructions) (you|above))",
    ),
    ("reveal", r"\bsystem (prompt|message)\b"),
    (
        "persona",
        r"\b(you are now|from now on you|act as|pretend (to be|you)|role-?play|roleplay|imagine you are|behave as|simulate (a|an|being)|your new (role|persona|name))\b",
    ),
    (
        "persona",  # instructions aimed at the assistant, not the topic words ("does Sentinel block jailbreaks?")
        r"\b(do anything now|developer mode|god mode|dan mode)\b|\b(you (are|will be|must be|should be)|you'?re|be|become|act (as|like)|behave (as|like)|respond as|answer as)\b.{0,20}\b(dan|unfiltered|jailbroken|uncensored|unrestricted)\b|\b(enable|activate|enter|switch to|turn on|unlock)\b.{0,15}\b(jailbreak|dan|god|developer) ?(mode)?\b|\b(answer|respond|reply|talk|speak|act|behave|operate)\b.{0,30}\b(no restrictions|without (any )?(restrictions|filters|rules))\b|\b(you have|with) no (restrictions|rules|filters)\b",
    ),
    (
        "delimiter",
        r"(</?\s*(system|assistant|user|instructions?)\s*>|\[/?(inst|system)\]|<\|(im_start|im_end|system|endoftext)\|>|^\s*(system|assistant)\s*:)",
    ),
    ("delimiter", r"\b(new|updated|additional) (instructions?|rules?|system prompt)\s*[:\-]"),
    (
        "reveal",  # "repeat the text above", "translate your instructions", "starting with 'You are'"
        r"\b(repeat|print|output|echo|recite|reproduce|display|show|read back|copy|dump|translate|summari[sz]e|paraphrase|rephrase|tell me|give me|write out|say)\b.{0,40}\b(text|words|messages?|content|everything|conversation|context)\b.{0,25}\b(above|before this|before my|so far|earlier|preceding|verbatim|word for word)\b",
    ),
    ("reveal", r"\b(text|everything|words|content|messages?) (above|before this)\b"),
    ("reveal", r"\bstarting with ['\"“‘]?you are\b"),
    (
        "reveal",
        r"\b(repeat|print|output|echo|recite|reproduce|display|show|reveal|translate|summari[sz]e|paraphrase|rephrase|list|read|share|tell me|give me|leak|dump|write out|disclose|explain)\b.{0,25}\b(your|these|those|the (initial|original|hidden|secret|system|developer|above|previous|first|full|exact))\s+(\w+ )?(instructions?|prompts?|rules|guidelines|directives|configuration|setup|programming|policies|policy)\b",
    ),
    ("reveal", r"\b(developer|system) (message|prompt|instructions?)\b"),
    ("delimiter", r"#{2,}\s*(system|assistant|user|human|instructions?|prompt|new|developer)\b"),
    ("delimiter", r"<\|[^>]{0,30}\|>|\[/?inst\]|<<\s*/?sys\s*>>|\[/?system\]"),
    (
        "override",  # the same instruction in Spanish, French, German and Hindi
        r"\b(ignora\w*|ignore[zr]?|olvida\w*|descarta\w*|oubli\w*|vergiss\w*|ignorier\w*|missacht\w*)\b.{0,40}\b(instrucci\w+|instructions?|r[eè]gles?|reglas?|anweisung\w*|regeln|pr[eé]c[eé]dent\w*|anterior\w*|vorherig\w*|prompt|sistema|syst[eè]me)\b|(अनदेखा|भूल|नज़रअंदाज़|नजरअंदाज).{0,30}(निर्देश|नियम)|(निर्देश|नियम).{0,30}(अनदेखा|भूल|नज़रअंदाज़|नजरअंदाज)|\b(ignore|bhool)\s+(karo|kar do|jao)\b",
    ),
    (
        "reveal",
        r"\b(prompt|mensaje|message|nachricht) (del |du |de |des )?(sistema|syst[eè]me|systeme)\b|\bsystem-?prompt\b|\bprompt syst[eè]me\b|\b(tu|su|votre|ton|dein\w*) (prompt|instrucciones|instructions|anweisungen)\b|(आपके|तुम्हारे|अपने) (निर्देश|प्रॉम्प्ट)|\b(my|your|the) instructions (say|state|tell|are)\b|\binstructions say\b",
    ),
    (
        "delimiter",  # a fake tool result or a JSON blob that carries a role
        r"\btool (result|output|response|message|call)s?\s*[:=]|\bfunction[_ ]?(call|result)s?\s*[:=]|[\"']role[\"']\s*:\s*[\"'](system|assistant|developer|tool)|[\"']content[\"']\s*:\s*[\"']",
    ),
    ("delimiter", r"(?:^|[#>\])\n]|\s{2,})\s*(system|assistant|developer|human)\s*:"),
    (
        "reveal",  # (a) questions about the assistant's own instructions, rules, configuration, hidden text
        r"\b(your|this assistant'?s|the assistant'?s|the bot'?s|the chatbot'?s)\s+(\w+\s)?(instructions?|prompts?|rules|guidelines|directives|config\w*|constraints|restrictions|programming|policies|policy)\b",
    ),
    (
        "reveal",
        r"\bwhat (\w+ )?(were|was|are|have) (you|they|we) (been )?(told|given|instructed|configured|programmed|prompted|asked)\b|\b(you|assistant) (were|was|have been|had been) (told|given|instructed|configured|programmed|prompted)\b|\bbefore (this|our|the|your) (conversation|chat|session|exchange)\b|\b(hidden|secret|internal|initial|original) (text|message|prompt|instructions?|rules|context)\b|\b(show|print|reveal|display|dump|output|leak|expose)\b.{0,20}\b(config\w*|settings|environment|env vars?|keys?|tokens?|secrets?|credentials)\b",
    ),
    (
        "authority",  # (b) authority, permission and role claims
        r"\bas your (developer|admin\w*|owner|creator|maker|programmer|boss|operator)\b|\bas the (developer|admin\w*|owner|creator|operator|maker) of (this|the) (bot|assistant|chatbot|system|site)\b|\bas (aditya|anthropic|openai|groq|google)\W+(i|we)\b|\bi ?(am|'m|’m) (aditya|your (developer|admin\w*|owner|creator|boss|operator|maker)|the (developer|admin\w*|owner|creator|operator|maker)|from (anthropic|openai|google|groq))\b|\b(i|we) (hereby )?(authori[sz]e|allow|permit|order|command|instruct|direct|require|demand) you\b|\byou (are|have been|were) (now |hereby )?(allowed|permitted|authori[sz]ed|free|cleared|unlocked)\b|\b(restrictions?|limits?|filters?|rules|guardrails|safeguards?|safety) (are |have been |were |is )?(lifted|removed|disabled|suspended|waived|void)\b|(?:^|[.!?;:,]\s+|\b(?:please|now|just|and|then)\s+)(drop|remove|disable|lift|turn off|bypass|ignore|override|circumvent)\s+(all |any |the |your |these |those )*(restrictions?|limits?|filters?|rules|guardrails|safeguards?|safety|policies|constraints)\b|\b(admin|override|sudo|god) (code|password|mode)\b|\bsudo\b|\bunlock (yourself|your)\b",
    ),
    (
        "persona",  # (c) persona and mode switches
        r"\b(switch|change|alter|modify|update|reset|enter|activate|enable)\s+(yourself|your)\b.{0,20}\b(mode|persona|role|behaviou?r|personality|name|character)\b|\bstop being (an? )?(assistant|chatbot|bot)\b|\bact (like|as if|as though)\b|\byou('re| are) (no longer|not) (an? )?(assistant|chatbot|bot|bound)\b|\bbreak (character|free)\b",
    ),
    (
        "exfil",
        r"\b(your|the (server|system|app|backend|deployment|assistant)'?s?)\s+(\w+\s)?(api[ -]?keys?|secrets?|passwords?|credentials?|environment variables?|env vars?|tokens?)\b.{0,30}\b(print|show|reveal|leak|give|tell|what)",
    ),
    (
        "exfil",
        r"\b(print|show|reveal|leak|dump|give|tell)( me| us)?\s+(your |the |all |every )?(\w+\s)?(api[ -]?keys?|secrets?|passwords?|credentials?|environment variables?|env vars?)\b",
    ),
    (
        "encoded",
        r"\b(base64|rot13|hex[- ]?encoded|decode this|in reverse order)\b.{0,40}\b(instruction|prompt|command)",
    ),
    (
        "tool",  # a command to run now, not "does AgentFlow run Python?"
        r"\b(execute|run|eval)\b.{0,12}\b(this|the following|my|these)\b.{0,12}\b(code|command|script|shell|python|sql)\b|\b(execute|run|eval)\s*:",
    ),
]
_LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)


# letters from other alphabets that look like Latin ones (Cyrillic and Greek homoglyphs)
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ї": "i",
        "ѕ": "s", "ј": "j", "ԁ": "d", "ɡ": "g", "ӏ": "l", "һ": "h", "ԛ": "q", "ԝ": "w", "к": "k",
        "м": "m", "н": "h", "т": "t", "в": "b", "ο": "o", "α": "a", "ν": "v", "ι": "i", "ε": "e",
        "ρ": "p", "τ": "t", "υ": "u", "κ": "k", "χ": "x", "η": "n", "ω": "w", "ɑ": "a", "ǝ": "e",
    }
)  # fmt: skip


def _normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = "".join(
        ch
        for ch in t
        if unicodedata.category(ch) not in ("Cf", "Co")
        and not (
            unicodedata.category(ch) == "Mn" and 0x300 <= ord(ch) <= 0x36F
        )  # Latin accents only
    )
    t = t.translate(_HOMOGLYPHS)
    return re.sub(r"\s+", " ", t)


def _despace(text: str) -> str:
    """ "i g n o r e  p r e v i o u s": words separated by two or more spaces, letters by one."""
    parts = re.split(r"\s{2,}", text.strip())
    return " ".join(p.replace(" ", "") if re.fullmatch(r"(?:\S ){2,}\S", p) else p for p in parts)


_SECOND = re.compile(
    r"\b(you|your|yours|yourself|you're|you've|you'd)\b|\b(text|everything|words|messages?|content) (above|before)\b|\bbefore (this|our) (conversation|chat)\b",
    re.I,
)
_THIRD = re.compile(
    r"\b(his|aditya'?s|rag chatbot'?s?|evalbench'?s?|agentflow'?s?|promptlab'?s?|sft[- ]?forge'?s?|diffuselab'?s?|visionlens'?s?|voicescribe'?s?|sentinel'?s?|modelrouter'?s?)\b",
    re.I,
)
_LONG_TOKEN = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/])")


def _encoded_blob(text: str) -> bool:
    """A long base64- or hex-looking token is a signal by itself: it is never decoded, only refused."""
    for m in _LONG_TOKEN.finditer(text):
        tok = m.group(0)
        if re.search(r"\d", tok) or re.search(r"[a-z][A-Z]", tok) or tok.endswith("="):
            return True
    return False


_HIJACK = re.compile(
    r"\b(respond|reply|answer|output|write|speak|talk|format|return)\b.{0,15}\b(in|as|using|with)\b.{0,4}\b(json|xml|yaml|html|base64|binary|hex|morse|emoji|pirate|l33t|reverse|uppercase|all caps|a poem|rhymes?|code blocks?)\b|\b(reply|respond|answer|say|output|print|type|write|repeat) (only|just|exactly|nothing but|solely)\b|\b(end|start) (every|each|your) (answer|reply|response|message) with\b|\bsay\b.{0,30}\b(and nothing else|nothing else|and only that)\b|\b(complete|finish|continue|fill in)\b.{0,12}\b(this|the following|my)\b.{0,15}\b(sentence|phrase|text|line)\b",
    re.I,
)


def screen_injection(text: str) -> str | None:
    """The id of the first rule that matches the text, or None. User text and history are screened alike."""
    if _encoded_blob(unicodedata.normalize("NFKC", text)):
        return "encoded"
    plain = _normalise(text)
    candidates = [
        plain,
        plain.translate(_LEET),
        re.sub(r"[^a-z0-9 ]", "", plain),
        _normalise(_despace(text)),
    ]
    spaced = re.sub(r"(?<=\b\w) (?=\w\b)", "", plain)  # "i g n o r e" -> "ignore"
    candidates.append(spaced)
    third = bool(_THIRD.search(plain)) and not _SECOND.search(plain)
    for rule, pattern in _INJECTION:
        if third and rule == "reveal":
            continue  # about a project's own prompt or rules, not this assistant's
        for c in candidates:
            if re.search(pattern, c, flags=re.I | re.M):
                return rule
    if (
        len(re.findall(r"\b\w\b", plain)) >= 6
    ):  # letters spread out one by one: squash them and look again
        squashed = re.sub(r"[^a-z]", "", plain)
        if re.search(
            r"(ignore|disregard|forget|override|bypass)(all|any|your|the|these|those|previous|prior|above|earlier|system)*(instructions?|prompts?|rules?|guidelines?)",
            squashed,
        ):
            return "override"
    if _HIJACK.search(plain) and not _ABOUT_HIM.search(plain):
        return "format"  # (d) an output-format hijack in a request that is not about him
    return None


# --------------------------------------------------------------------- 3. topics that are not published
@dataclass(frozen=True)
class Deflection:
    topic: str
    text: str


def _email(email: str) -> str:
    return email


_TOPICS = [
    (
        "salary",
        r"\b(pay|pays|paying|paid)\b.{0,20}\b(him|aditya|he)\b|\b(he|aditya)\b.{0,15}\b(is paid|gets paid|get paid|earns?|makes money|draws?)\b|\bhow much\b.{0,30}\b(pay|paid|earn)\b",
    ),
    (
        "personal",
        r"\b(for fun|weekends?|spare time|leisure|pastimes?|free time|outside (of )?(work|his job)|off[- ]duty|after work|vacations?)\b",
    ),
    (
        "salary",
        r"\b(minimum|expected|current|desired|required|target|base|starting|asking|preferred|ideal)\b.{0,15}\b(pay|salary|compensation|requirements?|package|ctc|stipend|remuneration|rate|wage|income|money)\b|\b(pay|salary|compensation|ctc|stipend)\b.{0,15}\b(requirements?|expectations?|range|figure|number|demands?|ask)\b|\blooking for\b.{0,20}\b(compensation|pay|money|salary)\b",
    ),
    (
        "location",
        r"\b(hybrid(?! (retrieval|search|ranking|fusion|index|bm25|rag))|remote(?! (api|server|code|repo|service|control|call|procedure))|on-?site|wfh|work from home|in-?office|office-based|telecommut\w*)\s+(arrangement|preference|option|setup|model|work|working|role|job|basis|days?|policy|schedule|position)\b|\b(consider|prefer\w*|open to|willing|okay with|fine with|happy with|comfortable with|up for)\b.{0,25}\b(hybrid(?! (retrieval|search|ranking|fusion|index|bm25|rag))|remote(?! (api|server|code|repo|service|control|call|procedure))|on-?site|wfh|work from home|in-?office|travel\w*)\b",
    ),
    (
        "salary",
        r"\b(ctc|lpa|lakhs?|pay ?(grade|band|scale|slab|package)|salary band|earn(s|ing|ings)?|earned|income|take[- ]home|paid|what('?s| is| does| do) (he|aditya) (on|get|gets|make|makes|take|takes|draw)|per (annum|month|year)|annual (pay|income|package)|compensation package|total package|package (he|aditya|is|expectations?)|(what|which|roughly|approx\w*|expected|current|desired|his|aditya'?s)\b.{0,25}\bpackage\b(?!.{0,30}\b(manager|python|pip|npm|install|import|librar))|how much (does|do|would|will|is|are)\b.{0,15}\b(he|aditya|they)\b.{0,15}\b(cost|charge|want|ask)|what (does|would|will) he (charge|cost|ask for|want))\b",
    ),
    (
        "start",
        r"\b(can|could|will|would|does|do) (he|aditya)\b.{0,12}\b(start|join|begin|onboard|come on board|take up)\b|\bwhen\b.{0,20}\b(can|could|will|would|should|is|are|does)\b.{0,10}\b(he|aditya)\b.{0,15}\b(start|join|begin|onboard|be free|be available)|\b(start|starting|joining|join) (on |from |by |in |next |this )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|next (week|month)|immediately|asap|soon|date)|\bhow soon\b|\bavailable (from|immediately|right away|asap|next|by|on)\b|\bavailability (date|to join|to start)\b|\bimmediate joiner\b|\bserv(e|es|ing) (a|his|the) notice\b|\blast working day\b|\bjoin (us|our|the team|my team|the company|full[- ]?time)\b|\bearliest (start|joining)\b",
    ),
    (
        "grades",
        r"\b(report card|transcripts?|academic (record|performance|results?|standing)|how did he do in (college|university|his degree|exams)|mark ?sheet|his scores in)\b",
    ),
    (
        "location",
        r"\b((willing|open|ready|happy|able) to (relocate|move|travel|commute)|where (is|are) he (willing|open) to work|city preference|prefer(red)? (location|city|country)|move (abroad|overseas|to another)|can (he|aditya) work (hybrid|remotely|on-?site|from home|in the office|from)|(prefer|like|want)\w*\b.{0,25}\b(to work|to live|work from|location|city))\b",
    ),
    (
        "visa",
        r"\b(immigration|eligible to work|legally (allowed|authori[sz]ed)|work in (the )?(us|usa|uk|canada|europe|germany|australia|singapore)|(us|uk|canadian|eu) (work )?(permit|citizen)|oci|opt|stem opt|tn status)\b",
    ),
    (
        "phone",
        r"\b(dial|landline|can i (call|ring|phone|text|whatsapp)|call (him|aditya)|phone (him|aditya)|(his|aditya'?s) (mobile|cell)\b|digits to call|voice call|speak (to|with) him on the phone)\b",
    ),
    (
        "paper",
        r"\b(thesis|dissertation|where will (it|the paper) be published|paper (title|topic|venue|submission|status)|(his|aditya'?s) research\b)",
    ),
    (
        "salary",
        r"\b(salary|salaries|ctc|lpa|compensation|pay (expectation|range|scale)|expected (pay|salary)|how much (does|would|will|is|do)\b.{0,25}\b(earn|paid|make|cost to hire|charge)|stipend|remuneration|wage|hourly rate|rate card)\b",
    ),
    (
        "grades",
        r"\b(cgpa|gpa|grades?|marks|percentage|percentile|academic score|class rank|rank in (his )?(class|college)|scored in (his )?(exams?|college|semester)|semester (results?|scores?))\b",
    ),
    (
        "start",
        r"\b(notice period|start date|joining date|date of joining|when (can|could|will|would) he (start|join)|how soon (can|could) he|available (to|from|for) (join|start)|availability to (join|start)|earliest (he )?(can|could)|immediate(ly)? (join|start|available))\b",
    ),
    (
        "location",
        r"\b(relocat\w*|remote (work|working|role|job|position|only|first|setup|friendly)|(work|working|works) remotely|fully remote|open to remote|work from home|wfh|hybrid (work|working|role|model of work|setup)|on-?site|in-?office|willing to move|open to moving|work location|office location|commute)\b",
    ),
    (
        "visa",
        r"\b(visa|work permit|work authori[sz]ation|citizenship|sponsorship|green card|h-?1b|right to work|nationality|passport)\b",
    ),
    (
        "phone",
        r"\b(phone|(mobile|cell|telephone) ?(number|no)|contact number|whatsapp|call him|ring him|his number|number to call|text him)\b",
    ),
    (
        "paper",
        r"\b(research paper|his paper|publications?|journal|conference paper|arxiv|co-?authors?|paper'?s? (title|topic|abstract|venue|result)|what is (his|the) (paper|research) (about|on))\b",
    ),
    (
        "fifteenforteen",
        r"\b(15 ?forteen|fifteen ?forteen)\b.{0,80}\b(what did|responsibilit\w*|achievement|work(ed)? on|duties|do there|did he do|details?|tell me more|accomplish\w*|projects?|impact|day[- ]to[- ]day)\b|\b(what did|responsibilit\w*|duties|did he do|details?|accomplish\w*|tell me more|work(ed)? on|which projects?|any details?)\b.{0,60}\b(15 ?forteen|fifteen ?forteen)\b",
    ),
    (
        "personal",
        r"\b(married|wife|girlfriend|boyfriend|spouse|partner|dating|age|how old|birthday|date of birth|born|religion|caste|politic\w*|family|parents|father|mother|sibling|brother|sister|hobb(y|ies)|home address|where does he live exactly|street|salary of his|personal life|gender|ethnicity|health|medical|disabilit\w*|free time)\b",
    ),
]


def unpublished_topic(question: str) -> str | None:
    q = _normalise(question)
    for topic, pattern in _TOPICS:
        if re.search(pattern, q, flags=re.I):
            if (
                topic == "salary"
                and re.search(
                    r"\b(python|pip|npm|pypi|librar\w+|install\w*|import|manager|software)\b", q
                )
                and not re.search(r"\b(salary|ctc|compensation|pay|paid|earn\w*)\b", q)
            ):
                continue  # a software package, not a pay package
            return topic
    return None


_OUTPUT_TOPICS = [
    (
        "salary",
        r"\b(salary|salaries|ctc|lpa|compensation|stipend|remuneration|pay (expectations?|range|band|scale)|expected pay|lakhs?)\b",
    ),
    (
        "start",
        r"\b(notice period|joining date|start date|available (to|from|for) (join|start)|can (start|join)|earliest (start|joining)|immediate joiner)\b",
    ),
    ("grades", r"\b(cgpa|gpa|report card|transcript)\b"),
    (
        "location",
        r"\b(relocat\w*|remote (work|role|only|first)|hybrid (work|role|arrangement|setup)|on-?site (work|role|only)|work from home|wfh|willing to (move|travel))\b",
    ),
    (
        "visa",
        r"\b(visa|work permit|work authori[sz]ation|citizenship|sponsorship|green card|h-?1b)\b",
    ),
    (
        "personal",
        r"\b(married|spouse|wife|girlfriend|boyfriend|date of birth|years old|religion|caste)\b",
    ),
]
_PAPER_OK = re.compile(
    r"coming soon|once (it is |the paper is )?published|working on a research paper", re.I
)
_PAPER_ANY = re.compile(
    r"\b(research paper|his paper|publications?|journal|arxiv|co-?authors?|submitted to|accepted (at|to))\b",
    re.I,
)
_FIFTEEN_DETAIL = re.compile(
    r"responsib|dut(y|ies)|achiev|worked on|projects?|impact|built|led\b|managed|developed|delivered|reported|team|clients?|tasks?",
    re.I,
)


def unpublished_in_answer(answer: str) -> str | None:
    """The unpublished topic an answer talks about, whatever the question was: pay, joining date, grades,
    relocation or work arrangement, visa, family or age, details of the paper, details of 15FORTEEN beyond the
    role and dates. A model must not say these even if a passage seems to, so the caller replaces the answer."""
    text = _normalise(answer)
    for topic, pattern in _OUTPUT_TOPICS:
        if re.search(pattern, text, flags=re.I):
            return topic
    if _PAPER_ANY.search(text) and not _PAPER_OK.search(text):
        return "paper"
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if re.search(r"15 ?forteen", sentence) and _FIFTEEN_DETAIL.search(sentence):
            return "fifteenforteen"
    return None


def deflection(topic: str, email: str) -> Deflection:
    mail = _email(email)
    texts = {
        "salary": f"Compensation isn't published, and it is best discussed with Aditya directly: {mail}.",
        "grades": f"His grades and CGPA aren't published. You can ask him directly: {mail}.",
        "start": f"His notice period and start date aren't published. You can ask him directly: {mail}.",
        "location": f"His relocation and remote-work preferences aren't published. You can ask him directly: {mail}.",
        "visa": f"His visa or work-authorisation status isn't published. You can ask him directly: {mail}.",
        "phone": f"I don't share a phone number. The way to reach Aditya is email: {mail}, or through the LinkedIn and GitHub links on his portfolio.",
        "paper": f"He is working on a research paper, and the details will be shared once it is published; nothing more is available yet. You can ask him directly: {mail}.",
        "fifteenforteen": f"He held a Founder's Office role at 15FORTEEN, Delhi, from June 2025 to February 2026. No further detail is published; you can ask him directly: {mail}.",
        "personal": f"That's personal and isn't published. I can tell you about his work, projects and experience instead, and you can ask him directly: {mail}.",
    }
    return Deflection(topic, texts[topic] + " Ask me about his projects or experience.")


# ---------------------------------------------------------------------------- 4. assistant and off-topic
_META = re.compile(
    r"\b(who are you|what are you|are you (an? )?(ai|bot|human|real|chatbot|robot|person)|how do you work|who (made|built|created|developed|programmed) you|how (were|are) you (built|made|trained)|what (model|llm|technology) (are you|do you use|powers you)|powered by|about (this|the) (assistant|chatbot|bot)|this (assistant|chatbot)|your (name|purpose|sources|knowledge)|what can you (do|answer|help)|what do you know)\b",
    re.I,
)
_OFFTOPIC = re.compile(
    r"\b(weather|forecast|temperature outside|joke|poem|song|lyrics|recipe|horoscope|stock price|bitcoin|crypto price|exchange rate|football|cricket score|world cup|election|president of|prime minister of|capital of|population of|translate|(write|generate|create|code|implement)\b.{0,25}\b(code|function|script|program|essay|story|email|poem|song|cover letter|class|sql query|regex)|(fix|debug) (my|this|the) (code|bug|error)|solve (this|the|my)|what is \d+ ?[+*/x-] ?\d+|calculate|what('s| is)\s+\d+(\.\d+)?\s*(times|plus|minus|divided by|multiplied by|x|\*|\+|/)\s*\d+|derivative of|integral of|who won|latest news|movie|netflix|restaurant|travel plan|how to (cook|bake|lose weight|invest|fix|install|center)|meaning of life|tell me about (yourself|the universe)|best (phone|laptop|programming language|framework))\b",
    re.I,
)
_ABOUT_HIM = re.compile(
    r"\b(aditya|he|his|him|resume|résumé|cv|portfolio|projects?|experience|skills?|education|certif\w*|contact|email|linkedin|github|ethara|genesis|ggsipu|university|intern\w*|co-?lead|evalbench|agentflow|promptlab|sft ?forge|diffuselab|visionlens|voicescribe|sentinel|modelrouter|rag|post-?training)\b",
    re.I,
)


_CONTACT = re.compile(
    r"\b(contact (him|aditya|info|details|information)|(how|where|best way)\b.{0,30}\b(contact|reach|email|e-?mail|message|get in touch|hire)\b|get in touch|reach (him|aditya|out)|his (email|e-?mail|linkedin|github( profile| account| page| link)?|r[ée]sum[ée]( link)?|cv)|(email|e-?mail) (address|id)|linkedin (profile|link|url)|github (profile|link|url|account)|(link|url) to (his )?(r[ée]sum[ée]|cv|linkedin|github)|download (his )?(r[ée]sum[ée]|cv)|r[ée]sum[ée] (link|pdf|download)|(portfolio|personal) (website|site|page|url|link)|his (website|site|homepage|portfolio)|where can i (find|see|view|read|browse) his (portfolio|work|code|projects?|repos?|repositories|source)|his (code|repos|repositories|source code)|link to his (code|repos|projects)|where is (his|the) code)\b",
    re.I,
)


def is_contact_question(question: str) -> bool:
    """How to reach him (email, LinkedIn, GitHub, résumé link): answered from the Contact passage, no model."""
    q = question.lower()
    if re.search(
        r"\b(latest|most recent|newest|on his (cv|r[eé]sum[eé])|in his (cv|r[eé]sum[eé]))\b", q
    ) and not re.search(
        r"\b(link|url|download|pdf|email|e-mail|linkedin|github|contact|website)\b", q
    ):
        return False
    return bool(_CONTACT.search(question))


_SUMMARY = re.compile(
    r"\b(who is (aditya|he)\b|who'?s (aditya|he)\b|introduce (him|aditya)|(an? )?(short |quick |brief )?(bio|introduction|intro|elevator pitch)\b|summar(y|ise|ize) (of )?(him|aditya)|describe (him|aditya)|what (does|do) (aditya|he) do\b|(tell|say|give) me (something )?about (aditya|him)\b(?! ?(\'s|’s))(?!.{0,40}\b(project|experience|skills?|education|work|certif|contact|role|internship)))",
    re.I,
)
_PROJECT_WORDS = re.compile(
    r"\b(evalbench|agentflow|promptlab|sft|diffuselab|visionlens|voicescribe|sentinel|modelrouter|rag|chatbot|project)",
    re.I,
)


_SUMMARY2 = re.compile(
    r"\b(summar(y|ise|ize|ised)|sum up|outline|overview|rundown)\b.{0,25}\b(him|aditya|his (background|profile|career|r[ée]sum[ée]|cv|story|journey))\b|\b(his|aditya'?s) (background|story|journey)\b|\b(describe|outline) (his|aditya'?s) (background|profile|career)\b|\bbackground (of|on) (him|aditya)\b|\bwho is this (person|guy)\b|\bwhat('s| is) he (like|about)\b",
    re.I,
)


def is_summary_question(question: str) -> bool:
    """ "Tell me about Aditya in two sentences", "who is he": answered from the About passage, no model."""
    return bool(_SUMMARY.search(question) or _SUMMARY2.search(question)) and not (
        _PROJECT_WORDS.search(question) or _ENTITY_WORDS.search(question)
    )


_ENTITY_WORDS = re.compile(
    r"\b(ethara|genesis|15 ?forteen|ggsipu|universit\w*|college|degree)\b", re.I
)
_FABRICATE = re.compile(
    r"\b(add|insert|include|say|state|claim|write|mention|append|pretend|assert)\b.{0,40}\b(that )?(he|aditya|his)\b.{0,40}\b(has|have|is|was|won|holds?|graduated|worked|got|earned|achieved|speaks?|knows?)\b|\b(add|say|claim|state|write|include|mention|insert)\b.{0,40}\b\d+\+? (years?|yrs)\b",
    re.I,
)


def is_fabrication_request(question: str) -> bool:
    """A request to put a claim about him into the answer ("... and add that he has 10 years of experience")."""
    return bool(_FABRICATE.search(question))


def is_assistant_question(question: str) -> bool:
    return bool(_META.search(_normalise(question)))


_TASK = re.compile(
    r"^\s*(please\s+)?(can you |could you |would you )?(fix|debug|write|solve|calculate|compute|translate|refactor|rewrite|convert|correct|optimi[sz]e|generate|compose|draw)\b|\b(fix|debug|solve|translate|rewrite|refactor|correct)\s+(this|the following|my|these)\b",
    re.I,
)


def is_offtopic_pattern(question: str) -> bool:
    q = _normalise(question)
    if _ABOUT_HIM.search(q):
        return False
    return bool(_OFFTOPIC.search(q) or _TASK.search(q))


def mentions_him(question: str) -> bool:
    return bool(_ABOUT_HIM.search(_normalise(question)))


# ------------------------------------------------------------------------------- the six false claims
@dataclass(frozen=True)
class Trap:
    key: str
    project: str
    question: re.Pattern
    assertion: re.Pattern
    safe: str


def _t(key, project, q, a, safe):
    return Trap(key, project, re.compile(q, re.I), re.compile(a, re.I), safe)


TRAPS = [
    _t(
        "promptlab",
        "PromptLab",
        r"promptlab|evolution\w* prompt|prompt ?lab",
        r"(evolution\w*|promptlab|it)\b.{0,60}\b(beats?|outperform\w*|better than|superior to|wins? against|surpass\w*)\b.{0,40}\brandom",
        "No. On the two real tasks, CommonsenseQA and BoolQ, PromptLab's evolutionary search ties random search at equal budget instead of beating it; it is ahead only on a synthetic task. Its README reports the measured comparison.",
    ),
    _t(
        "modelrouter",
        "ModelRouter",
        r"model ?router|\brouter\b",
        r"(modelrouter|it|router)\b.{0,60}\b(saves?|saved|saving|cuts?|reduces?|lowers?)\b.{0,50}\b(live|production|real[- ]world|real traffic|actual|customers?)",
        "No. ModelRouter's cost figures come from an offline analysis of recorded RouterBench outcomes, not from live traffic.",
    ),
    _t(
        "sftforge",
        "SFT Forge",
        r"sft[- ]?forge",
        r"(cleaned|cleaning|clean data|the data|it)\b.{0,60}\b(trains?|produces?|gives?|yields?|improves?|makes?)\b.{0,30}\b(better|improved|stronger|higher[- ]quality) model",
        "No. SFT Forge's own experiment found no measurable benefit from the cleaned data at that scale.",
    ),
    _t(
        "sentinel",
        "Sentinel",
        r"sentinel",
        r"(sentinel|it)\b.{0,50}\b(stops?|blocks?|catches?|detects?|prevents?)\b.{0,25}\b(all|every|100|any|each)\b.{0,25}(attack|injection|prompt)",
        "No. Sentinel is one layer of defence, not a complete one: at its default setting it misses a large share of unseen attacks.",
    ),
    _t(
        "agentflow",
        "AgentFlow",
        r"agent ?flow",
        r"(agentflow|it)\b.{0,50}\b(high|great|excellent|strong|impressive|near[- ]perfect|reliable)\b.{0,25}(success|completion|accuracy|task)",
        "No. AgentFlow does not claim a high task success rate; its README reports the measured results, including the parts that did not work.",
    ),
    _t(
        "evalbench",
        "EvalBench",
        r"eval ?bench",
        r"(rubric )?(judge|llm[- ]as[- ]judge|it)\b.{0,50}\b(is|are|was)\b.{0,15}\b(highly |very |extremely )?(reliable|accurate|trustworthy|dependable)",
        "No. EvalBench does not claim its rubric judge is highly reliable; it uses exact-match and rule checks first and the judge only where no rule exists, and its README reports the measured results.",
    ),
]
_NEG = re.compile(
    r"\b(not|no|n't|never|nor|without|only|ties|tied|tie|offline|misses|missed|cannot|can't|doesn't|does not|isn't|aren't|fails?|unproven|no measurable|rather than|instead of|neither|isn't)\b",
    re.I,
)


def trap_for_question(question: str) -> Trap | None:
    q = _normalise(question)
    triggers = {
        "promptlab": r"(beat|better|outperform|win|superior|than random|vs\.? random|versus random|random search|random baseline|best|always|general)",
        "modelrouter": r"(sav|cost|cheap|live|production|traffic|real|reduc|money|budget|expens)",
        "sftforge": r"(better|improv|train|boost|quality|model|benefit|help|worth)",
        "sentinel": r"(all|every|stop|block|catch|100|perfect|complete|any|fool|bypass|immune|secure|safe|protect)",
        "agentflow": r"(success|reliab|high|accura|rate|complete|work|perform|good|percent|%)",
        "evalbench": r"(judge|reliab|accura|trust|agree|valid|human)",
    }
    for t in TRAPS:
        if t.question.search(q) and re.search(triggers[t.key], q):
            return t
    return None


def asserts_false_claim(answer: str) -> Trap | None:
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        for t in TRAPS:
            if t.assertion.search(sentence) and not _NEG.search(sentence):
                return t
    return None


# -------------------------------------------------------------------------------- 5. output verification
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")
_FOLLOW_UP = re.compile(
    r"(?:^|(?<=[.!?])\s+)(ask me about|you can ask|want to know|would you like|ask about|i can tell you)[^.!?]*[.!?]?\s*$",
    re.I,
)


_LIST_MARKER = re.compile(
    r"(?m)^\s*(?:[-*]\s*)?\d{1,2}[.)]\s+|(?<![\w.,])\(?[1-9]\)(?=\s)|(?<=[\s:;(])\*{0,2}[1-9]\.\*{0,2}\s+(?=[A-Z*])"
)


def numbers_in(text: str) -> set[str]:
    out = set()
    for m in _NUMBER.finditer(_LIST_MARKER.sub("", text)):
        n = m.group(0).rstrip("%").replace(",", "").rstrip(".")
        if n:
            out.add(n)
    return out


def split_follow_up(answer: str) -> tuple[str, str]:
    m = _FOLLOW_UP.search(answer)
    if not m:
        return answer, ""
    return answer[: m.start()].strip(), answer[m.start() :].strip()


@dataclass
class Verdict:
    ok: bool
    answer: str
    reason: str = ""


MAX_WORDS = 110
SUPPORT_MIN = 0.5
_SKIP_CAPS = {
    "he",
    "his",
    "aditya",
    "jain",
    "ask",
    "the",
    "it",
    "in",
    "at",
    "on",
    "for",
    "and",
    "but",
    "also",
    "a",
    "an",
    "no",
    "yes",
    "see",
    "you",
    "i",
    "this",
    "that",
    "these",
    "there",
    "its",
}


_MONTHS = {
    "january": "jan",
    "february": "feb",
    "march": "mar",
    "april": "apr",
    "june": "jun",
    "july": "jul",
    "august": "aug",
    "september": "sep",
    "sept": "sep",
    "october": "oct",
    "november": "nov",
    "december": "dec",
}


def _canon(word: str) -> str:
    w = word.lower().strip(".'")
    return _MONTHS.get(w, w)


def _seen_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for tok in tokens(text):
        for part in re.split(r"[-+:]", tok):
            if part:
                out.add(_canon(part))
                out.add(_canon(part).rstrip("s"))
    return out


def _supported(word: str, seen: set[str], prefixes: set[str]) -> bool:
    w = _canon(word)
    return w in seen or w.rstrip("s") in seen or (len(w) >= 5 and w[:5] in prefixes)


def _unseen_names(body: str, haystack: str) -> list[str]:
    seen = _seen_tokens(haystack)
    prefixes = {s[:5] for s in seen if len(s) >= 5}
    bad = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
        words = [
            w[:-2] if w.endswith("'s") else w
            for w in re.findall(r"(?<![0-9A-Za-z])[A-Za-z][A-Za-z0-9'.+#-]*", sentence)
        ]
        for i, w in enumerate(words):
            if i == 0 or not w[0].isupper() or w.lower() in _SKIP_CAPS or len(w) < 3:
                continue
            parts = [p for p in re.split(r"[-+:]", w) if len(p) >= 2 and not p.isdigit()]
            if parts and not all(
                _supported(p, seen, prefixes) or p.lower() in _SKIP_CAPS for p in parts
            ):
                bad.append(w)
    return bad


def verify_output(answer: str, passages: list[str], question: str = "") -> Verdict:
    """Check an answer against the passages it was generated from. Runs before anything is sent."""
    text = answer.strip()
    if not text:
        return Verdict(False, "", "empty")
    if PLACEHOLDER in text:
        return Verdict(False, "", "placeholder")
    if phone_like(text):
        return Verdict(False, "", "phone")
    haystack = "\n".join(passages)
    body, follow = split_follow_up(text)
    bad_numbers = sorted(numbers_in(text) - numbers_in(haystack))
    if bad_numbers:
        return Verdict(False, "", f"number not in sources: {bad_numbers[:3]}")
    trap = asserts_false_claim(text)
    if trap:
        return Verdict(False, "", f"false claim: {trap.key}")
    check = body or text
    words = content_tokens(check)
    if words:
        # words the visitor used count as supported (an echo of the question is not a new fact); numbers and
        # names are still checked against the passages alone
        seen = _seen_tokens(haystack) | _seen_tokens(question)
        prefixes = {s[:5] for s in seen if len(s) >= 5}
        support = sum(_supported(w, seen, prefixes) for w in words) / len(words)
        if support < SUPPORT_MIN:
            return Verdict(False, "", f"low support {support:.2f}")
    names = _unseen_names(check, haystack)
    if names:
        return Verdict(False, "", f"name not in sources: {names[:3]}")
    # length: keep whole sentences up to the limit
    if len(text.split()) > MAX_WORDS:
        kept: list[str] = []
        for s in re.split(r"(?<=[.!?])\s+", body or text):
            if len(" ".join(kept + [s]).split()) > MAX_WORDS - 15:
                break
            kept.append(s)
        if not kept:
            return Verdict(False, "", "too long")
        text = " ".join(kept) + (
            (" " + follow)
            if follow and len(" ".join(kept).split()) + len(follow.split()) <= MAX_WORDS
            else ""
        )
    return Verdict(True, text)
