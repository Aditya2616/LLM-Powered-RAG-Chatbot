"""Answers to the structured questions about him, from the portfolio's own sections, with no model.

Certifications, education, the roles he has held, leadership, skills (and "does he know X"), the list of projects,
where he is based and whether he is open to roles all have one right answer in the corpus, and a recruiter asks
them first. A language model sometimes declines them or drops a part, so they are answered here, from the full
sections kept in the index (``profile``), and never reach a provider.
"""

from __future__ import annotations

import datetime as dt
import re

MAX_WORDS = 100

_GLOSSARY = re.compile(
    r"\b(mean|means|meaning|stand for|stands for|define|definition|explain|difference|why|how (does|did|do))\b",
    re.I,
)
_HE = r"(he|aditya|his|him)"

_LEAD = re.compile(
    r"\b(led|lead|leads|leader|leadership|manag\w*|mentor\w*|supervis\w*|team lead|co-?lead)\b",
    re.I,
)
_CERT = re.compile(
    r"\b(certif\w*|credentials?|courses?|licen[cs]es|coursera|deeplearning|linkedin learning|deloitte|job simulation)\b",
    re.I,
)
_COURSEWORK = re.compile(
    r"\b(university|college|degree|semester|syllabus|curriculum|coursework|b\.?tech|subjects?|classes|class)\b",
    re.I,
)
_EDU_STOP = set(
    "related classes class courses course subject subjects module modules taken took take study studied degree semester college university".split()
)
_CERT_STOP = set(
    """a an the is are was were be been he his him aditya does do did have has had any which what who how many
    much list all of in on at for from with about to and or certification certifications certified certificate
    certificates certify credential credentials course courses taken take took hold holds got get earned earn
    completed complete done licence license licences licenses training program programs some anything""".split()
)
_STUDENT = re.compile(
    r"\b(is|are|was) (he|aditya)\b.{0,15}\b(a |an )?(student|undergrad\w*|studying|pursuing|enrolled)\b|\bis (he|aditya) (still )?(in|at) (college|university|school)\b",
    re.I,
)
_EDU = re.compile(
    r"\b(educat\w*|degree|universit\w*|college|studies|studied|study|graduat\w*|coursework|b\.?tech|major|school|gguipu|ggsipu|classes|subjects?)\b",
    re.I,
)
_COMPANY = [
    ("Ethara", re.compile(r"\bethara\b", re.I)),
    ("Genesis", re.compile(r"\bgenesis\b", re.I)),
    ("15FORTEEN", re.compile(r"\b15 ?forteen\b|\bfounder'?s office\b", re.I)),
]
_EXPERIENCE = re.compile(
    r"\b(experience|work history|career|employment|employer|worked|work (now|currently|right now)|currently|right now|current (role|job|position)|internship|intern|where does (he|aditya) work|what does (he|aditya) do at|roles?|positions?|jobs?|held|titles?|employed|latest|most recent|newest|before|previous(ly)?|earlier|past|formerly|company|employer|organi[sz]ation|firm|work(s|ing)? (for|at))\b",
    re.I,
)
_SKILLS = re.compile(
    r"\b(skills?|tech stack|technolog\w*|tools?|languages?|frameworks?|librar(y|ies)|proficien\w*|expertise|good at|strengths?|infra\w*|stack)\b",
    re.I,
)
_KNOWS2 = re.compile(
    r"\b(?:any|some|much|real|hands-on|practical|professional)\s+(?P<term>[\w.+#&/-]{2,30}(?: [\w.+#&/-]{2,20})?)\s+experience\b",
    re.I,
)
_KNOWS = re.compile(
    r"\b(know|knows|code in|codes in|program in|programs in|write code in|develop in|build with|script in|use|uses|using|used|work(ed|s)? (with|on)|experience (with|in|of)|experienced (with|in|at)|familiar (with)?|proficient (in|with)|skilled (in|at|with)|knowledge of|expertise in|comfortable with|tried|touched|handled?)\s+(?:the |any |of |in |with )?(?P<term>[\w .+#&/-]{2,40}?)\s*[?.!]*$",
    re.I,
)
_PROJECTS_SHORT = re.compile(r"\bprojects?\b(?:\W+\w+){0,4}\W*$", re.I)
_NOT_SKILLS = re.compile(r"\b(speak|spoken|fluent|hindi|english|native)\b", re.I)
_PROJECTS = re.compile(
    r"(\b(list|show|name|which|what|how many|all)\b.{0,30}\bprojects?\b|\bprojects?\b.{0,20}\b(has|did|does|have|had)\b.{0,6}\b(he|aditya)\b|\bwhat (has|did|does) (he|aditya) (built|build|make|made|created?|worked? on)\b|\b(his|aditya'?s) projects?\b|\bflagship\b|\bopen[- ]source\b|\bportfolio of\b)",
    re.I,
)
_LOCATION = re.compile(
    r"\b(where (is|does) (he|aditya) (live|based|located|stay)|based (in|where)|located|location|which city|what city|which country|time ?zone|where.{0,10}\bfrom\b)\b",
    re.I,
)
_OPEN = re.compile(
    r"\b(open to|looking for|available|opportunit\w*|hir(e|ing)|job search|seeking|on the market)\b|\b(what|which) (kind|type|sort)s? of (roles?|jobs?|work|positions?|opportunit\w*)\b|\b(roles?|jobs?|positions?|work) (is|does|would) (he|aditya) (after|want|seek\w*|prefer|look\w*)\b",
    re.I,
)

# which skill categories a question word points at
_CATEGORIES = [
    (r"\blanguages?\b|\bprogramming\b|\bcode in\b", ["Languages"]),
    (
        r"\bdeep learning\b|\bmachine learning\b|\bai ?/ ?ml\b|\breinforcement\b|\bfine-?tun\w*\b|\btraining\b",
        ["AI / ML"],
    ),
    (
        r"\bgenerative\b|\bllms?\b|\bprompt\w*\b|\brag\b|\bagents?\b|\blangchain\b",
        ["Generative AI"],
    ),
    (
        r"\blibrar(y|ies)\b|\bframeworks?\b|\bpytorch\b|\bdata\b",
        ["ML & data", "Generative AI", "Tools & infra"],
    ),
    (r"\btools?\b|\binfra\w*\b|\bdevops\b|\bdeploy\w*\b|\bstack\b", ["Tools & infra", "ML & data"]),
    (
        r"\bidentity\b|\baccess\b|\bsso\b|\bsaml\b|\bscim\b|\bentra\b|\bmicrosoft\b",
        ["Identity & access"],
    ),
    (
        r"\badmin\w*\b|\benterprise\b|\bplatform\b|\bconnectors?\b|\bmcp\b",
        ["AI platform admin", "Platform operations"],
    ),
    (r"\bdocument\w*\b|\breports?\b|\brunbooks?\b", ["Documentation"]),
]


def _words(text: str) -> int:
    return len(text.split())


def _fit(parts: list[str]) -> str:
    """Whole parts joined while they fit the word limit (the first part always stays)."""
    out: list[str] = []
    for p in parts:
        if out and _words(" ".join([*out, p])) > MAX_WORDS:
            break
        out.append(p)
    return " ".join(out)


def _bullets(text: str) -> list[str]:
    items = re.findall(r"(?m)^(?:- |\d+\. )(.+(?:\n {2,}\S.*)*)", text)
    return [re.sub(r"\s*\n\s*", " ", i).strip() for i in items]


_MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
_CURRENT = re.compile(
    r"\b(currently|right now|now|current|present|latest|most recent|newest)\b|\b(which|what) (company|organi[sz]ation|firm|employer)\b|\bwho (does|is) (he|aditya) (work|working|employed)\b|\bwork(s|ing)? (for|at)\b|\bemployer\b|\bat his (company|job|workplace)\b",
    re.I,
)
_HOW_LONG = re.compile(
    r"\b(how long|for how long|duration|tenure|how many (months|years)|since when|how much time)\b",
    re.I,
)
_STOP_TOPIC = set(
    """which what any are there the and for from with that this does did has have his him aditya built build work
    working worked projects project deal deals dealing about related around involving involve involves cover covers
    use uses using like kind kinds type types anything something where who how can could would should into onto over
    show tell give list name his its our your one some many much more most""".split()
)
_GENERIC_TOPIC = {
    "llm",
    "llms",
    "model",
    "models",
    "learning",
    "machine",
    "data",
    "language",
    "large",
}


class Profile:
    """The portfolio sections as kept in the index, and the answers built from them."""

    def __init__(self, sections: list[dict], chunks: list[dict], email: str) -> None:
        self.sections = {s["section"]: s["text"] for s in sections}
        self.chunks = chunks
        self.email = email

    # ----------------------------------------------------------------------------- lookups
    def _section(self, prefix: str) -> str:
        for k, v in self.sections.items():
            if k == prefix or (k.startswith(prefix) and " > " not in k[len(prefix) :]):
                return v
        return ""

    def _roles(self) -> list[tuple[str, list[str]]]:
        out = []
        for k, v in self.sections.items():
            if k.startswith("Experience > "):
                out.append((k.split(" > ", 1)[1], _bullets(v)))
        return out

    @staticmethod
    def _duration(head: str) -> str:
        """ "about 4 months" for ``Oct 2025 to Feb 2026``; a role that is current runs to today."""
        m = re.search(
            r"([A-Za-z]{3})[a-z]* (\d{4}) to (?:([A-Za-z]{3})[a-z]* (\d{4})|Present)", head, re.I
        )
        if not m or m.group(1).lower() not in _MONTHS:
            return ""
        y1, m1 = int(m.group(2)), _MONTHS[m.group(1).lower()]
        if m.group(4):
            y2, m2 = int(m.group(4)), _MONTHS.get(m.group(3).lower(), 12)
        else:
            today = dt.date.today()
            y2, m2 = today.year, today.month
        n = max(1, (y2 - y1) * 12 + (m2 - m1))
        years, months = divmod(n, 12)
        bits = []
        if years:
            bits.append(f"{years} year{'s' if years > 1 else ''}")
        if months:
            bits.append(f"{months} month{'s' if months > 1 else ''}")
        return "about " + " and ".join(bits)

    @staticmethod
    def _span(head: str) -> tuple[int, int] | None:
        """The months (year * 12 + month) a role covers, or None."""
        m = re.search(
            r"([A-Za-z]{3})[a-z]* (\d{4}) to (?:([A-Za-z]{3})[a-z]* (\d{4})|Present)", head, re.I
        )
        if not m or m.group(1).lower() not in _MONTHS:
            return None
        start = int(m.group(2)) * 12 + _MONTHS[m.group(1).lower()]
        if m.group(4):
            end = int(m.group(4)) * 12 + _MONTHS.get(m.group(3).lower(), 12)
        else:
            today = dt.date.today()
            end = today.year * 12 + today.month
        return start, end

    def in_period(self, question: str) -> str | None:
        """ "What did he do between June 2025 and February 2026?": the roles that overlap the period."""
        found = re.findall(r"(?:\b([A-Za-z]{3,9})\s+)?\b((?:19|20)\d\d)\b", question)
        if not found:
            return None
        lows = [int(y) * 12 + (_MONTHS.get(mo[:3].lower()) or 1) for mo, y in found]
        highs = [int(y) * 12 + (_MONTHS.get(mo[:3].lower()) or 12) for mo, y in found]
        lo, hi = min(lows), max(highs)  # a bare year covers the whole year
        parts = []
        for h, b in self._roles():
            span = self._span(h)
            if span and span[0] <= hi and span[1] >= lo:
                first = next((x for x in b if "no further detail" not in x.lower()), "")
                parts.append(self._role_line(h) + (f": {first.rstrip('.')}" if first else "") + ".")
        return _fit(["In that period he held:", *parts]) if parts else None

    @staticmethod
    def _role_line(head: str) -> str:
        parts = [p.strip() for p in head.split("·")]
        name, role = parts[0], parts[1] if len(parts) > 1 else ""
        when = parts[2] if len(parts) > 2 else ""
        place = parts[3] if len(parts) > 3 else ""
        when = re.sub(r"to Present", "to present", when)
        return f"{role} at {name}, {when}" + (f" ({place})" if place else "")

    # ------------------------------------------------------------------------------ answers
    def certifications(self, question: str) -> str | None:
        text = self._section("Certifications")
        items = _bullets(text)
        if not items:
            return None
        words = [
            w
            for w in re.findall(r"[a-z0-9.+&-]+", question.lower())
            if w not in _CERT_STOP and len(w) > 1
        ]
        picked, phrase = items, ""
        for n in range(
            len(words), 0, -1
        ):  # the longest run of the question's own words that names some
            for i in range(len(words) - n + 1):
                part = " ".join(words[i : i + n])
                hit = [c for c in items if part in c.lower()]
                if hit and len(hit) < len(items):
                    picked, phrase = hit, part
                    break
            if phrase:
                break
        if words and not phrase and picked is items:
            head = f"None of his {len(items)} listed certifications names {' '.join(words)}. He lists {len(items)}:"
        elif picked is items:
            head = f"He lists {len(items)} certifications:"
        elif len(picked) == 1:
            head = f"One of his {len(items)} certifications matches:"
        else:
            head = f"{len(picked)} of his {len(items)} certifications match:"
        body = (
            "; ".join(f"{i + 1}. {c}" for i, c in enumerate(picked))
            if picked is items
            else "; ".join(picked)
        )
        note = (
            " His résumé lists five of the seven; the portfolio lists all of them."
            if picked is items and re.search(r"r[ée]sum[ée]( PDF)? lists five", text, re.I)
            else ""
        )
        return _fit([head, body + ".", note.strip()] if note else [head, body + "."])

    def student(self) -> str | None:
        """ "Is he a student?": the degree in progress and the internship he holds alongside it."""
        items = [
            i for i in _bullets(self._section("Education")) if not i.lower().startswith("no cgpa")
        ]
        if not items:
            return None
        parts = [f"Yes, he is studying: {items[0].rstrip('.')}."]
        for h, _ in self._roles():
            if re.search(r"\bcurrent\b", h, re.I):
                parts.append(f"Alongside, he works as {self._role_line(h)}.")
                break
        return _fit(parts)

    def education(self, question: str = "") -> str | None:
        items = _bullets(self._section("Education"))
        if not items:
            return None
        keep = [i for i in items if not i.lower().startswith("no cgpa")]
        if re.search(
            r"\b(classes|class|courses?|coursework|subjects?|modules?|taken|took|take)\b",
            question,
            re.I,
        ):
            words = [
                w
                for w in re.findall(r"[a-z0-9+#]+", question.lower())
                if len(w) > 3 and w not in _CERT_STOP | _EDU_STOP
            ]
            for line in keep:
                m = re.match(r"Coursework:\s*(.+)", line, re.I)
                if not m or not words:
                    continue
                courses = [c.strip().rstrip(".") for c in m.group(1).split(",")]
                hit = [
                    c
                    for c in courses
                    if any(c.lower().startswith(w[:5]) or w[:5] in c.lower() for w in words)
                ]
                if hit:
                    verb = "matches" if len(hit) == 1 else "match"
                    return _fit(
                        [
                            f"Of his coursework, {len(hit)} {verb}: {', '.join(hit)}.",
                            line.rstrip(".") + ".",
                        ]
                    )
        return _fit([i.rstrip(".") + "." for i in keep])

    def experience(self, question: str) -> str | None:
        roles = self._roles()
        if not roles:
            return None
        named = [n for n, rx in _COMPANY if rx.search(question)]
        if re.search(
            r"\b(before|previous(ly)?|prior|earlier|past|former(ly)?|used to|other than|apart from)\b",
            question,
            re.I,
        ):
            older = [
                (h, b)
                for h, b in roles
                if not re.search(r"\bcurrent\b", h, re.I)
                and not any(n.lower() in h.lower() for n in named)
            ]
            if older:
                parts = ["Before his current role he held:"]
                for h, b in older:
                    first = next((x for x in b if "no further detail" not in x.lower()), "")
                    parts.append(
                        self._role_line(h) + (f": {first.rstrip('.')}" if first else "") + "."
                    )
                return _fit(parts)
        if not named and _CURRENT.search(question):
            named = ["current"]  # the role whose heading is marked current
        picked = [
            (h, b) for h, b in roles if not named or any(n.lower() in h.lower() for n in named)
        ]
        if not picked:
            return None
        how_long = bool(_HOW_LONG.search(question))

        def line(h: str) -> str:
            d = self._duration(h) if how_long else ""
            return self._role_line(h) + (f". That is {d}" if d else "")

        if len(picked) == 1:
            h, b = picked[0]
            return _fit(
                [
                    line(h) + ".",
                    *[x.rstrip(".") + "." for x in b if "no further detail" not in x.lower()],
                ]
            )
        parts = []
        for h, b in picked:
            first = next((x for x in b if "no further detail" not in x.lower()), "")
            parts.append(line(h) + (f": {first.rstrip('.')}" if first else "") + ".")
        return _fit(parts)

    def leadership(self) -> str | None:
        for h, b in self._roles():
            if re.search(r"\blead\b|co-lead", h, re.I):
                return _fit([self._role_line(h) + ".", *[x.rstrip(".") + "." for x in b]])
        return None

    def _skill_lines(self) -> list[tuple[str, str]]:
        lines = []
        for b in _bullets(self._section("Skills")):
            if ":" in b:
                cat, items = b.split(":", 1)
                lines.append((cat.strip(), items.strip()))
        return lines

    def skills(self, question: str) -> str | None:
        lines = self._skill_lines()
        if not lines:
            return None
        wanted: list[str] = []
        for rx, cats in _CATEGORIES:
            if re.search(rx, question, re.I):
                wanted += [c for c in cats if c not in wanted]
        by_name = dict(lines)
        picked = [(c, by_name[c]) for c in wanted if c in by_name] or lines
        parts = []
        for c, items in picked:
            names = [x.strip() for x in re.split(r",\s*(?![^()]*\))", items)]
            shown = ", ".join(names[:12]) + (" and more" if len(names) > 12 else "")
            parts.append(f"{c}: {shown}.")
        return _fit(parts)

    def _where(self, term: str) -> str | None:
        """Where a skill appears: "under Languages", "in the stack of X", "" when it is not listed, None when
        the term is too vague to test."""
        t = term.strip(" .?!").lower()
        t = re.sub(r"^(the|any|a|an)\s+", "", t)
        if len(t) < 2 or t in {
            "ai",
            "it",
            "data",
            "code",
            "things",
            "anything",
            "stuff",
            "work",
            "tech",
        }:
            return None
        forms = {t, t.rstrip("s")} if len(t) > 3 else {t}
        for form in forms:
            pat = rf"(?<![\w-]){re.escape(form)}(?![\w])"
            for cat, items in self._skill_lines():
                if re.search(pat, items, re.I):
                    return f"under {cat}: {items}"
            stacks = [
                m.group(1)
                for m in re.finditer(
                    r"(?m)^\d+\.\s+\*\*([^*\n]+)\*\*[^\n]*·[^\n]*·\s*([^\n]*)",
                    self._section("Projects"),
                )
                if re.search(pat, m.group(2), re.I)
            ]
            if stacks:
                return f"in the stack of {', '.join(stacks[:4])}"
        return ""  # a passing mention in README prose is not a skill

    def knows(self, term: str) -> str | None:
        """Yes/no for "does he know X": from the skills list, then the project stacks."""
        name = term.strip(" .?!")
        where = self._where(term)
        if where is None:
            return None
        if not where:
            return f"{name} is not listed among his skills or project stacks."
        if where.startswith("under"):
            return f"Yes. {name} is in his skills list, {where}."
        return f"Yes. {name} appears {where}."

    def knows_many(self, terms: list[str]) -> str | None:
        """Several skills in one question: the listed ones with where they appear, then the unlisted ones."""
        wheres = [(x.strip(" .?!"), self._where(x)) for x in terms]
        if any(w is None for _, w in wheres):
            return None
        yes = [f"{n} ({w.split(':')[0]})" for n, w in wheres if w]
        no = [n for n, w in wheres if not w]
        parts = []
        if yes:
            parts.append("Listed: " + ", ".join(yes) + ".")
        if no:
            joined = ", ".join(no)
            parts.append(
                f"{joined} {'is' if len(no) == 1 else 'are'} not listed among his skills or project stacks."
            )
        return " ".join(parts)

    def projects(self, question: str) -> str | None:
        text = self._section("Projects")
        entries = []
        for m in re.finditer(
            r"(?m)^\d+\.\s+\*\*(.+?)\*\*\s*(\([^)]*\))?\s*·\s*([^·\n]+?)(?:\s*·|\n)", text
        ):
            desc = m.group(3).strip()
            if desc.startswith("http"):
                desc = (m.group(2) or "").strip("() ")
            entries.append((m.group(1), desc))
        if not entries:
            return None
        if re.search(
            r"\bflagship\b|\bmain project\b|\bbest project\b|\bmost (important|impressive)\b",
            question,
            re.I,
        ):
            first = entries[0]
            return (
                f"His flagship project is the {first[0]}, open source: it answers questions about your own "
                'documents, ties each answer to its source and says "I don\'t know" instead of guessing.'
            )
        listing = "; ".join(f"{n} ({d})" if d else n for n, d in entries)
        return _fit(
            [
                f"He has {len(entries)} projects, all public on GitHub and MIT-licensed:",
                listing + ".",
            ]
        )

    def _project_entries(self) -> list[tuple[str, str, str]]:
        """(name, one-liner, whole entry text) for each numbered project of the Projects section."""
        text = self._section("Projects")
        out = []
        for block in re.split(r"(?m)^(?=\d+\.\s+\*\*)", text):
            m = re.match(r"\d+\.\s+\*\*(.+?)\*\*\s*(\([^)]*\))?\s*·\s*([^·\n]+?)(?:\s*·|\n)", block)
            if m:
                desc = m.group(3).strip()
                if desc.startswith("http"):
                    desc = (m.group(2) or "").strip("() ")
                out.append((m.group(1), desc, block))
        return out

    def projects_about(self, question: str) -> str | None:
        """ "Which projects deal with LLM evaluation or routing?": the projects whose entry mentions the topic words."""
        words = [
            w
            for w in re.findall(r"[a-z0-9+#.-]+", question.lower())
            if w not in _STOP_TOPIC and len(w) > 2
        ]
        specific = [w for w in words if w not in _GENERIC_TOPIC] or words
        if not specific:
            return None
        scored = []
        for name, desc, block in self._project_entries():
            low = block.lower()
            score = sum(
                bool(re.search(r"(?<![a-z0-9])" + re.escape(w[:5]), low)) for w in set(specific)
            )
            if score:
                scored.append((score, f"{name} ({desc})" if desc else name))
        if not scored:
            return None
        best = max(s for s, _ in scored)
        hits = [text for s, text in scored if s == best]
        return _fit(["Projects that match:", "; ".join(hits) + "."])

    def open_source(self) -> str | None:
        entries = self._project_entries()
        if not entries:
            return None
        return (
            f"Yes. All {len(entries)} of his projects are public on GitHub and MIT-licensed, and the "
            f"flagship, the {entries[0][0]}, is open source."
        )

    def location(self) -> str | None:
        m = re.search(r"Based in ([^.]+(?:\([^)]*\))?[^.]*)\.", self._section("About"))
        return f"He is based in {m.group(1).strip()}." if m else None

    def openness(self) -> str | None:
        m = re.search(r"Open to ([^\n]+?)\.?(?:\n|$)", self._section("About"))
        if not m:
            return None
        return f"He is open to {m.group(1).rstrip('.')}. For anything more specific, email him: {self.email}."

    # ------------------------------------------------------------------------------ routing
    def answer(self, question: str, project_named: bool) -> tuple[str, str] | None:
        """(answer text, kind) for a structured question, or None to let the normal pipeline handle it."""
        q = question.strip()
        if project_named or _GLOSSARY.search(q):
            return None
        if _LEAD.search(q) and re.search(rf"\b{_HE}\b|\bteam\b|\bpeople\b|\bleadership\b", q, re.I):
            if a := self.leadership():
                return a, "leadership"
        if _STUDENT.search(q) and (a := self.student()):
            return a, "student"
        if _CERT.search(q) and not _COURSEWORK.search(q) and (a := self.certifications(q)):
            return a, "certifications"
        m = _KNOWS.search(q) or _KNOWS2.search(q)
        if m and re.match(
            rf"\s*(does|do|did|has|have|is|can|could|what|which)\b.*\b{_HE}\b", q, re.I
        ):
            term = m.group("term")
            parts = [x for x in re.split(r"\s*(?:,|\band\b|\bor\b|/)\s*", term) if x]
            if len(parts) > 1 and (a := self.knows_many(parts)):
                return a, "skill"
            if len(parts) == 1 and not _COMPANY[0][1].search(term) and (a := self.knows(parts[0])):
                return a, "skill"
        if any(rx.search(q) for _, rx in _COMPANY) and (a := self.experience(q)):
            return a, "experience"
        if re.search(r"open[- ]?source", q, re.I) and re.search(rf"\b{_HE}\b", q, re.I):
            if a := self.open_source():
                return a, "open_source"
        if _EDU.search(q) and re.search(
            rf"\b{_HE}\b|\beducation\b|\bdegree\b|\bcoursework\b", q, re.I
        ):
            if a := self.education(q):
                return a, "education"
        if _OPEN.search(q) and re.search(rf"\b{_HE}\b", q, re.I) and (a := self.openness()):
            return a, "open"
        if (
            re.search(r"\b(19|20)\d\d\b", q)
            and re.search(rf"\b{_HE}\b", q, re.I)
            and re.search(r"\b(what|where|which|who)\b", q, re.I)
            and (a := self.in_period(q))
        ):
            return a, "period"
        if (
            re.search(r"\bprojects?\b", q, re.I)
            and re.search(r"\b(which|what|any|are there)\b", q, re.I)
            and not _PROJECTS_SHORT.search(q)
            and (a := self.projects_about(q))
        ):
            return a, "projects"
        if (
            _EXPERIENCE.search(q)
            and re.search(rf"\b{_HE}\b", q, re.I)
            and (a := self.experience(q))
        ):
            return a, "experience"
        if _NOT_SKILLS.search(q):
            return None
        if (
            _SKILLS.search(q)
            and re.search(rf"\b{_HE}\b|\bskills?\b", q, re.I)
            and (a := self.skills(q))
        ):
            return a, "skills"
        if (
            _PROJECTS.search(q)
            and (_PROJECTS_SHORT.search(q) or re.search(r"\bflagship\b", q, re.I))
            and (a := self.projects(q))
        ):
            return a, "projects"
        if _LOCATION.search(q) and (a := self.location()):
            return a, "location"
        return None
