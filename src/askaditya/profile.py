"""Answers to the structured questions about him, from the portfolio's own sections, with no model.

Certifications, education, the roles he has held, leadership, skills (and "does he know X"), the list of projects,
where he is based and whether he is open to roles all have one right answer in the corpus, and a recruiter asks
them first. A language model sometimes declines them or drops a part, so they are answered here, from the full
sections kept in the index (``profile``), and never reach a provider.
"""

from __future__ import annotations

import re

MAX_WORDS = 100

_GLOSSARY = re.compile(
    r"\b(mean|means|meaning|stand for|stands for|define|definition|explain|difference|why|how (does|did|do))\b",
    re.I,
)
_HE = r"(he|aditya|his|him)"

_LEAD = re.compile(
    r"\b(led|lead|leads|leader|leadership|manage[sd]?|managing|mentor\w*|supervis\w*|team lead|co-?lead)\b",
    re.I,
)
_CERT = re.compile(r"\b(certif\w*|credentials?|courses? (has|did|have)|licen[cs]es)\b", re.I)
_EDU = re.compile(
    r"\b(educat\w*|degree|universit\w*|college|studies|studied|study|graduat\w*|coursework|b\.?tech|major|school|gguipu|ggsipu)\b",
    re.I,
)
_COMPANY = [
    ("Ethara", re.compile(r"\bethara\b", re.I)),
    ("Genesis", re.compile(r"\bgenesis\b", re.I)),
    ("15FORTEEN", re.compile(r"\b15 ?forteen\b|\bfounder'?s office\b", re.I)),
]
_EXPERIENCE = re.compile(
    r"\b(experience|work history|career|employment|employer|worked|work (now|currently|right now)|currently|right now|current (role|job|position)|internship|intern|where does (he|aditya) work|what does (he|aditya) do at)\b",
    re.I,
)
_SKILLS = re.compile(
    r"\b(skills?|tech stack|technolog\w*|tools?|languages?|frameworks?|librar(y|ies)|proficien\w*|expertise|good at|strengths?|infra\w*|stack)\b",
    re.I,
)
_KNOWS = re.compile(
    r"\b(know|knows|use|uses|using|used|work(ed|s)? (with|on)|experience (with|in|of)|familiar (with)?|proficient (in|with)|skilled (in|at|with)|knowledge of|expertise in|comfortable with|handled?)\s+(?:the |any |of |in |with )?(?P<term>[\w .+#&/-]{2,40}?)\s*[?.!]*$",
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
    r"\b(open to|looking for|available|opportunit\w*|hir(e|ing)|job search|seeking|on the market)\b",
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
        want = [
            w for w in ("deeplearning", "linkedin", "coursera", "deloitte") if w in question.lower()
        ]
        picked = [i for i in items if any(w in i.lower() for w in want)] or items
        head = (
            f"He lists {len(items)} certifications:"
            if picked is items
            else f"{len(picked)} of his {len(items)} certifications match:"
        )
        body = (
            "; ".join(f"{i + 1}. {c}" for i, c in enumerate(picked))
            if picked is items
            else "; ".join(picked)
        )
        note = (
            " His résumé lists five of the seven; the portfolio lists all of them."
            if re.search(r"r[ée]sum[ée]( PDF)? lists five", text, re.I)
            else ""
        )
        return _fit([head, body + ".", note.strip()] if note else [head, body + "."])

    def education(self) -> str | None:
        items = _bullets(self._section("Education"))
        if not items:
            return None
        keep = [i for i in items if not i.lower().startswith("no cgpa")]
        return _fit([i.rstrip(".") + "." for i in keep])

    def experience(self, question: str) -> str | None:
        roles = self._roles()
        if not roles:
            return None
        named = [n for n, rx in _COMPANY if rx.search(question)]
        if not named and re.search(r"\b(currently|right now|now|current)\b", question, re.I):
            named = ["Ethara"]
        picked = [
            (h, b) for h, b in roles if not named or any(n.lower() in h.lower() for n in named)
        ]
        if not picked:
            return None
        if len(picked) == 1:
            h, b = picked[0]
            return _fit(
                [
                    self._role_line(h) + ".",
                    *[x.rstrip(".") + "." for x in b if "no further detail" not in x.lower()],
                ]
            )
        parts = []
        for h, b in picked:
            first = next((x for x in b if "no further detail" not in x.lower()), "")
            parts.append(self._role_line(h) + (f": {first.rstrip('.')}" if first else "") + ".")
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

    def knows(self, term: str) -> str | None:
        """Yes/no for "does he know X": from the skills list, then the project stacks and READMEs."""
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
        for cat, items in self._skill_lines():
            if re.search(rf"(?<![\w-]){re.escape(t)}(?![\w])", items, re.I):
                return f"Yes. {term.strip(' .?!')} is in his skills list, under {cat}: {items}."
        stacks = [
            f"{m.group(1)}"
            for m in re.finditer(
                r"\*\*([^*]+)\*\*[^\n]*·[^\n]*·\s*([^\n]*)", self._section("Projects")
            )
            if re.search(rf"(?<![\w-]){re.escape(t)}(?![\w])", m.group(2), re.I)
        ]
        if stacks:
            return f"Yes. {term.strip(' .?!')} appears in the stack of {', '.join(stacks[:4])}."
        titles = []
        for c in self.chunks:
            if c["title"] not in ("Portfolio", "Résumé") and re.search(
                rf"(?<![\w-]){re.escape(t)}(?![\w])", c["text"], re.I
            ):
                name = c["title"].replace(" README", "")
                if name not in titles:
                    titles.append(name)
        if titles:
            return f"{term.strip(' .?!')} is mentioned in the README of {', '.join(titles[:4])}, though it is not in his skills list."
        return f"{term.strip(' .?!')} is not listed among his skills, project stacks or READMEs."

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
        if _CERT.search(q) and (a := self.certifications(q)):
            return a, "certifications"
        m = _KNOWS.search(q)
        if m and re.match(
            rf"\s*(does|do|did|has|have|is|can|could|what|which)\b.*\b{_HE}\b", q, re.I
        ):
            term = m.group("term")
            parts = [x for x in re.split(r"\s*(?:,|\band\b|\bor\b|/)\s*", term) if x]
            found = [self.knows(x) for x in parts]
            if parts and all(f and f.startswith("Yes") for f in found):
                said = " and ".join(x.strip(" .?!") for x in parts)
                return f"Yes. {said} are in his skills list or project stacks.", "skill"
            if len(parts) == 1 and found[0] and not _COMPANY[0][1].search(term):
                return found[0], "skill"
        if any(rx.search(q) for _, rx in _COMPANY) and (a := self.experience(q)):
            return a, "experience"
        if _EDU.search(q) and re.search(
            rf"\b{_HE}\b|\beducation\b|\bdegree\b|\bcoursework\b", q, re.I
        ):
            if a := self.education():
                return a, "education"
        if _OPEN.search(q) and re.search(rf"\b{_HE}\b", q, re.I) and (a := self.openness()):
            return a, "open"
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
