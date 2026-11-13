import pytest

from askaditya.build import table_sections
from askaditya.index import AskIndex
from askaditya.pipeline import AskPipeline
from askaditya.profile import MAX_WORDS, Profile
from tests.ask_fakes import ScriptedProvider, make_chain

EMAIL = "me@example.test"

SECTIONS = [
    {
        "section": "About",
        "text": (
            "- Aditya Jain. Computer Science undergraduate at Some University, City, 2023 to 2027. "
            "Based in New Delhi, India (IST, UTC+5:30).\n"
            "- Works mainly on generative AI.\n"
            "- Now: LLM Post-Training Intern at Ethara AI (Gurugram), Feb 2026 to present: SFT.\n"
            "- Open to AI / ML roles and collaborations."
        ),
    },
    {
        "section": "Experience > Ethara AI · LLM Post-Training Intern · Feb 2026 to Present · Gurugram · current",
        "text": "- Contribute to post-training pipelines including SFT and RLHF.\n- Design prompt strategies.",
    },
    {
        "section": "Experience > Genesis · Co-Lead · Oct 2025 to Feb 2026 · New Delhi",
        "text": "- Led a team of developers working on AI-driven projects.\n- Mentored junior members.",
    },
    {
        "section": "Experience > 15FORTEEN · Founder's Office · Jun 2025 to Feb 2026 · Delhi",
        "text": "- Founder's Office role at 15FORTEEN, Delhi. (No further detail is published.)",
    },
    {
        "section": "Education",
        "text": (
            "- B.Tech, Computer Science Engineering, GGSIPU, New Delhi, Aug 2023 to Jun 2027 (in progress).\n"
            "- Coursework: DSA, Machine Learning.\n"
            "- No CGPA or grades are published."
        ),
    },
    {
        "section": "Certifications (seven)",
        "text": (
            "1. Machine Learning Specialization, DeepLearning.AI / Coursera\n"
            "2. Prompt Engineering for Developers, DeepLearning.AI\n"
            "3. AI-First Product Leader, LinkedIn Learning\n"
            "4. GenAI for Marketers, LinkedIn Learning\n"
            "5. GenAI Recruiting, LinkedIn Learning\n"
            "6. Intro to AI, LinkedIn Learning\n"
            "7. Technology Job Simulation, Deloitte Australia\n"
            "(The résumé PDF lists five of these; the site lists all seven.)"
        ),
    },
    {
        "section": "Skills (as listed on the site)",
        "text": (
            "- Languages: Python, SQL\n- AI / ML: Machine Learning, Deep Learning\n"
            "- ML & data: PyTorch, FAISS, NumPy\n- Tools & infra: FastAPI, Docker, Git"
        ),
    },
    {
        "section": "Projects (ten; all public on GitHub, MIT-licensed)",
        "text": (
            "1. **RAG Chatbot** (flagship, open source) · https://github.com/x/RAG\n   Q&A.\n"
            "2. **EvalBench** · LLM evaluation harness · https://github.com/x/EvalBench · Python, Kotlin\n"
            "   Runs prompts."
        ),
    },
]
profile = Profile(SECTIONS, [], EMAIL)


def ask(q, project_named=False):
    r = profile.answer(q, project_named)
    return r[0] if r else None


@pytest.mark.parametrize(
    "q",
    [
        "What certifications does he have?",
        "Which certificates does Aditya hold?",
        "List his certifications",
        "Does he have any credentials?",
    ],
)
def test_certifications_list_all_seven_and_note_the_resume(q):
    a = ask(q)
    assert a and "7 certifications" in a and "Deloitte" in a and "five" in a
    assert len(a.split()) <= MAX_WORDS


def test_certifications_can_be_filtered_by_issuer():
    a = ask("Which certifications does he have from DeepLearning.AI?")
    assert "Machine Learning Specialization" in a and "Deloitte" not in a


@pytest.mark.parametrize(
    "q",
    [
        "Has he led a team before?",
        "Does he have leadership experience?",
        "Has Aditya managed people?",
        "Did he ever mentor anyone?",
    ],
)
def test_leadership_questions_get_the_co_lead_role(q):
    a = ask(q)
    assert a and "Co-Lead at Genesis" in a and "team of developers" in a


@pytest.mark.parametrize(
    ("q", "expect"),
    [
        ("What does he study?", "GGSIPU"),
        ("Which university is he at?", "GGSIPU"),
        ("When does he graduate?", "Jun 2027"),
        ("What is his education?", "Computer Science"),
        ("What is his work experience?", "Genesis"),
        ("Where does Aditya work right now?", "Ethara AI"),
        ("What did he do at Genesis?", "Led a team"),
        ("What role did he hold at 15FORTEEN and when?", "Jun 2025"),
        ("Which programming languages does he list?", "Python, SQL"),
        ("Which tools and infrastructure does he use?", "Docker"),
        ("Which deep learning library does he list?", "PyTorch"),
        ("Where is he based?", "New Delhi"),
        ("Is he open to new roles?", "open to AI / ML roles"),
        ("How many projects has he built?", "2 projects"),
        ("What is his flagship project?", "RAG Chatbot"),
    ],
)
def test_structured_questions_are_answered_from_the_sections(q, expect):
    a = ask(q)
    assert a and expect in a and len(a.split()) <= MAX_WORDS


@pytest.mark.parametrize(
    ("q", "start"),
    [
        ("Does he know SQL?", "Yes"),
        ("Has he worked with PyTorch?", "Yes"),
        ("Does he use Docker?", "Yes"),
        ("Does he know Python and SQL?", "Yes"),
        ("Does he know Kotlin?", "Yes"),  # only in a project stack
        ("Does he know Kubernetes?", "Kubernetes is not listed"),
    ],
)
def test_does_he_know_x_is_answered_from_the_lists_without_guessing(q, start):
    a = ask(q)
    assert a and a.startswith(start)


@pytest.mark.parametrize(
    "q",
    [
        "What does RLHF mean in his work at Ethara?",  # a glossary question goes to the passages
        "How does he lead his team?",
        "What is the weather like?",
        "Does he speak Hindi?",
    ],
)
def test_other_questions_are_left_to_the_normal_pipeline(q):
    assert ask(q) is None


def test_a_named_project_is_not_a_profile_question():
    assert ask("Which tools can EvalBench call?", project_named=True) is None


def test_the_stage_never_calls_a_provider():
    chunks = [{"id": 0, "title": "Portfolio", "url": "u", "section": "About", "text": "x"}]
    ix = AskIndex(chunks, "t", None, None, SECTIONS)
    b = ScriptedProvider("groq", ["UNKNOWN"])
    r = AskPipeline(ix, make_chain([b]), EMAIL).answer("Has he led a team before?")
    assert r.stage == "profile" and not b.calls and r.sources and r.grounded


# ------------------------------------------------------------------------------ whole tables
def test_a_short_table_becomes_one_compact_chunk_with_its_columns():
    md = "# P\n\n## Results\n\n| Model | Pass |\n|---|---|\n| a | 10% |\n| b | 40% |\n"
    ((heading, text),) = table_sections(md, "P")
    assert heading == "P > Results" and "Columns: Model | Pass" in text and "b | 40%" in text


def test_a_long_table_is_not_kept_whole():
    rows = "\n".join(f"| row{i} | {'x' * 120} |" for i in range(30))
    assert table_sections(f"## T\n\n| A | B |\n|---|---|\n{rows}\n", "P") == []


def test_a_whole_table_competes_only_for_superlative_questions_about_its_project():
    chunks = [
        {
            "id": 0,
            "title": "EvalBench README",
            "url": "u",
            "section": "EvalBench",
            "text": "EvalBench compares four models.",
        },
        {
            "id": 1,
            "title": "EvalBench README",
            "url": "u",
            "section": "EvalBench > Results",
            "text": "EvalBench results are below.",
        },
        {
            "id": 2,
            "title": "EvalBench README",
            "url": "u",
            "section": "EvalBench > Results",
            "kind": "table",
            "text": "Table under Results. Columns: model | pass\nalpha | 44.6%\nbeta | 27.0%",
        },
    ]
    ix = AskIndex(chunks, "t")
    best = ix.search("Which model scored best in EvalBench?", 3)
    assert any(h.chunk.get("kind") == "table" for h in best)
    plain = ix.search("What does EvalBench compare?", 3)
    assert all(h.chunk.get("kind") != "table" for h in plain)


# ------------------------------------------------------------------ composite table cells
def test_composite_cells_become_named_values():
    from askaditya.build import expand_cell

    assert (
        expand_cell("Action P / R / F1", "0.93 / 0.84 / 0.89")
        == "Action precision 0.93; Action recall 0.84; Action F1 0.89"
    )
    assert (
        expand_cell("Owners ok / wrong / none", "17 / 0 / 9")
        == "Owners ok 17; Owners wrong 0; Owners none 9"
    )
    assert (
        expand_cell("CPT 90% / 95% / 99%", "24% / 42% / 76%")
        == "CPT 90% 24%; CPT 95% 42%; CPT 99% 76%"
    )
    assert (
        expand_cell("First token mean / p95", "480 ms / 770 ms")
        == "First token mean 480 ms; First token p95 770 ms"
    )


def test_cells_whose_parts_do_not_match_are_left_alone():
    from askaditya.build import expand_cell, tables_to_rows

    assert expand_cell("Action P / R / F1", "0.93 / 0.84") is None
    assert expand_cell("Rate", "1 / 2") is None
    assert expand_cell("A / long header here", "1 / 2") is None
    md = "| Set | A / B / C |\n|---|---|\n| x | 1 / 2 |\n"
    assert tables_to_rows(md).strip() == "- Set: x; A / B / C: 1 / 2"


def test_a_composite_row_is_written_with_names_and_the_default_row_is_marked():
    from askaditya.build import tables_to_rows

    md = (
        "| Config | Held-out | Action P / R / F1 |\n|---|---|---|\n"
        "| rules (default) | 9 meetings | 0.93 / 0.84 / 0.89 |\n"
        "| rules + verifier | 9 meetings | 0.90 / 0.80 / 0.85 |\n"
    )
    out = tables_to_rows(md).split("\n")
    assert out[0].startswith("- (default / headline row) Config: rules (default)")
    assert "Action P / R / F1: Action precision 0.93; Action recall 0.84; Action F1 0.89" in out[0]
    assert not out[1].startswith("- (default") and "Action F1 0.85" in out[1]
