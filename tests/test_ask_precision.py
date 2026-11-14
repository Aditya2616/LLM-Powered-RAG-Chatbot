"""Precision of the deterministic stages: a rule must not fire on a question it should not touch.

The recall tests elsewhere check that deflections, injections and profile answers happen when they should. These
check the other side: questions about the projects (taken from the ten READMEs) and skills or experience questions
that use risky words ("management", "pay off", "grade", "notes", "remote", "offline", "start", "instructions",
"prompt", "rules", "system") must never be refused, deflected as unpublished, or called off topic. A project-detail
question must end at the model stage (an empty provider chain raises ``Unavailable``, the API's 503) unless a
deterministic answer is genuinely right. On the output side, realistic model answers about the projects must never
be replaced by a deflection.

The project questions run against an index built from the private sources next to this repository (the résumé,
the corpus export and the ten READMEs) and are skipped where those files are absent (CI); the unit tests at the
end use the fictional persona and always run.
"""

from __future__ import annotations

import datetime as dt

import pytest

from askaditya import safety
from askaditya.build import build
from askaditya.index import AskIndex
from askaditya.pipeline import AskPipeline
from askaditya.profile import Profile
from askaditya.providers import Unavailable
from tests.ask_fakes import ROOT, ScriptedProvider, make_chain
from tests.test_ask_profile import SECTIONS

EMAIL = "me@example.test"
PLACEHOLDER_PLATFORM = "an AI assistant platform"  # never the real product name in tests
PROJECT_REPOS = [
    "SFT-Forge",
    "VoiceScribe",
    "VisionLens",
    "Sentinel",
    "AgentFlow",
    "DiffuseLab",
    "PromptLab",
    "EvalBench",
    "ModelRouter",
    "LLM-Powered-RAG-Chatbot",
]
SOURCES = ROOT.parent
RESUME = SOURCES.parent / "Portfolio" / "Aditya_Jain_Resume.pdf"
CORPUS = SOURCES / "_ASK_ADITYA_CORPUS.md"
HAVE_SOURCES = (
    RESUME.is_file()
    and CORPUS.is_file()
    and all((SOURCES / r / "README.md").is_file() for r in PROJECT_REPOS)
)
needs_sources = pytest.mark.skipif(
    not HAVE_SOURCES, reason="the private source documents are absent"
)


@pytest.fixture(scope="module")
def index(tmp_path_factory) -> AskIndex:
    out = tmp_path_factory.mktemp("precision-index")
    build(
        RESUME,
        CORPUS,
        [SOURCES / r for r in PROJECT_REPOS],
        out,
        platform=PLACEHOLDER_PLATFORM,
        today=dt.date(2026, 1, 2),
    )
    return AskIndex.load(out)


def run(index: AskIndex, question: str, history=None, replies=None):
    """('model', None) when the question reaches the generator (503 with an empty chain), else (stage, Result)."""
    chain = make_chain([ScriptedProvider("fake", replies)] if replies else [])
    try:
        r = AskPipeline(index, chain, EMAIL).answer(question, history)
    except Unavailable:
        return "model", None
    return r.stage, r


# ------------------------------------------------------------------------------------ project questions
PROJECT_QUESTIONS = [
    # VoiceScribe
    "What does VoiceScribe extract from meetings?",
    "How does VoiceScribe turn a recording into action items?",
    "Which speech recognition model does VoiceScribe use?",
    "What F1 did VoiceScribe reach on action items?",
    "How many held-out meetings was VoiceScribe tested on?",
    "Does VoiceScribe run offline on a CPU?",
    "How does VoiceScribe find owners and deadlines in a transcript?",
    "What are the limitations of VoiceScribe's speaker identification?",
    "Can VoiceScribe use a language model as a verifier, and does it help?",
    "What does VoiceScribe do with word-level timestamps?",
    "What are the decisions and owners that VoiceScribe pulls from a meeting?",
    # EvalBench
    "What does EvalBench measure?",
    "How does EvalBench score a model's outputs?",
    "How many prompts did EvalBench run across the checkpoints?",
    "What failure modes does EvalBench assign to wrong outputs?",
    "What pass rates did the four checkpoints reach in EvalBench?",
    "When does EvalBench use the rubric judge instead of exact match?",
    "Which checkpoints did EvalBench compare?",
    "How does EvalBench group failures on its map?",
    "What are the limitations of the EvalBench study?",
    "How does EvalBench grade an answer that has no rule?",
    # AgentFlow
    "What does AgentFlow do?",
    "Which tools can AgentFlow call?",
    "How does AgentFlow's verifier check each step?",
    "Which model does AgentFlow run offline?",
    "What is the plan, act, verify loop in AgentFlow?",
    "Where does the AgentFlow clean run fail?",
    "What does the AgentFlow send guard do?",
    "How is the AgentFlow sandbox limited?",
    "What changed with the larger model in AgentFlow?",
    # PromptLab
    "How does PromptLab search for better prompts?",
    "What is the held-out test split in PromptLab used for?",
    "Which tasks did PromptLab run on?",
    "How does the PromptLab viewer show a prompt variant?",
    "Which mutation operators does PromptLab use?",
    "How is the random-search baseline set up in PromptLab?",
    "What did PromptLab measure about children beating their parent?",
    "Does PromptLab need a hosted model?",
    # SFT Forge
    "What three gates does SFT Forge apply?",
    "How does SFT Forge remove duplicates from instruction pairs?",
    "Which data formats can SFT Forge read?",
    "What did the SFT Forge fine-tuning comparison measure?",
    "How large was the model in the SFT Forge experiment?",
    "What does the SFT Forge safety gate filter out?",
    "How does SFT Forge balance the categories?",
    "Does SFT Forge need API keys?",
    "How does SFT Forge score the quality of a response?",
    # DiffuseLab
    "What does DiffuseLab record at every denoising step?",
    "How does DiffuseLab show the noise schedule?",
    "Which samplers did DiffuseLab score with FID and KID?",
    "Which dataset are the DiffuseLab models trained on?",
    "Can DiffuseLab run without a GPU?",
    "What is the clean image estimate in DiffuseLab's step view?",
    "How does the DiffuseLab explainer let you scrub through steps?",
    "What does DiffuseLab conclude about the linear and cosine schedules?",
    # VisionLens
    "What are the three stages of VisionLens?",
    "Which detector does VisionLens use?",
    "How does the VisionLens tracker keep identities stable?",
    "How does VisionLens describe a scene in plain language?",
    "How is VisionLens tracking quality measured?",
    "What does the VisionLens rewrite guard check?",
    "Does VisionLens run offline without API keys?",
    "What are the limitations of VisionLens on real footage?",
    "What does the VisionLens scene captioner add?",
    "How does the user opt in to the VisionLens language-model rewrite?",
    # Sentinel
    "How does Sentinel screen a prompt?",
    "How many rules does Sentinel check?",
    "What share of the deepset attacks does Sentinel block?",
    "What is Sentinel's false-positive rate on hard-benign text?",
    "What presets does Sentinel have?",
    "Which classifier does Sentinel use?",
    "How can Sentinel be deployed as a service or a proxy?",
    "What was not run in the Sentinel comparison?",
    "What percentage of attacks does Sentinel miss?",
    "What does the Sentinel README say about its limitations?",
    # ModelRouter
    "How does ModelRouter decide which model gets a request?",
    "What share of the cost does ModelRouter keep at the Match big 95% preset?",
    "Which data is ModelRouter evaluated on?",
    "Does ModelRouter transfer to unseen benchmark families?",
    "What does the ModelRouter ledger record?",
    "How does ModelRouter compare with random routing on the local Qwen pair?",
    "Which classifier does ModelRouter use?",
    "What does the cost meter show in ModelRouter?",
    # RAG chatbot
    "How does the RAG chatbot chunk documents?",
    "Which vector store does the RAG chatbot use?",
    "How does hybrid retrieval work in the RAG chatbot?",
    "What share of unanswerable questions does the RAG chatbot decline?",
    "Which endpoints does the RAG chatbot's FastAPI expose?",
    "Which file types can the RAG chatbot ingest?",
    "How does the retrieval-score gate in the RAG chatbot work?",
    "Can the RAG chatbot run without an API key?",
    "How is the latency of the RAG chatbot measured?",
]

# questions that name no project but are about his work: a deterministic answer here is right
PROJECT_QUESTIONS_DETERMINISTIC_OK = {
    "which projects use whisper?": "profile",
    "which of his projects deals with speech?": "profile",
    "which project deals with diffusion?": "profile",
}


def test_there_are_at_least_eighty_project_questions():
    assert len(PROJECT_QUESTIONS) >= 80
    assert len(set(PROJECT_QUESTIONS)) == len(PROJECT_QUESTIONS)


@needs_sources
@pytest.mark.parametrize("question", PROJECT_QUESTIONS)
def test_a_project_question_ends_at_the_model_stage(index, question):
    stage, r = run(index, question)
    assert stage == "model", (question, stage, r.answer[:120] if r else "")


# ------------------------------------------------------------------- skills and experience, risky words
SKILL_QUESTIONS = [
    ("Does he have identity and access management skills?", "Entra ID"),
    ("Does he have change management skills?", "Change Management"),
    ("Does he have vendor management experience?", "Vendor Management"),
    ("Does he know platform governance?", "Platform Governance"),
    ("Which of his skills relate to access management?", "Entra ID"),
    ("Does he have experience with Conditional Access?", "Conditional Access"),
    ("Does he know identity and access management?", "Entra ID"),
    ("What are his documentation skills?", "Runbooks"),
    ("Does he have operational support skills?", "Operational Support"),
    ("Does he know SSO and SCIM?", "SSO"),
    ("What does he know about privileged roles?", None),
    ("Does he have experience writing system prompts?", None),
    ("Has he worked with prompt engineering?", None),
    ("Does he have experience writing instructions for language models?", None),
    ("Has he done supervised fine-tuning on instruction and response pairs?", None),
    ("Does he know how to write rules that detect prompt injection?", None),
    ("Does he work with remote APIs?", None),
    ("Has he built anything that works offline?", None),
    ("Did his work on routing pay off?", None),
    ("Does he know how to grade model outputs?", None),
    ("Has he taken meeting notes automatically with software?", None),
    ("Has he worked on a system that starts from a checkpoint?", None),
    ("Does he have experience with deadlines and owners in action items?", None),
    ("What did he do at Ethara AI?", "Ethara"),
    ("When did he start his internship at Ethara AI?", "Ethara"),
    ("What was his role at Genesis and when did it start?", "Genesis"),
    ("What was his start date at Genesis?", "Genesis"),
    ("Does he have experience managing enterprise applications?", None),
    ("Which skills did he use in his current role?", None),
    ("Does he know Docker and FastAPI?", "Tools & infra"),
    ("Does he have experience building a remote-first tool for teams?", None),
]


def test_there_are_at_least_thirty_skills_questions():
    assert len(SKILL_QUESTIONS) >= 30


@needs_sources
@pytest.mark.parametrize(("question", "expect"), SKILL_QUESTIONS)
def test_a_skills_or_experience_question_is_not_refused_deflected_or_off_topic(
    index, question, expect
):
    stage, r = run(index, question)
    assert stage in ("model", "profile"), (question, stage, r.answer[:120] if r else "")
    if expect:
        assert stage == "profile" and expect in r.answer, (question, r.answer[:200])


@needs_sources
@pytest.mark.parametrize(
    "question",
    [
        "Does he have identity and access management skills?",
        "Does he have change management skills?",
        "Does he have vendor management experience?",
        "Does he have experience managing enterprise applications?",
    ],
)
def test_management_skills_are_not_leadership(index, question):
    stage, r = run(index, question)
    assert not (r and "Co-Lead" in r.answer), (question, r.answer[:160])


# -------------------------------------------------------------------------- the six reported questions
@needs_sources
def test_voicescribe_extract_question_reaches_the_model(index):
    assert run(index, "What does VoiceScribe extract from meetings?")[0] == "model"


@needs_sources
def test_which_project_deals_with_a_topic_names_the_matching_project(index):
    cases = {
        "Which of his projects deals with speech?": "VoiceScribe",
        "Which project deals with diffusion?": "DiffuseLab",
        "Which of his projects works with images?": "DiffuseLab",
        "Which project is about evaluation?": "EvalBench",
        "Which project uses agents and tools?": "AgentFlow",
        "Which project is about prompt optimisation?": "PromptLab",
        "Which project prepares fine-tuning data?": "SFT Forge",
        "Which project does object detection and tracking?": "VisionLens",
        "Which project is about prompt injection safety?": "Sentinel",
        "Which project is about cost and routing?": "ModelRouter",
        "Which project is about retrieval and RAG?": "RAG Chatbot",
    }
    for question, name in cases.items():
        stage, r = run(index, question)
        assert stage == "profile" and name in r.answer, (question, stage, r.answer if r else "")
        assert "He has 10 projects" not in r.answer, question


@needs_sources
def test_a_topic_no_project_covers_goes_to_the_model_not_to_the_list(index):
    assert run(index, "Which project deals with quantum chemistry?")[0] == "model"


@needs_sources
@pytest.mark.parametrize(
    "question",
    [
        "What projects has he built?",
        "List his projects",
        "How many projects does he have?",
        "Show me all his projects",
    ],
)
def test_the_full_list_only_for_list_questions(index, question):
    stage, r = run(index, question)
    assert stage == "profile" and "He has 10 projects" in r.answer, (question, stage)


@needs_sources
def test_licence_question_is_answered_directly(index):
    stage, r = run(index, "What licence are his projects under?")
    assert stage == "profile"
    assert "MIT-licensed" in r.answer and "public on GitHub" in r.answer
    assert "LLM-Powered" not in r.answer  # not the whole list


@needs_sources
def test_platform_admin_is_answered_from_the_skills_line(index):
    stage, r = run(index, "Which AI assistant platform does he administer?")
    assert stage == "profile" and r.reason == "platform_admin"
    assert "AI platform admin" in r.answer and "SCIM" in r.answer


@needs_sources
def test_docker_and_fastapi_is_a_sentence(index):
    stage, r = run(index, "Does he know Docker and FastAPI?")
    assert stage == "profile"
    assert r.answer.startswith("Yes, both are in his skills list under Tools & infra")
    assert len(r.answer.split()) <= 100


@needs_sources
def test_chatbot_readme_assistant_sections_are_not_retrieved_for_questions_about_him(index):
    meta = [c for c in index.chunks if c.get("meta")]
    assert meta and all(c["title"].startswith("LLM-Powered-RAG-Chatbot") for c in meta)
    for q in (
        "What does he administer?",
        "Which skills does he list for platform administration?",
        "What is his AI platform experience?",
    ):
        hits = index.search(q, 7)
        assert not any(h.chunk.get("meta") for h in hits), q
    about = index.search("How does the RAG chatbot build the Ask-Aditya index?", 7)
    assert any(h.chunk.get("meta") for h in about)


# ------------------------------------------------------------------- output side: realistic model answers
MODEL_ANSWERS = [
    (
        "What does VoiceScribe extract from meetings?",
        "VoiceScribe turns a meeting recording into a transcript with word-level timestamps and pulls out decisions, "
        "owners and deadlines as structured, validated action items. Ask me about its speech recognition.",
    ),
    (
        "How does VoiceScribe score?",
        "On 9 held-out meetings its default rules reach an action-item F1 of 0.89. Ask me about its owners.",
    ),
    (
        "Does VoiceScribe run offline?",
        "Yes, VoiceScribe runs offline on a CPU, using Whisper (faster-whisper) for speech recognition. Ask me about its limitations.",
    ),
    (
        "How does EvalBench score outputs?",
        "EvalBench uses exact match and rule checks first, and a small rubric judge only where no rule exists. Ask me about its failure modes.",
    ),
    (
        "How many prompts did EvalBench run?",
        "In the recorded run, four sub-1B checkpoints pass between 27.0% and 44.6% of 408 prompts. Ask me about its failure map.",
    ),
    (
        "What failure modes does EvalBench find?",
        "It assigns each failure a mode with deterministic detectors: hallucination, format break, over-refusal, wrong answer or degenerate output. Ask me about the checkpoints.",
    ),
    (
        "What does AgentFlow do?",
        "AgentFlow plans a multi-step task, calls tools and checks each result before it moves on: plan, act, verify, built as a LangGraph state graph. Ask me about its tools.",
    ),
    (
        "Which tools does AgentFlow have?",
        "It has five tools: search, Python, SQL, email and documents, with a JSONL trace of every call. Ask me about the verifier.",
    ),
    (
        "How does AgentFlow run offline?",
        "It runs fully offline with a small open model (Qwen3-0.6B), on a CPU or a GPU; OpenAI and Anthropic adapters exist behind the same interface. Ask me about its sandbox.",
    ),
    (
        "How does PromptLab work?",
        "PromptLab keeps a population of prompt variants, mutates them, scores each on a dev split and keeps the strongest for the next generation. Ask me about the held-out split.",
    ),
    (
        "How does PromptLab pay off against a baseline?",
        "The winners are scored once on a held-out test split, next to a random-search baseline given the same budget. Ask me about its viewer.",
    ),
    (
        "What are SFT Forge's gates?",
        "SFT Forge passes pairs through three gates: dedupe, quality and safety, then balances the survivors across categories. Ask me about the experiment.",
    ),
    (
        "Does SFT Forge need keys?",
        "It runs fully offline on CPU after a one-time model download and needs no API keys. Ask me about its design decisions.",
    ),
    (
        "What did the SFT Forge comparison find?",
        "On the one downstream test, a 135M model with three seeds, cleaning made no practical difference. Ask me about the gates.",
    ),
    (
        "What does DiffuseLab record?",
        "At every denoising step it records the noisy sample, the noise the network predicts, the clean image it would guess and the schedule quantities. Ask me about the schedules.",
    ),
    (
        "What did DiffuseLab measure?",
        "Of the 3 samplers scored with FID and KID, the linear schedule is clearly better under 1, the cosine schedule under 0, and the other 2 are not separated. Ask me about the models.",
    ),
    (
        "How does VisionLens work?",
        "A detection model finds objects frame by frame, a tracker he wrote keeps their identities stable, and the track history becomes a plain-language description. Ask me about tracking.",
    ),
    (
        "Does VisionLens need keys?",
        "It runs offline, on a CPU or a GPU, with no API keys. Ask me about its scene captioner.",
    ),
    (
        "How good is VisionLens tracking?",
        "Tracking is measured on synthetic scenes with exact ground truth only; there is no accuracy number for real footage. Ask me about its limitations.",
    ),
    (
        "How does Sentinel screen prompts?",
        "Every prompt is normalised, checked against 46 rules for known injection patterns and scored by a small fine-tuned classifier; suspicious prompts are blocked and logged. Ask me about its presets.",
    ),
    (
        "What does Sentinel block?",
        "On two datasets it was never trained on, it blocks 54% of the deepset attacks and 54% of the SPML attacks. It is one layer, not a complete one. Ask me about its limitations.",
    ),
    (
        "How does Sentinel run?",
        "It runs as a Python library, a CLI, a FastAPI service and a proxy. Ask me about its rules.",
    ),
    (
        "How does ModelRouter route?",
        "A small scikit-learn classifier predicts, before any model runs, whether the cheap model will be good enough; an OpenAI-compatible gateway routes each request and keeps a ledger of what it saved. Ask me about its presets.",
    ),
    (
        "What does ModelRouter measure?",
        "On RouterBench the router keeps 95% of the big model's quality at 37% of the cost, from an offline analysis of recorded outcomes. Ask me about its limits.",
    ),
    (
        "How does the RAG chatbot answer?",
        "It ingests PDF, TXT and DOCX, chunks them, embeds them, stores them in a FAISS index and answers with citations through FastAPI. Ask me about its retrieval.",
    ),
    (
        "How well does the RAG chatbot decline?",
        "On the bundled 100-question test set, a 1.5B open model answers 93% of answerable questions correctly and declines 72% of the unanswerable ones. Ask me about the gate.",
    ),
    (
        "How is the chatbot cached?",
        "It has query logging, in-memory LRU caching and a Dockerfile. Ask me about its endpoints.",
    ),
    (
        "What does VoiceScribe do with notes?",
        "It writes the extracted decisions as structured notes with owners and deadlines for each action item. Ask me about its transcripts.",
    ),
    (
        "How is VoiceScribe checked?",
        "Its rules were frozen before the held-out meetings were written, and the scores on them are reported next to the dev set. Ask me about its owners.",
    ),
    (
        "How does EvalBench grade?",
        "It grades outputs with rule checks first and a rubric judge only where no rule exists. Ask me about its pass rates.",
    ),
    (
        "How does Sentinel use remote services?",
        "It can run as a service behind a proxy, and its remote API takes a prompt and returns a decision. Ask me about its deployment.",
    ),
    (
        "Does SFT Forge help models?",
        "Its own experiment found no measurable benefit from the cleaned data at that scale, so the gates are measured against injected defects only. Ask me about the gates.",
    ),
    (
        "What are VisionLens limits?",
        "The tracking results come from synthetic scenes only, and the optional rewrite is checked by a faithfulness guard. Ask me about the captioner.",
    ),
    (
        "How are ModelRouter results scored?",
        "It does no better than random routing on a local Qwen pair, and it does not transfer to unseen benchmark families. Ask me about RouterBench.",
    ),
    (
        "Does AgentFlow have a sandbox?",
        "It has a sandbox for its Python tool, and the README says what it does and does not protect. Ask me about its design decisions.",
    ),
    (
        "How does the RAG chatbot scale?",
        "It runs with no API key on a local open model, with an optional Streamlit UI and an animated web viewer. Ask me about its API.",
    ),
    (
        "What does PromptLab's viewer do?",
        "Each column is a generation and each node a prompt variant; survivors glow and culled variants fade. Ask me about the held-out data.",
    ),
    (
        "What is the DiffuseLab explainer?",
        "It is an instrumented diffusion sampler plus an explainer UI that runs real models, so you can scrub the denoising steps. Ask me about the metrics.",
    ),
    (
        "How does Sentinel treat benign text?",
        "Its false-positive rate on hard-benign text is 1%, and clean prompts pass through untouched. Ask me about the presets.",
    ),
    (
        "What are the VoiceScribe deadlines?",
        "Deadlines such as a date or day are extracted next to the owner and the decision, and a wrong name counts against a method more than an unassigned item. Ask me about its F1.",
    ),
]


def test_there_are_forty_model_answers():
    assert len(MODEL_ANSWERS) >= 40


@pytest.mark.parametrize(("question", "answer"), MODEL_ANSWERS)
def test_no_project_answer_asserts_unpublished_personal_information(question, answer):
    assert safety.unpublished_in_answer(answer) is None, answer


@needs_sources
def test_no_project_answer_is_replaced_by_a_deflection(index):
    replaced = []
    stages = {}
    for question, answer in MODEL_ANSWERS:
        stage, r = run(index, question, replies=[answer])
        stages[stage] = stages.get(stage, 0) + 1
        if stage in ("topic", "refused", "offtopic"):
            replaced.append((question, stage, r.reason))
    assert replaced == []
    assert stages.get("model", 0) >= 30, (
        stages
    )  # the rest ended in "I don't know", never a deflection


@pytest.mark.parametrize(
    "answer",
    [
        "His CGPA is 8.9.",
        "He is open to remote work and can relocate.",
        "His salary expectation is 12 LPA.",
        "He needs a work visa for the US.",
        "His notice period is 30 days.",
        "He is willing to relocate to Bangalore.",
        "His grades are excellent.",
        "He can join immediately.",
    ],
)
def test_an_answer_that_does_assert_personal_information_is_still_replaced(answer):
    assert safety.unpublished_in_answer(answer) is not None


# --------------------------------------------------------------------------- question side, no sources needed
@pytest.mark.parametrize(
    "question",
    [
        "What does VoiceScribe extract from meetings?",
        "What percentage of attacks does Sentinel miss?",
        "Which model family does EvalBench compare?",
        "Does the VisionLens README say it is available on PyPI?",
        "Which arXiv paper does PromptLab follow?",
        "How does VoiceScribe handle phone recordings?",
        "Can the retrieval run on a phone?",
        "What marks does the rubric use in EvalBench?",
        "How does SFT Forge grade responses?",
        "Does the RAG chatbot support remote vector stores?",
        "How do I opt out of the optional rewrite in VisionLens?",
        "How old is the dataset DiffuseLab trains on?",
        "What transcripts does VoiceScribe produce?",
        "Did the work on ModelRouter pay off?",
        "Is the ModelRouter gateway available on GitHub?",
    ],
)
def test_project_questions_are_not_unpublished_topics(question):
    assert safety.unpublished_topic(question) is None


@pytest.mark.parametrize(
    "question",
    [
        "What is his CGPA?",
        "What percentage did he score in college?",
        "Does he have a visa?",
        "Is he available from next month?",
        "Can he work remotely?",
        "What are his grades?",
        "Where is his research paper published?",
        "Does he have a family?",
        "What is his phone number?",
    ],
)
def test_personal_questions_are_still_unpublished_topics(question):
    assert safety.unpublished_topic(question) is not None


@pytest.mark.parametrize(
    "question",
    [
        "Does he have identity and access management skills?",
        "Does he have change management skills?",
        "Does he have vendor management experience?",
        "Does he have experience managing enterprise applications?",
        "Does he know platform governance?",
    ],
)
def test_management_wording_is_not_leadership(question):
    sections = [*SECTIONS]
    sections[6] = {
        "section": "Skills (as listed on the site)",
        "text": (
            "- Languages: Python, SQL\n"
            "- Identity & access: Microsoft Entra ID, SSO, SAML, SCIM, Conditional Access\n"
            "- Platform operations: Platform Governance, Change Management, Vendor Management"
        ),
    }
    prof = Profile(sections, [], EMAIL)
    found = prof.answer(question, False)
    assert found is None or found[1] != "leadership", (question, found)


@pytest.mark.parametrize(
    "question",
    [
        "Has he led a team before?",
        "Does he have management experience?",
        "Has Aditya managed people?",
        "Does he have team management experience?",
        "Did he ever mentor anyone?",
    ],
)
def test_people_management_wording_is_leadership(question):
    found = Profile(SECTIONS, [], EMAIL).answer(question, False)
    assert found and found[1] == "leadership"


def test_the_project_list_licence_and_topic_questions_on_the_persona():
    prof = Profile(SECTIONS, [], EMAIL)
    assert prof.answer("What licence are his projects under?", False)[1] == "licence"
    assert "2 projects" in prof.answer("How many projects has he built?", False)[0]
    found = prof.answer("Which of his projects deals with evaluation?", False)
    assert found and "EvalBench" in found[0] and "2 projects" not in found[0]
    assert prof.answer("Which of his projects deals with quantum chemistry?", False) is None


def test_docker_and_fastapi_sentence_and_the_unlisted_case_on_the_persona():
    prof = Profile(SECTIONS, [], EMAIL)
    assert prof.answer("Does he know Docker and FastAPI?", False)[0] == (
        "Yes, both are in his skills list under Tools & infra: Docker and FastAPI."
    )
    mixed = prof.answer("Does he know Docker and Kubernetes?", False)[0]
    assert mixed.startswith("Listed: Docker (under Tools & infra).")
    assert "Kubernetes is not listed" in mixed


def test_the_assistant_sections_of_the_chatbot_readme_are_marked_and_left_out(tmp_path):
    from tests.ask_fakes import PERSONA, make_resume_pdf

    proj = tmp_path / "projects" / "LLM-Powered-RAG-Chatbot"
    proj.mkdir(parents=True)
    (proj / "README.md").write_text(
        "# Chatbot\n\nA retrieval chatbot with hybrid search.\n\n"
        "## Ask-Aditya: the portfolio assistant\n\n"
        "The assistant says an AI assistant platform for a local project and rebuilds the index from a private corpus.\n",
        encoding="utf-8",
    )
    pdf = tmp_path / "resume.pdf"
    make_resume_pdf(pdf)
    build(
        pdf,
        PERSONA / "corpus.md",
        [tmp_path / "projects"],
        tmp_path / "idx",
        today=dt.date(2026, 1, 2),
    )
    ix = AskIndex.load(tmp_path / "idx")
    meta = [c for c in ix.chunks if c.get("meta")]
    assert meta and all("Ask-Aditya" in c["section"] for c in meta)
    assert not any(h.chunk.get("meta") for h in ix.search("Which platform does he administer?", 5))
    assert any(
        h.chunk.get("meta") for h in ix.search("How does the RAG chatbot build its index?", 5)
    )
