# ==============================================================================
# tests/test_prompt_template.py
# ------------------------------------------------------------------------------
# Unit tests for prompts/prompt_template.py.
#
# WHY TEST PROMPT BUILDING:
#   The exact wording sent to the LLM directly determines answer quality
#   and groundedness. These tests don't call OpenAI (no API key needed) —
#   they just verify the prompt STRING is assembled correctly, which is
#   fast, free, and catches regressions like "oops, I forgot to include the
#   page number in the context block."
#
# HOW TO RUN:
#       pytest tests/test_prompt_template.py -v
# ==============================================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.prompt_template import build_context_block, build_user_prompt, SYSTEM_PROMPT
from retrieval.retriever import RetrievedChunk


def _sample_chunks():
    return [
        RetrievedChunk(
            text="The annual deductible is $1,500 per individual.",
            source_file="Aetna_EOC_2026.pdf",
            page_number=14,
            similarity_score=0.12,
        ),
        RetrievedChunk(
            text="Emergency room visits require a $250 copay.",
            source_file="Aetna_SBC_2026.pdf",
            page_number=3,
            similarity_score=0.18,
        ),
    ]


def test_build_context_block_includes_source_and_page():
    block = build_context_block(_sample_chunks())
    assert "Aetna_EOC_2026.pdf" in block
    assert "page 14" in block
    assert "Aetna_SBC_2026.pdf" in block
    assert "page 3" in block
    assert "$1,500" in block


def test_build_context_block_handles_empty_list():
    block = build_context_block([])
    assert "No relevant context" in block


def test_build_user_prompt_includes_question_and_context():
    question = "What is my deductible?"
    prompt = build_user_prompt(question, _sample_chunks())
    assert question in prompt
    assert "$1,500" in prompt
    assert "cite" in prompt.lower()


def test_system_prompt_enforces_grounding_rules():
    # Sanity-check that the key safety/grounding rules are present so an
    # accidental edit doesn't silently remove them.
    assert "ONLY" in SYSTEM_PROMPT
    assert "cite" in SYSTEM_PROMPT.lower()
    assert "do not" in SYSTEM_PROMPT.lower() or "don't" in SYSTEM_PROMPT.lower()
