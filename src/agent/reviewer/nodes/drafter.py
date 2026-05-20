import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.reviewer.state import AgentState
from src.core.llm import get_llm


class DrafterOutput(BaseModel):
    review_text: str = Field(description="The simulated review in the user's voice")


_llm = get_llm("gemini").with_structured_output(DrafterOutput)

# Patterns that carry stylistic voice without content-specific nouns:
# - interjections / openers: "So far,", "I gotta say,"
# - adverb+adjective: "absolutely worth it", "surprisingly good"
# - first-person + strong verb/adj (2-4 words): "I loved it", "I'll never"
# - exclamation tails: "must try!", "go in!"
_STYLISTIC_FRAGMENT = re.compile(
    r"""
    (?:
        \b(?:wow|honestly|seriously|surprisingly|absolutely|definitely|
           clearly|genuinely|truly|unfortunately|thankfully)\b\s+\w+  # adverb+word
      | \bI(?:'ll|'ve|'d|'m)?\s+(?:love|hate|never|always|highly|totally|
            definitely|honestly|really|still)\b[^.!?,]{0,20}          # I + strong verb start
      | \b(?:must|can't|cannot|won't|don't)\s+\w+(?:\s+\w+)?!?        # modal + verb (short)
      | \b\w+(?:\s+\w+){1,3}[!]                                        # 2-4 words ending in !
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _extract_anchors(reviews: list[dict], max_anchors: int = 5) -> list[str]:
    candidates: list[tuple[int, str]] = []
    for review in reviews:
        for match in _STYLISTIC_FRAGMENT.finditer(review["text"]):
            fragment = match.group().strip().rstrip(",")
            words = fragment.split()
            if not (2 <= len(words) <= 5):
                continue
            score = 0
            if fragment.endswith("!"):
                score += 2
            if fragment.lower().startswith("i "):
                score += 1
            if any(
                w.lower() in ("never", "always", "must", "love", "hate", "absolutely")
                for w in words
            ):
                score += 1
            candidates.append((score, fragment))

    candidates.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    anchors: list[str] = []
    for _, fragment in candidates:
        normalised = fragment.lower()
        if normalised not in seen:
            seen.add(normalised)
            anchors.append(fragment)
        if len(anchors) == max_anchors:
            break
    return anchors


def _avg_word_count(reviews: list[dict]) -> int:
    if not reviews:
        return 80
    counts = [len(r["text"].split()) for r in reviews]
    return round(sum(counts) / len(counts))


def _build_system_prompt(
    avg_words: int, anchors: list[str], predicted_rating: float, categories: str
) -> str:
    anchor_rule = (
        (
            "2. Anchor phrases: Naturally weave in 1–3 of these phrases extracted from "
            "their past writing (do not force all of them):\n"
            + "\n".join(f'   - "{a}"' for a in anchors)
        )
        if anchors
        else ("2. No anchor phrases available — rely entirely on the manifesto tone.")
    )

    return (
        "You are an Elite Ghostwriter for a Stateful Persona Agent. Your goal is to write "
        "a simulated review for a target business that is indistinguishable from the user's "
        "historical voice.\n\n"
        "Rules:\n"
        f"1. Match length: Write approximately {avg_words} words — match the user's typical review length.\n"
        f"{anchor_rule}\n"
        "3. Match tone: Mirror the vocabulary, sentence rhythm, and punctuation style in the Writing Samples. "
        "Use the samples for style ONLY — do not reference, copy, or echo any specific business names, "
        "menu items, dishes, or factual details from the Writing Samples into the new review.\n"
        f"4. Justify the rating: Use the Reasoning Trace to explain the 'why' behind {predicted_rating} stars "
        "— but express it in the user's voice, not an analyst's voice.\n"
        "5. Be specific: Mention concrete attributes of the target business (e.g. WiFi, parking, price range). "
        "No generic filler like 'This place was great.' If you cannot be specific, be vague in the user's "
        "natural voice — never use placeholder text like '[mention dish here]' or bracketed instructions.\n"
        "6. Do not break character. Write as the user, not about the user.\n"
        f"7. Stay on category: The target business belongs to '{categories}'. Every sentence must "
        "be consistent with that category. Do not introduce references to unrelated cuisines, "
        "venue types, or activities inferred from the user's past reviews."
    )


def _format_samples(state: AgentState) -> str:
    if state["new_experience"] or not state["retrieved_reviews"]:
        return "No writing samples available — rely on manifesto tone only."
    lines = []
    for r in state["retrieved_reviews"]:
        lines.append(f"[{r['stars']}/5 — {r['biz_name']}]\n{r['text']}")
        lines.append("")
    return "\n".join(lines).strip()


def drafter(state: AgentState) -> dict:
    reviews = state["retrieved_reviews"]
    anchors = _extract_anchors(reviews) if reviews else []
    avg_words = _avg_word_count(reviews)

    system_prompt = _build_system_prompt(
        avg_words, anchors, state["predicted_rating"], state["categories"]
    )
    samples = _format_samples(state)

    content = (
        f"## Persona Manifesto\n{state['user_manifesto']}\n\n"
        f"## Writing Samples\n{samples}\n\n"
        f"## Reasoning Trace\n{state['reasoning_log']}\n\n"
        f"## Target\n"
        f"Business: {state['biz_name']}\n"
        f"Rating to justify: {state['predicted_rating']}/5"
    )
    output: DrafterOutput = _llm.invoke(  # type: ignore[assignment]
        [SystemMessage(content=system_prompt), HumanMessage(content=content)]
    )
    return {"draft_review": output.review_text}
