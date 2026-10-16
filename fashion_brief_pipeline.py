#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The Data Fashion Brief — LangGraph pipeline.

Ingest public YouTube runway URLs, extract every look with Gemini, compute
per-designer percentages, and draft an editorial. The graph pauses for an
editor before it finishes.

Spyder:
    1. Install packages in Spyder's console (see below).
    2. Create google_api_key.txt in this same folder and put the Gemini key
       alone on the first line.
    3. Open this file and press F5. Type editor replies in the console, not here.

    pip install langgraph langchain-core langchain-google-genai pydantic

Gemini 3 model ids can be used for vision. Those calls switch on agentic
video reading, which suits a full runway better than fixed frame sampling.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, TypedDict

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, field_validator


def load_google_api_key() -> str:
    """Read the Gemini key from google_api_key.txt, then from the environment.

    The text file wins so a bad key already stored in Spyder does not override it.
    """
    key_path = Path(__file__).with_name("google_api_key.txt")
    if key_path.is_file():
        key = key_path.read_text(encoding="utf-8").strip().strip('"').strip("'")
        if key:
            os.environ["GOOGLE_API_KEY"] = key
            return key

    key = os.environ.get("GOOGLE_API_KEY", "").strip().strip('"').strip("'")
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        return key

    raise RuntimeError(
        "Missing Gemini API key. In the same folder as this script, create "
        "google_api_key.txt and put the key alone on the first line. Then press F5."
    )


load_google_api_key()

VISION_MODEL = os.environ.get("VISION_MODEL", "gemini-2.5-flash")
WRITER_MODEL = os.environ.get("WRITER_MODEL", "gemini-2.5-flash")

HeritageFidelity = Literal[
    "strict adherence",
    "modern subversive reinterpretation",
    "minimal/absent",
]

LUXURY_HOUSE_CODES: Dict[str, List[str]] = {
    "Dior": [
        "Bar jacket silhouette",
        "Cinched waist / New Look",
        "Cannage quilting",
        "Signature bow",
        "Houndstooth",
    ],
    "Chanel": [
        "Bouclé tweed suit",
        "Camellia flower motif",
        "Interlocking CC / Chain hardware",
        "Quilted leather (matelassé)",
        "Two-tone slingback footwear",
    ],
    "Schiaparelli": [
        "Surrealist anatomical elements (eyes, ears, lips)",
        "Shocking pink hue",
        "Gilded body jewelry / brass hardware",
        "Padlock closure motif",
        "Celestial / sunburst embroidery",
    ],
}

APPROVAL_WORDS = {"", "approve", "approved", "lgtm"}
MAX_REVISIONS = 2


class RunwaySource(TypedDict):
    source_id: int
    designer: str
    youtube_url: str


class FashionBriefState(TypedDict, total=False):
    event_name: str
    designers: List[str]
    video_urls: List[str]
    runway_sources: List[RunwaySource]
    raw_tags: List[Dict[str, Any]]
    aggregated_metrics: Dict[str, Any]
    macro_signals: Dict[str, Any]
    draft_article: str
    editor_feedback: str
    revision_count: int
    status: str


class FashionLookSchema(BaseModel):
    look_id: int = Field(description="Sequential look number in the show, starting at 1")
    garment_category: str = Field(
        default="daywear",
        description="One word: outerwear, eveningwear, tailored_set, or daywear",
    )
    colors: List[str] = Field(
        default_factory=list,
        description="Up to 4 color names, one word each",
    )
    color_palette_type: str = Field(
        default="neutral",
        description="Two-word palette label such as pastel_duo, monochrome, saturated_primary",
    )
    silhouette_label: str = Field(
        default="",
        description="2 to 4 words, such as oversized blazer or column gown",
    )
    textile_label: str = Field(
        default="",
        description="1 to 3 words, such as bouclé tweed or silk satin",
    )
    details_and_embellishments: List[str] = Field(
        default_factory=list,
        description="Up to 4 short tags",
    )
    accessories: List[str] = Field(
        default_factory=list,
        description="Up to 4 short tags",
    )
    vibrant_aesthetic: str = Field(
        default="",
        description="2 to 4 word mood label",
    )
    house_codes_present: List[str] = Field(
        default_factory=list,
        description="Exact house-code strings from the allowed list. Empty if none.",
    )
    heritage_fidelity: HeritageFidelity = Field(
        default="minimal/absent",
        description="strict adherence, modern subversive reinterpretation, or minimal/absent",
    )

    @field_validator("heritage_fidelity", mode="before")
    @classmethod
    def _coerce_fidelity(cls, value: Any) -> str:
        allowed = {
            "strict adherence",
            "modern subversive reinterpretation",
            "minimal/absent",
        }
        text = str(value or "").strip()
        if text in allowed:
            return text
        return "minimal/absent"


class ShowAnalysis(BaseModel):
    looks: List[FashionLookSchema] = Field(
        description="Every distinct runway look, in show order. One object per model exit."
    )


def _video_block(url: str) -> Dict[str, str]:
    block = {
        "type": "video",
        "url": url,
        "mime_type": "video/mp4",
    }
    # Agentic timeline reading is available on Gemini 3 and later.
    if VISION_MODEL.startswith("gemini-3"):
        block["media_processing"] = "AGENTIC"
    return block


def _present_once(counter: Counter, values: List[Any]) -> None:
    """Count each value at most once per look so percentages stay within 0-100."""
    seen: List[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    counter.update(seen)


def _share_of_looks(counter: Counter, total_looks: int) -> Dict[str, float]:
    if total_looks <= 0:
        return {}
    return {
        item: round((count / total_looks) * 100, 1)
        for item, count in counter.most_common()
    }


def _normalize_color(value: Any) -> str:
    text = " ".join(str(value).strip().lower().replace("-", " ").split())
    aliases = {
        "off white": "white",
        "ivory": "cream",
        "ecru": "cream",
        "navy blue": "navy",
        "light blue": "blue",
        "sky blue": "blue",
        "baby blue": "blue",
        "dark blue": "navy",
        "gray": "grey",
        "charcoal": "grey",
    }
    return aliases.get(text, text)


def format_color_report(metrics: Dict[str, Any]) -> str:
    """One color list per house. A percentage is the share of that house's looks."""
    houses = (metrics or {}).get("per_designer_analysis") or {}
    lines = [
        "COLOR BREAKDOWN BY HOUSE",
        "Each percentage is the share of that house's looks in which the color appears.",
        "One look can include more than one color, so a house's percentages can add up to more than 100.",
        "",
    ]
    if not houses:
        lines.append("No house colors were recorded.")
        return "\n".join(lines)

    for designer, data in houses.items():
        total = int(data.get("total_looks") or 0)
        colors = data.get("color_distribution") or {}
        lines.append(f"{designer} — {total} looks")
        if not colors:
            lines.append("  No colors recorded.")
            lines.append("")
            continue
        leader, leader_pct = next(iter(colors.items()))
        lines.append(f"  {leader} leads at {leader_pct}% of looks")
        for name, pct in colors.items():
            look_count = int(round((pct / 100) * total)) if total else 0
            lines.append(f"  {name:<16} {pct:>5.1f}%    {look_count} looks")
        lines.append("")
    return "\n".join(lines).rstrip()


LOOKS_PER_CALL = 30
MAX_LOOKS_PER_COLLECTION = 100
MAX_LOOK_BATCHES = 6

vision_llm = ChatGoogleGenerativeAI(
    model=VISION_MODEL,
    temperature=0.1,
    max_output_tokens=32768,
    timeout=600,
)
structured_vision_llm = vision_llm.with_structured_output(
    ShowAnalysis, include_raw=True
)

writer_llm = ChatGoogleGenerativeAI(
    model=WRITER_MODEL,
    temperature=0.4,
    max_output_tokens=8192,
    timeout=180,
)


def ingestion_node(state: FashionBriefState) -> Dict[str, Any]:
    """Pair each supplied YouTube URL with its designer."""
    event_name = state.get("event_name", "Fashion Week")
    print(f"--- [INGESTION AGENT] Processing YouTube links for {event_name} ---")

    urls = state.get("video_urls") or []
    designers = state.get("designers") or []
    if not urls:
        raise ValueError("Ingestion error: initial state has no video_urls.")
    if len(urls) != len(designers):
        raise ValueError(
            f"Ingestion error: {len(designers)} designers for {len(urls)} video_urls. "
            "Pass one designer name per URL, in the same order."
        )

    runway_sources: List[RunwaySource] = []
    for index, (designer, url) in enumerate(zip(designers, urls), start=1):
        if "youtube.com/watch" not in url and "youtu.be/" not in url:
            raise ValueError(f"Ingestion error: not a public YouTube URL: {url}")
        runway_sources.append(
            {
                "source_id": index,
                "designer": designer,
                "youtube_url": url,
            }
        )
        print(f"    {index}. {designer}: {url}")

    return {"runway_sources": runway_sources, "status": "ingested"}


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        return "\n".join(parts)
    return str(content)


def _repair_json(text: str) -> Dict[str, Any]:
    """Parse model JSON, dropping a truncated final look if the reply was cut off."""
    start = text.find("{")
    if start < 0:
        raise ValueError("model output had no JSON object")
    blob = text[start:].replace("```json", "").replace("```", "")
    candidates = [blob]
    last = len(blob)
    for _ in range(12):
        cut = blob.rfind("}", 0, last)
        if cut < 0:
            break
        piece = blob[: cut + 1].rstrip().rstrip(",")
        extra_brackets = piece.count("[") - piece.count("]")
        extra_braces = piece.count("{") - piece.count("}")
        piece += "]" * max(extra_brackets, 0)
        piece += "}" * max(extra_braces, 0)
        candidates.append(piece)
        last = cut

    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    raise ValueError("could not repair the model JSON")


def _looks_from_result(result: Dict[str, Any]) -> Tuple[List[FashionLookSchema], bool]:
    parsed = result.get("parsed")
    raw = result.get("raw")
    error = result.get("parsing_error")
    meta = getattr(raw, "response_metadata", {}) or {}
    reason = str(meta.get("finish_reason") or meta.get("finishReason") or "").upper()
    truncated = error is not None or reason in {"MAX_TOKENS", "LENGTH", "MAX_OUTPUT_TOKENS"}

    if isinstance(parsed, ShowAnalysis):
        return list(parsed.looks), truncated

    payload = _repair_json(_message_text(raw))
    looks: List[FashionLookSchema] = []
    for item in payload.get("looks") or []:
        if not isinstance(item, dict):
            continue
        try:
            looks.append(FashionLookSchema.model_validate(item))
        except Exception:
            continue
    return looks, True


def _match_house_codes(returned: List[str], allowed: List[str]) -> List[str]:
    matched: List[str] = []
    allowed_set = set(allowed)
    for code in returned:
        text = str(code).strip()
        if not text:
            continue
        if text in allowed_set and text not in matched:
            matched.append(text)
            continue
        lowered = text.lower()
        for official in allowed:
            official_lower = official.lower()
            if official_lower in lowered or lowered in official_lower:
                if official not in matched:
                    matched.append(official)
                break
    return matched


def _look_identity(look: FashionLookSchema) -> Tuple[str, ...]:
    """One identity per exit. A repeated description of the same outfit collapses."""
    return (
        look.garment_category.strip().lower(),
        look.silhouette_label.strip().lower(),
        look.textile_label.strip().lower(),
        tuple(color.strip().lower() for color in look.colors),
        tuple(detail.strip().lower() for detail in look.details_and_embellishments),
        tuple(accessory.strip().lower() for accessory in look.accessories),
    )


def vision_tagging_node(state: FashionBriefState) -> Dict[str, Any]:
    """Watch each show in batches so a long runway is not cut off mid-JSON."""
    print("--- [VISION TAGGING AGENT] Extracting looks from YouTube videos ---")

    extracted_tags: List[Dict[str, Any]] = []
    for source in state.get("runway_sources") or []:
        designer_name = source["designer"]
        youtube_url = source["youtube_url"]
        allowed_codes = LUXURY_HOUSE_CODES.get(designer_name, [])
        if allowed_codes:
            code_lines = "\n".join(f"- {code}" for code in allowed_codes)
        else:
            code_lines = "- No codes are catalogued. Leave house_codes_present empty."

        print(f"    Watching {designer_name}: {youtube_url}")
        collected: List[FashionLookSchema] = []
        seen_identities = set()
        previous_hint = (
            "Start at the first model who walks onto the runway. "
            f"This reply covers the first {LOOKS_PER_CALL} exits only. "
            "Later replies will ask for the rest of the show."
        )

        for _batch in range(MAX_LOOK_BATCHES):
            if len(collected) >= MAX_LOOKS_PER_COLLECTION:
                break
            room = MAX_LOOKS_PER_COLLECTION - len(collected)
            page_size = min(LOOKS_PER_CALL, room)
            prompt = f"""
Watch this runway video for {designer_name} at {state.get("event_name", "Fashion Week")}.
{previous_hint}

Count one look only when a model enters in a new outfit.
Do not create a look for a close-up, a turn, a second walk in the same outfit, a guest, or backstage.
Do not invent outfits. Stop at the designers' final bow.
Return at most {page_size} looks in this reply.
If no exits remain, return an empty looks list.

Use short labels only. No sentences.

House codes for {designer_name}. Copy one only when it is clearly visible, using the exact string:
{code_lines}

heritage_fidelity must be exactly one of: strict adherence, modern subversive reinterpretation, minimal/absent.
If no listed code is visible, house_codes_present is [] and heritage_fidelity is minimal/absent.
""".strip()

            result = structured_vision_llm.invoke(
                [
                    HumanMessage(
                        content=[
                            _video_block(youtube_url),
                            {"type": "text", "text": prompt},
                        ]
                    )
                ]
            )
            looks, truncated = _looks_from_result(result)
            fresh: List[FashionLookSchema] = []
            for look in looks:
                identity = _look_identity(look)
                if identity in seen_identities:
                    continue
                seen_identities.add(identity)
                fresh.append(look)
            if not fresh:
                print(f"    {designer_name}: no further unique exits")
                break

            collected.extend(fresh)
            print(f"    {designer_name}: {len(collected)} unique looks so far")
            if len(collected) >= MAX_LOOKS_PER_COLLECTION:
                collected = collected[:MAX_LOOKS_PER_COLLECTION]
                print(f"    {designer_name}: stopped at the {MAX_LOOKS_PER_COLLECTION}-look ceiling")
                break
            if len(looks) < page_size and not truncated:
                break

            last = collected[-1]
            previous_hint = (
                f"Looks 1 through {len(collected)} are already recorded. "
                f"The last one is {last.silhouette_label} in {', '.join(last.colors) or 'unspecified colors'}. "
                f"Continue with the next real exits only, up to {page_size} in this reply. "
                "Do not repeat outfits already recorded. "
                "If the show has already ended, return an empty looks list."
            )

        if not collected:
            raise RuntimeError(
                f"Vision model returned no looks for {designer_name} ({youtube_url}). "
                "Confirm the video is public and the model id supports YouTube input."
            )

        for index, look in enumerate(collected, start=1):
            row = look.model_dump()
            row["look_id"] = index
            row["designer"] = designer_name
            row["youtube_url"] = youtube_url
            codes = _match_house_codes(row["house_codes_present"], allowed_codes)
            row["house_codes_present"] = codes
            if not codes:
                row["heritage_fidelity"] = "minimal/absent"
            extracted_tags.append(row)

        print(f"    {designer_name}: {len(collected)} looks")

    return {"raw_tags": extracted_tags, "status": "tagged"}


def analytics_aggregation_node(state: FashionBriefState) -> Dict[str, Any]:
    """Turn per-look tags into a percentage of looks for each designer."""
    print("--- [ANALYTICS AGENT] Computing per-designer percentages ---")

    designer_counters: Dict[str, Dict[str, Any]] = {}
    for tag in state.get("raw_tags") or []:
        designer = tag.get("designer") or "Unknown Designer"
        bucket = designer_counters.setdefault(
            designer,
            {
                "total_looks": 0,
                "categories": Counter(),
                "colors": Counter(),
                "palettes": Counter(),
                "silhouettes": Counter(),
                "textiles": Counter(),
                "details": Counter(),
                "accessories": Counter(),
                "aesthetics": Counter(),
                "looks_with_house_codes": 0,
                "house_codes_found": Counter(),
                "fidelity_ratings": Counter(),
            },
        )
        bucket["total_looks"] += 1
        _present_once(bucket["categories"], [tag.get("garment_category", "")])
        color_names = [
            _normalize_color(color)
            for color in (tag.get("colors") or [])
            if _normalize_color(color)
        ]
        _present_once(bucket["colors"], color_names)
        _present_once(bucket["palettes"], [tag.get("color_palette_type", "")])
        _present_once(bucket["silhouettes"], [tag.get("silhouette_label", "")])
        _present_once(bucket["textiles"], [tag.get("textile_label", "")])
        _present_once(bucket["details"], tag.get("details_and_embellishments") or [])
        _present_once(bucket["accessories"], tag.get("accessories") or [])
        _present_once(bucket["aesthetics"], [tag.get("vibrant_aesthetic", "")])

        codes = tag.get("house_codes_present") or []
        if codes:
            bucket["looks_with_house_codes"] += 1
            _present_once(bucket["house_codes_found"], codes)
        fidelity = tag.get("heritage_fidelity")
        if fidelity:
            _present_once(bucket["fidelity_ratings"], [fidelity])

    designer_breakdown: Dict[str, Any] = {}
    for designer, data in designer_counters.items():
        total = data["total_looks"]
        rate = round((data["looks_with_house_codes"] / total) * 100, 1) if total else 0.0
        designer_breakdown[designer] = {
            "total_looks": total,
            "house_code_integration_rate_pct": rate,
            "top_house_codes_used": _share_of_looks(data["house_codes_found"], total),
            "heritage_fidelity_breakdown": _share_of_looks(data["fidelity_ratings"], total),
            "color_distribution": _share_of_looks(data["colors"], total),
            "palette_distribution": _share_of_looks(data["palettes"], total),
            "category_distribution": _share_of_looks(data["categories"], total),
            "silhouette_distribution": _share_of_looks(data["silhouettes"], total),
            "textile_distribution": _share_of_looks(data["textiles"], total),
            "detail_distribution": _share_of_looks(data["details"], total),
            "accessory_distribution": _share_of_looks(data["accessories"], total),
            "aesthetic_distribution": _share_of_looks(data["aesthetics"], total),
        }
        print(f"    {designer}: {total} looks, house-code rate {rate}%")

    metrics = {
        "per_designer_analysis": designer_breakdown,
        "percentage_meaning": (
            "Each percentage is the share of that designer's looks in which "
            "the attribute appears at least once."
        ),
        "key_insight": (
            f"Brand-level trend breakdowns for {len(designer_breakdown)} designers."
        ),
    }
    color_report = format_color_report(metrics)
    metrics["color_report"] = color_report
    print("\n" + color_report + "\n")

    return {"aggregated_metrics": metrics, "status": "analyzed"}


def editorial_journalist_node(state: FashionBriefState) -> Dict[str, Any]:
    """Draft a structured article from the computed metrics."""
    print("--- [EDITORIAL AGENT] Writing the data fashion brief ---")

    feedback = (state.get("editor_feedback") or "").strip()
    revision_block = ""
    if feedback.lower() not in APPROVAL_WORDS:
        revision_block = f"\nRevise the previous draft using these editor notes:\n{feedback}\n"

    macro = state.get("macro_signals") or {}
    macro_instruction = (
        "A Macro signals section may be included from this data: "
        f"{macro}"
        if macro
        else "Do not invent search trends, earnings, or market-interest numbers. "
        "Omit any market-interest column and any macro section."
    )

    prompt = f"""
You are an expert fashion data journalist at The Data Fashion Brief.
Readers include industry professionals and people who do not work in fashion.

Write a trend analysis from these inputs only:
- Event: {state.get("event_name", "Runway Presentation")}
- Analytics: {state.get("aggregated_metrics", {})}
- {macro_instruction}
{revision_block}
Structure:
1. Short lede, then clear headings. No long blocks of text.
2. A Markdown table with one row per designer: Designer, Looks analyzed, House-code integration %, Top house code, Top color, Top textile.
3. A "Color breakdown" section. Give every house its own heading and list every color in color_distribution with the exact percentage. Example shape: "White 47.0% of looks. Blue 5.0% of looks." Do not drop small colors, round the figures, or invent colors.
4. After the colors, explain a house code in plain language: a recurring signature a brand repeats so the collection is recognizable. Explain a silhouette as the outline of the garment on the body.
5. For each designer, say what the largest percentages imply about the collection.
6. Analytical and direct. Every number must come from the analytics object.
""".strip()

    response = writer_llm.invoke([HumanMessage(content=prompt)])
    article = response.content
    if isinstance(article, list):
        article = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in article
        )
    return {"draft_article": article, "status": "drafted"}


def human_review_node(state: FashionBriefState) -> Dict[str, Any]:
    """Record editor sign-off. Feedback is written into state before this node runs."""
    print("--- [HITL] Recording editor decision ---")
    feedback = (state.get("editor_feedback") or "").strip()
    revisions = state.get("revision_count", 0)
    approved = feedback.lower() in APPROVAL_WORDS or revisions >= MAX_REVISIONS
    if approved:
        return {
            "status": "approved",
            "editor_feedback": feedback or "Approved.",
        }
    return {
        "status": "needs_revision",
        "revision_count": revisions + 1,
    }


def route_after_review(state: FashionBriefState) -> str:
    if state.get("status") == "approved":
        return "end"
    return "write_article"


workflow = StateGraph(FashionBriefState)
workflow.add_node("ingest", ingestion_node)
workflow.add_node("vision_tag", vision_tagging_node)
workflow.add_node("analyze", analytics_aggregation_node)
workflow.add_node("write_article", editorial_journalist_node)
workflow.add_node("review", human_review_node)

workflow.set_entry_point("ingest")
workflow.add_edge("ingest", "vision_tag")
workflow.add_edge("vision_tag", "analyze")
workflow.add_edge("analyze", "write_article")
workflow.add_edge("write_article", "review")
workflow.add_conditional_edges(
    "review",
    route_after_review,
    {"write_article": "write_article", "end": END},
)

app = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["review"],
)


if __name__ == "__main__":
    initial_state: FashionBriefState = {
        "event_name": "Paris Fashion Week SS27",
        "designers": ["Dior", "Chanel", "Schiaparelli"],
        "video_urls": [
            "https://www.youtube.com/watch?v=DB7oO82OeD4",
            "https://www.youtube.com/watch?v=weNFjc2l1Hw",
            "https://www.youtube.com/watch?v=T4ZzV7R6GXQ",
        ],
        "raw_tags": [],
        "aggregated_metrics": {},
        "macro_signals": {},
        "editor_feedback": "",
        "revision_count": 0,
        "status": "new",
    }

    config = {"configurable": {"thread_id": "pfw-ss27"}, "recursion_limit": 50}

    print("Starting The Data Fashion Brief pipeline\n")
    result = app.invoke(initial_state, config)

    while result.get("status") != "approved":
        print("\n================ COLOR BREAKDOWN ================\n")
        print(format_color_report(result.get("aggregated_metrics") or {}))
        print("\n================ DRAFT FOR REVIEW ================\n")
        print(result.get("draft_article", "No draft generated."))
        print("\nType editor notes to request a rewrite, or 'approve' to finish.")
        feedback = input("Editor: ").strip()
        app.update_state(config, {"editor_feedback": feedback or "approve"})
        result = app.invoke(None, config)

    print("\n================ COLOR BREAKDOWN ================\n")
    print(format_color_report(result.get("aggregated_metrics") or {}))
    print("\n================ FINAL EDITORIAL BRIEF ================\n")
    print(result.get("draft_article", "No draft generated."))
    print(f"\nStatus: {result.get('status')}")
