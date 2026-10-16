# 🧵 The Fashion Intelligence Brief: Multimodal Runway & Trend Analytics Agent

A stateful multi-agent system built with **LangGraph**, **Multimodal Gemini Models**, and **Python**. Designed for fashion data journalism and market analysis, this system ingests fashion runway assets via **YouTube links or direct image inputs**, extracts critical metadata (such as the **Fashion House and Brand Name**), and performs deep visual analysis to compute trend statistics, including **dominant colors, comprehensive color palettes, and detailed silhouettes**.

---

## 🏗️ System Architecture & Workflow

The platform utilizes a cyclical, stateful graph (`StateGraph`) managed by **LangGraph** to pass context seamlessly across specialized agent roles:

```text
               ┌────────────────────────────────────────┐
               │    Ingestion & Metadata Agent          │ (Ingests YouTube URLs / Images & extracts Brand/House name)
               └───────────────────┬────────────────────┘
                                   ▼
               ┌────────────────────────────────────────┐
               │   Computer Vision & Look Analysis Agent│ (Multimodal VLM analyzes visual assets for silhouettes & color palettes)
               └───────────────────┬────────────────────┘
                                   ▼
               ┌────────────────────────────────────────┐
               │   Data Analytics & Aggregation Agent   │ (Computes statistical trend dominance and color percentages)
               └───────────────────┬────────────────────┘
                                   ▼
               ┌────────────────────────────────────────┐
               │  Editorial Journalist Agent            │ (Synthesizes data into a cohesive, data-driven Fashion Brief)
               └───────────────────┬────────────────────┘
                                   ▼
               ┌────────────────────────────────────────┐
               │ Human-in-the-Loop Review Gate          │ (Editor reviews, requests edits, or signs off on publication)
               └────────────────────────────────────────┘
```

The graph is cyclical because the review gate can route back to the Editorial Journalist Agent: when the editor requests changes, their feedback is added to the shared state and the brief is redrafted. Only an explicit sign-off ends the run.

---

## ✨ Core Capabilities & Data Processing

1. **Multi-Source Ingestion:** Accepts both YouTube runway video URLs and high-resolution images as primary input sources.
2. **Metadata Tagging:** Extracts metadata - **Fashion House Name** and **Brand Name** for each collection.
3. **Granular Visual Analytics:** Analyzes runway frames to extract key fashion metrics:
   * **Silhouettes:** Identifies cut, fit, and structural shapes (e.g., relaxed, fluid, tailored, bold).
   * **Dominant Colors & Palettes:** Detects primary color protagonists and breaks down the complete season-wide color distribution percentages.
4. **Editorial Synthesis:** Transforms raw statistical data into professional, trade-publication-ready trend reports.

---

## 🚀 Tech Stack & Core Libraries

* **Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph) (stateful multi-agent orchestration and checkpointing)
* **LLM & Vision:** `langchain-google-genai` powered by Google Gemini (`gemini-2.5-flash` / Pro)
* **Data Validation:** `pydantic` (strict structured extraction schemas for metadata and looks)
* **Data Processing:** `pandas` / Python native types
* **Environment:** Python 3.13 / Anaconda

---


