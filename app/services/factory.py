from .anthropic_svc import AnthropicService
from .gemini import GeminiService
from .lmstudio import LMStudioService
from .ollama import OllamaService
from .openai_svc import OpenAIService

# Single source of truth for the cloud-provider model pickers (served via
# /models). Local providers (LM Studio, Ollama) are queried live instead.
CLOUD_MODELS = {
    "anthropic": [
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ],
    "openai": [
        "gpt-5",
        "gpt-5-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "gemini": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
    ],
}

PROMPT_TEMPLATES = {
    "meeting": """You are a meeting assistant. Below is a meeting transcript with speaker labels and timestamps.

Write a concise summary with bullet points covering:
- Key topics discussed
- Decisions made (if any)
- Action items — include who is responsible (if any)
- Participants

Use 4–8 bullet points total. Only summarize what is actually in the transcript. Use plain bullet points starting with -.

TRANSCRIPT:
{transcript}

SUMMARY:""",

    "call": """You are summarizing a phone call. Below is the transcript with speaker labels.

Write a concise summary with bullet points covering:
- The main subject discussed
- Key information exchanged
- Any follow-ups, plans, or open questions

Use 3–6 bullet points total. Only summarize what is in the transcript. Use plain bullet points starting with -.

TRANSCRIPT:
{transcript}

SUMMARY:""",

    "interview": """You are summarizing an interview. Below is the transcript with speaker labels.

Write a concise summary with bullet points covering:
- Topics covered (in order)
- The most important answers or insights given
- Notable quotes (only if directly from the transcript)

Use 4–8 bullet points total. Only summarize what is in the transcript. Use plain bullet points starting with -.

TRANSCRIPT:
{transcript}

SUMMARY:""",

    "lecture": """You are summarizing a lecture or talk. Below is the transcript with speaker labels.

Produce a structured outline:
- Main thesis or topic
- Key points and arguments
- Examples or evidence cited
- Takeaways or conclusions

Use 5–10 bullet points total. Only summarize what is in the transcript. Use plain bullet points starting with -.

TRANSCRIPT:
{transcript}

SUMMARY:""",
}

# Backwards-compat alias — tests and providers still import this name.
SUMMARIZATION_PROMPT = PROMPT_TEMPLATES["meeting"]


COMBINE_PROMPT = """You are merging several partial summaries of one long recording into a single, cohesive summary.

Below are the section summaries in order. Produce one unified bullet-point summary in the same style as the sections (key topics, decisions, action items, participants — whichever apply). Deduplicate, keep it concise, 6–10 bullet points.

SECTIONS:
{sections}

UNIFIED SUMMARY:"""


def build_summary_prompt(style: str, transcript: str, custom_prompt: str = "") -> str:
    """Resolve the summarization prompt for the given style, injecting the transcript."""
    if style == "custom" and custom_prompt.strip():
        template = custom_prompt
        if "{transcript}" not in template:
            template = template.rstrip() + "\n\nTRANSCRIPT:\n{transcript}\n\nSUMMARY:"
        return template.format(transcript=transcript)
    template = PROMPT_TEMPLATES.get(style) or PROMPT_TEMPLATES["meeting"]
    return template.format(transcript=transcript)


def format_transcript(segments: list, speaker_names: dict) -> str:
    lines = []
    for seg in segments:
        sid = seg.get("speaker", "SPEAKER_00")
        name = speaker_names.get(sid) or sid.replace("SPEAKER_", "Speaker ")
        start = seg.get("start", 0)
        mm, ss = int(start) // 60, int(start) % 60
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{mm:02d}:{ss:02d}] {name}: {text}")
    return "\n".join(lines)


def get_summarizer(cfg: dict):
    provider = cfg.get("provider") or "lmstudio"
    base_url = cfg.get("base_url") or ""
    model    = cfg.get("model") or ""
    api_key  = cfg.get("api_key") or ""

    if provider == "lmstudio":
        return LMStudioService(base_url=base_url or "http://localhost:1234", model=model)
    if provider == "ollama":
        return OllamaService(base_url=base_url or "http://localhost:11434", model=model)
    if provider == "gemini":
        return GeminiService(api_key=api_key, model=model)
    if provider == "anthropic":
        return AnthropicService(api_key=api_key, model=model)
    if provider == "openai":
        return OpenAIService(api_key=api_key, model=model)
    raise ValueError(f"Unknown provider: {provider!r}")
