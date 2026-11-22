"""Generate company summaries via a shared AI client (Groq, Aion Labs, OpenAI)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from llm_client import PROVIDERS, LLMError, ask_ai_with_response, list_models, resolve_model, response_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = Path("data/companies.jsonl")
DEFAULT_SAMPLE = Path("samples/representative-8.txt")
CACHE_VERSION = 1

REQUIRED_HEADINGS = (
    "Company",
    "Director & Shareholder",
    "Financials",
    "Litigation",
    "Review Rating",
)
CANONICAL_HEADINGS = {heading.lower(): heading for heading in REQUIRED_HEADINGS}
HEADING_PATTERN = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?"
    rf"(?P<heading>{'|'.join(map(re.escape, REQUIRED_HEADINGS))})"
    r"[ \t]*(?::(?:\*\*)?|(?:\*\*):?|$)[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)
MONEY_PATTERN = re.compile(
    r"(?<!\w)(?:(?P<prefix_sign>[-\u2212])[ \t]*)?RM[ \t]*(?P<suffix_sign>[-\u2212])?[ \t]*"
    r"(?P<amount>(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:\.\d+)?)"
    r"(?:[ \t]*(?P<unit>[km])\b)?(?!\w|[,.]\d)",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are a company-vetting analyst. Read the complete synthetic company JSON record, perform the required "
    "calculations yourself, decide how to express the facts concisely, and write a short summary.\n\n"
    "Use only the supplied record. Do not invent facts, estimate missing values, give advice, or add unsupported "
    "risk labels. Do not assume that missing data means zero.\n\n"
    "Return only five Markdown paragraphs with these headings, in this order:\n\n"
    "**Company:** State the current name; put '(formerly known as ...)' immediately after it when former names "
    "exist. Include company type, incorporation date, current status, industry, state, nature of business, and "
    "share capital.\n\n"
    "**Director & Shareholder:** Count directors and shareholders, use correct singular/plural wording, and name "
    "the largest shareholder and percentage. If shareholders are absent, say so.\n\n"
    "**Financials:** Use the latest financial year. Calculate YoY revenue growth against the immediately preceding "
    "financial year as (latest revenue - previous revenue) / abs(previous revenue) * 100. Calculate gross profit "
    "margin as gross profit / revenue * 100 and operating profit margin as operating profit / revenue * 100. "
    "Report free cash flow from the record and calculate net debt as borrowings - cash. Describe growth, decline, "
    "and positive or negative cash flow naturally. If financial data or a prior year is absent, state that clearly.\n\n"
    "**Litigation:** Count all cases, separately count and sum claim amounts for defendant and plaintiff cases, "
    "and aggregate the case statuses. If there are no cases, say 'No court cases were recorded.'\n\n"
    "**Review Rating:** Assign a review-priority score from 1/5 to 5/5, add the matching label, and give one short "
    "sentence explaining the most important reasons. Higher scores mean higher priority for further supplier or "
    "partner vetting. Use this score only to prioritise further review. Use this rubric:\n"
    "- 5/5 — Very high review priority: active status, broadly stable or positive financial performance, positive "
    "cash generation, and no material defendant litigation recorded.\n"
    "- 4/5 — High review priority: mostly sound indicators with one or two modest adverse signals.\n"
    "- 3/5 — Moderate review priority: mixed positive and negative signals, meaningful uncertainty, or missing "
    "financial information that prevents a confident assessment.\n"
    "- 2/5 — Low review priority: several material adverse signals such as declining performance, negative "
    "operating margin, negative free cash flow, substantial net debt, or material defendant litigation.\n"
    "- 1/5 — Very low review priority: winding-up status or a severe combination of financial weakness and "
    "material defendant litigation.\n"
    "Use 3/5 for missing financial information alone when it prevents a confident assessment. Recorded winding-up "
    "status takes priority over missing information and always receives 1/5. Do not treat plaintiff claim amounts "
    "as liabilities. Base the rating only on evidence in the supplied JSON.\n\n"
    "Formatting rules:\n"
    "- Use one decimal place for financial percentages and two decimal places for shareholder percentages.\n"
    "- Format ringgit compactly as RM, RMx.xk, or RMx.xxm. Put a minus sign before RM for negative amounts.\n"
    "- Keep the entire summary concise and readable. Use your own natural wording rather than copying a fixed "
    "sentence template.\n"
)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def version_paths(provider: str, model: str) -> tuple[Path, Path]:
    summaries = PROJECT_ROOT / "summaries"
    responses = PROJECT_ROOT / "responses"
    pattern = re.compile(rf"{re.escape(provider)}-v([1-9][0-9]*)")
    used = set()
    for path in summaries.glob(f"{provider}-v*.jsonl"):
        if match := pattern.fullmatch(path.stem):
            used.add(int(match[1]))
    for path in responses.glob(f"{provider}-v*-raw"):
        version = path.name.removesuffix("-raw")
        if match := pattern.fullmatch(version):
            used.add(int(match[1]))
            marker = path / ".model"
            if marker.is_file() and marker.read_text(encoding="utf-8").strip() == model:
                return summaries / f"{version}.jsonl", path

    version = f"{provider}-v{max(used, default=0) + 1}"
    return summaries / f"{version}.jsonl", responses / f"{version}-raw"


def user_prompt(company: dict[str, Any]) -> str:
    return "Read, calculate, and summarize this raw company record.\n\n" + json.dumps(company, ensure_ascii=False, indent=2)


def normalize_summary(summary: str) -> str:
    """Normalize heading variants, whitespace, and compact ringgit formatting."""
    summary = "\n".join(summary.splitlines()).replace("\u00a0", " ").replace("\u202f", " ")
    summary = HEADING_PATTERN.sub(lambda match: f"\n\n**{CANONICAL_HEADINGS[match['heading'].lower()]}:** ", summary)

    def replace_money(match: re.Match[str]) -> str:
        negative = bool(match.group("prefix_sign") or match.group("suffix_sign"))
        amount = float(match.group("amount").replace(",", "").replace(" ", ""))
        unit = (match.group("unit") or "").lower()
        if unit == "k":
            rendered = f"{amount:.1f}k"
        elif unit == "m":
            rendered = f"{amount:.2f}m"
        elif amount >= 1_000_000:
            rendered = f"{amount / 1_000_000:.2f}m"
        elif amount >= 1_000:
            rendered = f"{amount / 1_000:.1f}k"
        else:
            rendered = f"{amount:,.0f}"

        return f"{'-' if negative else ''}RM{rendered}"

    summary = MONEY_PATTERN.sub(replace_money, summary)
    summary = re.sub(r"[ \t]+$", "", summary, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", summary).strip()


def validate_summary(text: str) -> str:
    summary = normalize_summary(text)
    headings = list(HEADING_PATTERN.finditer(summary))
    if tuple(match["heading"] for match in headings) != REQUIRED_HEADINGS or headings[0].start() != 0:
        raise ValueError("Expected the five required summary headings exactly once and in order")
    for heading, following in zip(headings, [*headings[1:], None], strict=True):
        end = following.start() if following is not None else len(summary)
        section = summary[heading.end() : end].strip()
        if not section:
            raise ValueError(f"Empty summary section: {heading['heading']}")
        if heading["heading"] == REQUIRED_HEADINGS[-1] and not re.search(
            r"(?<![\w.-])[1-5][ \t]*/[ \t]*5(?![\w/]|\.\d)", section
        ):
            raise ValueError("Review Rating must include a score from 1/5 to 5/5")
    return summary


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)


def load_rows(path: Path, *, summaries: bool = False) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON: {error.msg}") from error
            registration = row.get("registrationNumber") if isinstance(row, dict) else None
            if not isinstance(registration, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", registration):
                raise ValueError(f"{path}:{number}: invalid registrationNumber")
            if summaries and not isinstance(row.get("summary"), str):
                raise ValueError(f"{path}:{number}: expected a summary string")
            if registration in rows:
                raise ValueError(f"{path}:{number}: duplicate registrationNumber {registration}")
            rows[registration] = row
    return rows


def select_companies(args: argparse.Namespace) -> list[dict[str, Any]]:
    companies = load_rows(project_path(args.input))
    if args.all:
        selected = list(companies)
    else:
        selected = list(args.registration_number or [])
        sample = args.registration_numbers_file
        if sample is None and not selected:
            sample = DEFAULT_SAMPLE
        if sample is not None:
            for line in project_path(sample).read_text(encoding="utf-8-sig").splitlines():
                registration = line.partition("#")[0].strip()
                if registration:
                    selected.append(registration)
        selected = list(dict.fromkeys(selected))
    missing = [registration for registration in selected if registration not in companies]
    if missing:
        raise ValueError(f"Unknown registration number(s): {', '.join(missing)}")
    if not selected:
        raise ValueError("No companies selected")
    return [companies[registration] for registration in selected[: args.limit]]


def generate_summary(
    company: dict[str, Any],
    provider: str,
    model: str,
    responses_dir: Path,
    *,
    force: bool = False,
) -> tuple[str, bool]:
    prompt = user_prompt(company)
    # Hash the exact request inputs so a new model, record, or prompt gets its own cache entry.
    request = json.dumps([CACHE_VERSION, provider, model, SYSTEM_PROMPT, prompt], ensure_ascii=False)
    fingerprint = hashlib.sha256(request.encode("utf-8")).hexdigest()
    cache_path = responses_dir / company["registrationNumber"] / f"{fingerprint}.json"
    if not force and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("version") == CACHE_VERSION and cached.get("fingerprint") == fingerprint:
                return validate_summary(response_text(cached["response"])), True
        except KeyError, ValueError:
            # Invalid JSON, unreadable completion or failed summary validation: fetch a fresh response.
            pass

    result = ask_ai_with_response(prompt, provider=provider, system_prompt=SYSTEM_PROMPT, model=model)
    entry = {"version": CACHE_VERSION, "fingerprint": fingerprint, "response": result.raw}
    serialized = json.dumps(entry, ensure_ascii=False, indent=2) + "\n"
    try:
        summary = validate_summary(result.text)
    except ValueError:
        atomic_write(cache_path.with_suffix(".invalid.json"), serialized)
        raise
    atomic_write(cache_path, serialized)
    return summary, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, default="groq")
    # parser.add_argument("--provider", choices=PROVIDERS, default="aionlabs")


    parser.add_argument("--model", help="Override the provider's configured model")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, help="Override the model's versioned summary file")
    parser.add_argument("--responses-dir", type=Path, help="Override the model's versioned response directory")
    parser.add_argument("--registration-number", action="extend", nargs="+", help="Select one or more company IDs")
    parser.add_argument("--registration-numbers-file", type=Path, help="Read company IDs from a file, one per line")
    parser.add_argument("--all", action="store_true", help="Process every company instead of the default eight-company sample")
    parser.add_argument("--limit", type=int, help="Process at most this many selected companies")
    parser.add_argument("--force", action="store_true", help="Refresh selected companies even when a valid cache exists")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print the selection and paths without contacting a provider")
    mode.add_argument("--list-models", action="store_true", help="List models available from the selected provider")
    args = parser.parse_args()
    if args.all and (args.registration_number or args.registration_numbers_file):
        parser.error("--all cannot be combined with explicit company selection")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def run(args: argparse.Namespace) -> None:
    if args.list_models:
        for model in list_models(args.provider):
            print(model)
        return

    provider = args.provider
    model = resolve_model(provider, args.model)
    companies = select_companies(args)
    default_output, default_responses = version_paths(provider, model)
    output = project_path(args.output) if args.output is not None else default_output
    responses = project_path(args.responses_dir) if args.responses_dir is not None else default_responses
    if output.resolve() in (project_path(args.input).resolve(), project_path(DEFAULT_INPUT).resolve()):
        raise ValueError("Summary output must not point to a company data file")
    if args.dry_run:
        print(f"{provider} / {model}: {len(companies)} selected")
        print(f"Output: {output}\nResponses: {responses}")
        for company in companies:
            print(company["registrationNumber"])
        return

    rows = load_rows(output, summaries=True) if output.exists() else {}
    if args.output is None or args.responses_dir is None:
        default_responses.mkdir(parents=True, exist_ok=True)
        marker = default_responses / ".model"
        try:
            with marker.open("x", encoding="utf-8") as file:
                file.write(model + "\n")
        except FileExistsError:
            if marker.read_text(encoding="utf-8").strip() != model:
                raise ValueError(f"{default_responses.name} is already assigned to another model. Run again.") from None

    for company in companies:
        registration = company["registrationNumber"]
        try:
            summary, cached = generate_summary(company, provider, model, responses, force=args.force)
        except (ValueError, LLMError) as error:
            raise ValueError(f"{registration}: {error}") from error
        rows[registration] = {"registrationNumber": registration, "summary": summary}
        atomic_write(output, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows.values()))
        status = "cached" if cached else f"{provider} generated"
        print(f"{registration}: {status}", flush=True)
    print(f"Saved {len(companies)} summaries to {output}")


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except (OSError, ValueError, LLMError) as error:
        raise SystemExit(f"error: {error}") from None


if __name__ == "__main__":
    main()
