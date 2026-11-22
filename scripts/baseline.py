"""Generate factual, deterministic company summaries as an evaluation baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def money(value: float | int) -> str:
    """Format a ringgit amount compactly without hiding its scale."""
    absolute = abs(value)
    sign = "-" if value < 0 else ""
    if absolute >= 1_000_000:
        return f"{sign}RM{absolute / 1_000_000:,.2f}m"
    if absolute >= 1_000:
        return f"{sign}RM{absolute / 1_000:,.1f}k"
    return f"{sign}RM{absolute:,.0f}"


def profile_paragraph(company: dict[str, Any]) -> str:
    profile = company["profile"]
    company_name = profile["companyName"]
    former_names = profile.get("formerNames") or []
    if former_names:
        previous_names = ", ".join(item["name"] for item in former_names)
        company_name += f" (formerly known as {previous_names})"

    return (
        f"**Company:** {company_name} is a {profile['companyType'].lower()} incorporated on "
        f"{profile['incorporationDate']}, currently {profile['status'].lower()}. It operates in the "
        f"{profile['industry']} industry and is located in {profile['state']}. Its stated business is "
        f"{profile['natureOfBusiness'].rstrip('.').lower()}. Share capital is {money(profile['shareCapital'])}."
    )


def relationships_paragraph(company: dict[str, Any]) -> str:
    relationships = company["relationships"]
    directors = relationships.get("directors") or []
    shareholders = relationships.get("shareholders") or []

    director_label = "director" if len(directors) == 1 else "directors"
    shareholder_label = "shareholder" if len(shareholders) == 1 else "shareholders"
    text = (
        f"**Director & Shareholder:** The company has {len(directors)} {director_label} "
        f"and {len(shareholders)} {shareholder_label}."
    )
    if not shareholders:
        return text

    largest = max(shareholders, key=lambda shareholder: shareholder["percentage"])
    return f"{text} The largest shareholder is {largest['name']} at {largest['percentage']:.2f}%."


def financials_paragraph(company: dict[str, Any]) -> str:
    years = company["financials"].get("years") or []
    if not years:
        return "**Financials:** No financial years are provided."

    ordered = sorted(years, key=lambda year: year["financialYear"])
    latest = ordered[-1]
    statements = latest["statements"]
    metrics = latest["ratiosAndMetrics"]
    profitability = metrics["profitability"]
    liquidity = metrics["liquidityAndCashFlow"]

    previous = ordered[-2] if len(ordered) > 1 else None
    if previous:
        previous_revenue = previous["statements"]["revenue"]
        if previous_revenue:
            yoy_growth = ((statements["revenue"] - previous_revenue) / abs(previous_revenue)) * 100
            if yoy_growth > 0:
                revenue_text = f"revenue grew {yoy_growth:.1f}% YoY"
            elif yoy_growth < 0:
                revenue_text = f"revenue declined {abs(yoy_growth):.1f}% YoY"
            else:
                revenue_text = "revenue was flat YoY"
        else:
            revenue_text = "YoY revenue growth was unavailable because prior-year revenue was zero"
    else:
        revenue_text = "YoY revenue growth was unavailable because no prior financial year was provided"

    free_cash_flow = liquidity["freeCashFlow"]
    previous_free_cash_flow = previous["ratiosAndMetrics"]["liquidityAndCashFlow"]["freeCashFlow"] if previous else None
    if free_cash_flow > 0:
        if previous_free_cash_flow is None:
            cash_flow_text = "was positive"
        elif previous_free_cash_flow > 0:
            cash_flow_text = "remained positive"
        else:
            cash_flow_text = "turned positive"
    elif free_cash_flow < 0:
        if previous_free_cash_flow is None:
            cash_flow_text = "was negative"
        elif previous_free_cash_flow < 0:
            cash_flow_text = "remained negative"
        else:
            cash_flow_text = "turned negative"
    else:
        cash_flow_text = "was break-even"

    net_debt = statements["borrowings"] - statements["cash"]

    return (
        f"**Financials:** In FY{latest['financialYear']}, {revenue_text}, with a gross profit margin of "
        f"{profitability['grossProfitMargin'] * 100:.1f}% and an operating profit margin of "
        f"{profitability['operatingProfitMargin'] * 100:.1f}%. Free cash flow {cash_flow_text} at "
        f"{money(free_cash_flow)}, while net debt stood at {money(net_debt)}."
    )


def litigation_paragraph(company: dict[str, Any]) -> str:
    cases = company["litigation"].get("cases") or []
    if not cases:
        return "**Litigation:** No court cases were recorded."

    defendant_cases = [case for case in cases if case["role"].lower() == "defendant"]
    plaintiff_cases = [case for case in cases if case["role"].lower() == "plaintiff"]
    defendant_claims = sum((case.get("claimAmount") or 0) for case in defendant_cases)
    plaintiff_claims = sum((case.get("claimAmount") or 0) for case in plaintiff_cases)

    status_counts: dict[str, int] = {}
    for case in cases:
        status = case["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    status_phrases = []
    status_order = ("Judgment entered", "Pending", "Withdrawn", "Dismissed", "Settled")
    for status in status_order:
        count = status_counts.get(status, 0)
        if not count:
            continue
        if status == "Judgment entered":
            status_phrases.append(f"{count} {'has judgment' if count == 1 else 'have judgments'} entered")
        elif status == "Pending":
            status_phrases.append(f"{count} {'is' if count == 1 else 'are'} pending")
        else:
            status_phrases.append(f"{count} {'was' if count == 1 else 'were'} {status.lower()}")

    statuses = status_phrases[0] if len(status_phrases) == 1 else ", ".join(status_phrases[:-1]) + f", and {status_phrases[-1]}"

    case_label = "case was" if len(cases) == 1 else "cases were"
    return (
        f"**Litigation:** {len(cases)} {case_label} recorded. {len(defendant_cases)} as defendant "
        f"({money(defendant_claims)} claimed) and {len(plaintiff_cases)} as plaintiff "
        f"({money(plaintiff_claims)} claimed). Of these, {statuses}."
    )


def summarise(company: dict[str, Any]) -> str:
    return "\n\n".join(
        (
            profile_paragraph(company),
            relationships_paragraph(company),
            financials_paragraph(company),
            litigation_paragraph(company),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/companies.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("summaries/baseline.jsonl"))
    parser.add_argument("--limit", type=int, help="Only process the first N records")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    companies = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        companies = companies[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for company in companies:
            row = {"registrationNumber": company["registrationNumber"], "summary": summarise(company)}
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(companies)} summaries to {args.output}")


if __name__ == "__main__":
    main()
