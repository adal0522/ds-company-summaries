# Company Summaries Case Study

## What a good summary is ?

A good summary helps a reviewer decide which potential suppliers or partners to examine first. It should be easy to read and understand in the company record. I focus on what the company does, who controls it, how its finances are changing, and whether it brought or faces any court claims. Missing information and unusual audit opinions should be easy to spot.

## What I built

I built a Python baseline for all 50 companies and a reusable prompt template for the LLM summaries. The template defines the same five sections and rating rules for every run. I can switch models or providers with `--model` and `--provider` while keeping the prompt and company records the same.

I ran the template with two models on Groq. The saved outputs are GPT-OSS 120B in [groq-v1](summaries/groq-v1.jsonl) and GPT-OSS 20B in [groq-v2](summaries/groq-v2.jsonl). I cache the raw responses so the script can reuse a valid response for the same model, prompt, and company record without spending more API tokens. The models calculate the figures from the raw records, and the script checks the summary format before writing JSONL for the viewer.

The 1–5 score prioritises further supplier review. A strong candidate scores 5, missing financials alone can warrant 3, and a company in winding up scores 1. I used Codex to help refactor the Python code, check the saved outputs, and draft this report.

## How to know whether it's good

I chose [eight companies](samples/representative-8.txt) to cover cases such as missing financials, falling revenue, mixed litigation roles, and winding up. Both Groq model versions covered all eight, and all 16 summaries passed the format check. I then spot-checked claims against the original records and used the baseline as a cross-check.

| Company | GPT-OSS 120B rating | GPT-OSS 20B rating | What I checked |
| --- | --- | --- | --- |
| SYN-000217 · Teguh Bestari Learning | 4/5 | 5/5 | Net debt was RM1.95m after rounding. The 20B model matched it, while the 120B model reported RM1.90m. Both omitted the FY2023 audit disclaimer. |
| SYN-000333 · Gemilang Harmoni Enterprise | 3/5 | 3/5 | Both flagged missing financials. The source contains no financial years. |
| SYN-001203 · Pacific Seri Freight | 3/5 | 3/5 | Both correctly reported one settled plaintiff case with a RM22.5k claim and no defendant cases. |
| SYN-000451 · Mutiara Delima Bakery | 2/5 | 2/5 | Both separated the plaintiff and defendant claims correctly. The 20B model labelled 2/5 as “High-risk review priority” instead of “Low review priority.” |
| SYN-000545 · Cahaya Golden Systems | 1/5 | 1/5 | Both counted eight defendant cases and applied the 1/5 winding-up rule. |
| SYN-000640 · Teguh Bestari Medical | 4/5 | 4/5 | Both got the case counts right. Five plaintiff claims totalled RM4.92m after rounding. The 20B model matched that total, while the 120B model reported RM4.90m. |
| SYN-001298 · Mutiara Delima Agro | 2/5 | 2/5 | Both said year-on-year growth was unavailable. The 120B model matched free cash flow of −RM6.64m, while the 20B model reported −RM6.60m. |
| SYN-000476 · Kirana Everbright Packaging | 2/5 | 4/5 | Revenue fell about 20% and free cash flow stayed positive. The models weighed those signals differently, producing the largest rating gap in the sample. |

Ratings agreed on six of eight companies. The spot checks found numerical and labelling errors that the format check missed. 

## What I will do next,and how I improve it

My first priority would be to make the summaries more reliable. I would use Python to calculate net debt and litigation totals, then pass those figures to the model. I would also flag unusual audit opinions so they are easier to include in the summary.

I would add examples to the rating rubric for companies with mixed signals. For example, Kirana Everbright Packaging had falling revenue but positive free cash flow, and the two models gave it 2/5 and 4/5. I would clarify how to weigh those signals and add a check that each score uses the correct label.

I would try models from other providers using the same prompt and eight company records. After each prompt change, I would rerun those cases and track factual errors, important omissions, and rating inconsistencies. I would then test the revised prompt on a fresh sample to see whether the improvements hold.
