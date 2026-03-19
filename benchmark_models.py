import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from summarizer import SUMMARY_STOPWORDS, TextSummarizer


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    genre: str
    expected_style: str
    text: str
    reference_summary: str
    fact_groups: list[list[str]]


BENCHMARK_CASES = [
    BenchmarkCase(
        case_id="news_transit_fire",
        genre="news",
        expected_style="expository",
        text=(
            "City officials said an electrical fire near the central rail hub disrupted the morning commute on Thursday. "
            "The fire damaged signal cables and forced crews to close two underground platforms for inspections. "
            "Transit staff redirected passengers to shuttle buses, but many riders still reported delays of about 40 minutes. "
            "Firefighters contained the smoke within half an hour and no injuries were reported. "
            "The transport agency expects normal train service to resume before the evening rush after emergency repairs."
        ),
        reference_summary=(
            "An electrical fire at the central rail hub damaged signal cables, closed two platforms, and caused morning train delays of about 40 minutes. "
            "No injuries were reported and officials expect service to return before the evening commute."
        ),
        fact_groups=[
            ["fire", "electrical"],
            ["rail", "train", "hub"],
            ["signal", "cables"],
            ["delay", "40", "minutes"],
            ["no", "injuries"],
            ["evening", "service", "resume"],
        ],
    ),
    BenchmarkCase(
        case_id="expository_urban_trees",
        genre="expository",
        expected_style="expository",
        text=(
            "Urban planners increasingly plant trees because shade lowers surface temperatures on streets and sidewalks during heat waves. "
            "Tree roots also help soil absorb rainwater, which reduces flash flooding after heavy storms. "
            "Researchers in three districts found that blocks with mature trees used less electricity for air conditioning in nearby buildings. "
            "However, planners note that trees only deliver those benefits when cities budget for watering, pruning, and pest control. "
            "As a result, the most effective programs combine new planting with long-term maintenance rather than one-time campaigns."
        ),
        reference_summary=(
            "Urban trees lower heat, reduce flooding, and can cut cooling demand, but those benefits depend on long-term maintenance such as watering and pruning."
        ),
        fact_groups=[
            ["trees", "urban"],
            ["heat", "shade", "temperature"],
            ["rainwater", "flooding", "storm"],
            ["electricity", "cooling", "air", "conditioning"],
            ["maintenance", "watering", "pruning", "pest"],
        ],
    ),
    BenchmarkCase(
        case_id="howto_release_checklist",
        genre="how-to",
        expected_style="structured",
        text=(
            "Release checklist:\n"
            "1. Freeze the main branch and confirm that all pull requests are merged.\n"
            "2. Run the automated test suite and fix any failing integration checks before tagging a build.\n"
            "3. Update the changelog so customer-facing fixes appear in the release notes.\n"
            "4. Deploy the build to staging and ask support to verify login, billing, and export workflows.\n"
            "5. After approval, publish the production release and monitor error dashboards for 30 minutes."
        ),
        reference_summary=(
            "The release process freezes the main branch, runs tests, updates the changelog, validates the build in staging, then publishes to production and monitors errors."
        ),
        fact_groups=[
            ["freeze", "main", "branch"],
            ["test", "suite", "integration"],
            ["changelog", "release", "notes"],
            ["staging", "verify", "support"],
            ["production", "monitor", "error", "dashboards"],
        ],
    ),
    BenchmarkCase(
        case_id="faq_subscription",
        genre="faq",
        expected_style="structured",
        text=(
            "FAQ:\n"
            "Q: When does the annual subscription renew?\n"
            "A: The annual plan renews automatically on the same calendar date each year unless billing is cancelled first.\n"
            "Q: Can finance teams request invoices?\n"
            "A: Yes. Workspace owners can download invoices from the billing page after each payment is processed.\n"
            "Q: What happens if a card fails?\n"
            "A: The system retries the payment for three days, then switches the workspace to read-only mode until the card is updated."
        ),
        reference_summary=(
            "The FAQ explains that annual subscriptions renew automatically each year, invoices are available on the billing page, and failed cards are retried for three days before the workspace becomes read-only."
        ),
        fact_groups=[
            ["annual", "subscription", "renews", "year"],
            ["invoice", "billing", "page"],
            ["card", "fails", "payment", "retry"],
            ["three", "days"],
            ["read-only", "workspace"],
        ],
    ),
    BenchmarkCase(
        case_id="conversation_incident_review",
        genre="conversation",
        expected_style="conversational",
        text=(
            "Why did the checkout outage last so long? Maya asked during the incident review. "
            "Liam said the on-call engineer restarted the payment worker, but the queue kept filling because a retry rule doubled failed requests. "
            "Maya replied that the alert only showed CPU usage and did not mention the growing queue length. "
            "The team agreed to add a queue-depth alarm, document the rollback steps, and rehearse the response before the next release."
        ),
        reference_summary=(
            "In the incident review, the team said a bad retry rule kept the payment queue growing, while monitoring failed to highlight queue depth. "
            "They agreed to add a queue alarm, document rollback steps, and practice the response."
        ),
        fact_groups=[
            ["retry", "rule"],
            ["payment", "queue"],
            ["alert", "alarm", "queue-depth"],
            ["rollback", "steps"],
            ["rehearse", "practice", "response"],
        ],
    ),
    BenchmarkCase(
        case_id="narrative_new_companion",
        genre="narrative",
        expected_style="narrative",
        text=(
            "Emma lived alone above her bookstore and followed the same routine every day. "
            "Emma opened the shop at sunrise, sorted old novels, and ate lunch in silence by the window. "
            "Later Emma adopted a nervous dog named Milo from the city shelter. "
            "Milo waited by the register, then slowly started greeting regular customers before Emma could speak. "
            "After a few weeks Noah began stopping by each afternoon to help Emma carry donations upstairs. "
            "Emma and Noah started walking Milo by the river, and the quiet routine of the shop changed. "
            "By autumn, Emma said the store felt warmer because Milo drew people in and Noah stayed for dinner. "
            "In the end, Emma realized that the dog had turned the bookstore into a place full of company instead of silence."
        ),
        reference_summary=(
            "Emma's lonely routine changes after she adopts Milo, whose presence draws in customers and brings Noah into her daily life. "
            "By the end, Emma sees the bookstore as a warm, shared place rather than a silent one."
        ),
        fact_groups=[
            ["Emma"],
            ["Milo", "dog"],
            ["Noah"],
            ["routine", "changed", "change"],
            ["bookstore", "warmer", "company", "shared"],
        ],
    ),
]


def normalize_terms(text: str) -> list[str]:
    return [
        term for term in TextSummarizer.__new__(TextSummarizer)._content_words(text)
        if term not in SUMMARY_STOPWORDS
    ]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_terms(prediction)
    ref_tokens = normalize_terms(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = {}
    ref_counter = {}
    for token in pred_tokens:
        pred_counter[token] = pred_counter.get(token, 0) + 1
    for token in ref_tokens:
        ref_counter[token] = ref_counter.get(token, 0) + 1

    overlap = sum(min(pred_counter.get(token, 0), ref_counter.get(token, 0)) for token in set(pred_counter) | set(ref_counter))
    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(ref_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def fact_coverage(summary: str, fact_groups: Iterable[Iterable[str]]) -> float:
    lowered_summary = summary.lower()
    groups = list(fact_groups)
    if not groups:
        return 0.0

    covered = 0
    for group in groups:
        if any(term.lower() in lowered_summary for term in group):
            covered += 1
    return covered / len(groups)


def sentence_support(summarizer: TextSummarizer, summary: str, source_text: str) -> float:
    summary_sentences = summarizer.split_into_sentences(summary)
    source_sentences = summarizer.split_into_sentences(source_text)
    if not summary_sentences or not source_sentences:
        return 0.0
    return statistics.mean(
        summarizer._sentence_support_score(sentence, source_sentences)
        for sentence in summary_sentences
    )


def bounded_score(value: float) -> float:
    return max(0.0, min(value, 1.0))


def evaluate_case(summarizer: TextSummarizer, case: BenchmarkCase) -> dict:
    result = summarizer.summarize(case.text, max_length=100, min_length=35)
    summary = result["final_summary"]
    detected_style = summarizer._build_document_profile(case.text).style

    reference_score = token_f1(summary, case.reference_summary)
    fact_score = fact_coverage(summary, case.fact_groups)
    support_score = sentence_support(summarizer, summary, case.text)
    style_score = 1.0 if detected_style == case.expected_style else 0.0
    accuracy = bounded_score(
        (reference_score * 0.45)
        + (fact_score * 0.3)
        + (support_score * 0.15)
        + (style_score * 0.1)
    )

    return {
        "case_id": case.case_id,
        "genre": case.genre,
        "expected_style": case.expected_style,
        "detected_style": detected_style,
        "summary": summary,
        "reference_score": round(reference_score, 4),
        "fact_score": round(fact_score, 4),
        "support_score": round(support_score, 4),
        "style_score": round(style_score, 4),
        "accuracy": round(accuracy, 4),
    }


def run_model(model_name: str) -> dict:
    summarizer = TextSummarizer(model_name=model_name, max_tokens=1024)
    case_results = [evaluate_case(summarizer, case) for case in BENCHMARK_CASES]

    return {
        "model_name": model_name,
        "average_accuracy": round(statistics.mean(item["accuracy"] for item in case_results), 4),
        "average_reference_score": round(statistics.mean(item["reference_score"] for item in case_results), 4),
        "average_fact_score": round(statistics.mean(item["fact_score"] for item in case_results), 4),
        "average_support_score": round(statistics.mean(item["support_score"] for item in case_results), 4),
        "style_detection_accuracy": round(statistics.mean(item["style_score"] for item in case_results), 4),
        "cases": case_results,
    }


def main() -> None:
    models = ["t5-small", "facebook/bart-large-cnn"]
    report = {
        "models": [run_model(model_name) for model_name in models],
    }

    output_path = Path("benchmark_report.json")
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()