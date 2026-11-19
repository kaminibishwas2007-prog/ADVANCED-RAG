import json
import re
import time

from pathlib import Path
from pipeline import AdvancedRAGPipeline


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "evaluation" / "test_cases.json"

REPORT_FOLDER = BASE_DIR / "evaluation" / "reports"

REPORT_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================
# TEXT NORMALIZATION
# ==========================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    return set(text.split())


# ==========================================
# RETRIEVAL PRECISION
# ==========================================

def calculate_retrieval_precision(
    retrieved_sources,
    expected_sources
):

    if not retrieved_sources:
        return 0.0

    retrieved = set(retrieved_sources)

    expected = set(expected_sources)

    relevant = retrieved.intersection(expected)

    precision = len(relevant) / len(retrieved)

    return round(precision * 100, 2)


# ==========================================
# SOURCE RECALL
# ==========================================

def calculate_source_recall(
    retrieved_sources,
    expected_sources
):

    if not expected_sources:
        return 0.0

    retrieved = set(retrieved_sources)

    expected = set(expected_sources)

    relevant = retrieved.intersection(expected)

    recall = len(relevant) / len(expected)

    return round(recall * 100, 2)


# ==========================================
# ANSWER KEYWORD COVERAGE
# ==========================================

def calculate_keyword_coverage(
    answer,
    expected_keywords
):

    if not expected_keywords:
        return 0.0

    answer_words = normalize_text(answer)

    matched = 0

    for keyword in expected_keywords:

        keyword_words = normalize_text(keyword)

        if keyword_words and keyword_words.issubset(answer_words):

            matched += 1

    coverage = matched / len(expected_keywords)

    return round(coverage * 100, 2)


# ==========================================
# EVALUATION ENGINE
# ==========================================

class RAGEvaluator:

    def __init__(self):

        print("\nInitializing RAG Evaluation System...")

        self.rag = AdvancedRAGPipeline()

        print("Evaluation System Ready!")

    def evaluate(self):

        if not TEST_FILE.exists():

            print("Test dataset not found!")

            return

        with open(TEST_FILE, "r", encoding="utf-8") as file:

            test_cases = json.load(file)

        results = []

        print("\n====================================")

        print("ADVANCED RAG EVALUATION")

        print("====================================")

        for index, test in enumerate(test_cases, start=1):

            print(f"\nEvaluating Test Case {index}")

            question = test["question"]

            start_time = time.perf_counter()

            result = self.rag.ask(
                question=question,
                top_k=test.get("top_k", 4),
                max_distance=test.get("max_distance", 1.0)
            )

            end_time = time.perf_counter()

            response_time = round(
                end_time - start_time,
                3
            )

            answer = result["answer"]

            retrieved_sources = [
                source["filename"]
                for source in result["sources"]
            ]

            expected_sources = test["expected_sources"]

            expected_keywords = test["expected_keywords"]

            # Calculate evaluation metrics

            precision = calculate_retrieval_precision(
                retrieved_sources,
                expected_sources
            )

            recall = calculate_source_recall(
                retrieved_sources,
                expected_sources
            )

            keyword_coverage = calculate_keyword_coverage(
                answer,
                expected_keywords
            )

            # Store evaluation results

            evaluation_result = {

                "test_case": index,

                "question": question,

                "retrieved_sources": retrieved_sources,

                "retrieval_precision": precision,

                "source_recall": recall,

                "answer_keyword_coverage": keyword_coverage,

                "response_time_seconds": response_time,

                "retrieved_chunks": result["retrieved_chunks"]

            }

            results.append(evaluation_result)

            print("\nQuestion:", question)

            print("Retrieval Precision:", precision, "%")

            print("Source Recall:", recall, "%")

            print("Answer Keyword Coverage:", keyword_coverage, "%")

            print("Response Time:", response_time, "seconds")

            print("Retrieved Chunks:", result["retrieved_chunks"])

        # ==================================
        # GENERATE SUMMARY
        # ==================================

        total = len(results)

        if total == 0:

            print("No evaluation results generated.")

            return

        average_precision = round(
            sum(r["retrieval_precision"] for r in results) / total,
            2
        )

        average_recall = round(
            sum(r["source_recall"] for r in results) / total,
            2
        )

        average_coverage = round(
            sum(r["answer_keyword_coverage"] for r in results) / total,
            2
        )

        average_time = round(
            sum(r["response_time_seconds"] for r in results) / total,
            3
        )

        report = {

            "project": "Advanced RAG Pipeline",

            "total_test_cases": total,

            "average_retrieval_precision": average_precision,

            "average_source_recall": average_recall,

            "average_answer_keyword_coverage": average_coverage,

            "average_response_time_seconds": average_time,

            "test_results": results

        }

        # ==================================
        # SAVE REPORT
        # ==================================

        report_path = REPORT_FOLDER / "evaluation_report.json"

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=4
            )

        print("\n====================================")

        print("EVALUATION SUMMARY")

        print("====================================")

        print("Total Test Cases:", total)

        print("Average Retrieval Precision:", average_precision, "%")

        print("Average Source Recall:", average_recall, "%")

        print("Average Answer Keyword Coverage:", average_coverage, "%")

        print("Average Response Time:", average_time, "seconds")

        print("\nReport saved at:")

        print(report_path)


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    evaluator = RAGEvaluator()

    evaluator.evaluate()