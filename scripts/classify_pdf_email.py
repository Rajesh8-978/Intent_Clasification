from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intent_classification import IntentClassificationService, IntentClassifierOptions
from intent_classification.factory import create_intent_classifier
from intent_classification.pdf_text_extractor import extract_text_from_pdf


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify the business intent of a PDF email."
    )
    parser.add_argument("pdf", help="Path to the PDF email.")
    parser.add_argument(
        "--provider",
        default="T5",
        choices=["T5", "Embedding", "OpenAI", "SmolLM2"],
        help="Classifier provider.",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Model name. Use 'flan-t5-base' for the higher-accuracy T5 model.",
    )
    parser.add_argument("--top-k", type=positive_int, default=3, help="Number of top candidates.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    options = IntentClassifierOptions(
        provider=args.provider,
        model=args.model,
        top_k=args.top_k,
    )
    classifier = create_intent_classifier(options)
    service = IntentClassificationService(classifier, options=options)

    email_text = extract_text_from_pdf(args.pdf)
    result = await service.classify_email_text(email_text)

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
