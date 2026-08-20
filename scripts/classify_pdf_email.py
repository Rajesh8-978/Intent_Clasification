"""Command-line entry point for classifying one text-based PDF email."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Allow the script to import the package when it is run directly from scripts/.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intent_classification import (
    IntentClassificationService,
    IntentClassifierOptions,
    ZeroShotIntentClassifier,
)
from intent_classification.config import DEFAULT_MODEL
from intent_classification.pdf_text_extractor import extract_text_from_pdf


def positive_int(value: str) -> int:
    """Argparse converter that rejects invalid TopK values early."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


async def main() -> int:
    """Build the classification pipeline and print its JSON result."""

    parser = argparse.ArgumentParser(
        description="Classify the business intent of a PDF email."
    )
    parser.add_argument("pdf", help="Path to the PDF email.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local Hugging Face zero-shot model name.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional Transformers device, for example 'cpu' or 'cuda:0'.",
    )
    parser.add_argument("--top-k", type=positive_int, default=3, help="Number of top candidates.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Keep command-line configuration in the same validated object used by the API.
    options = IntentClassifierOptions(
        model=args.model,
        top_k=args.top_k,
        device=args.device,
    )
    classifier = ZeroShotIntentClassifier(
        model_name=options.model,
        device=options.device,
    )
    service = IntentClassificationService(classifier, options=options)

    # PDF extraction is kept separate from classification so text can also be
    # supplied directly through IntentClassificationService in other applications.
    email_text = extract_text_from_pdf(args.pdf)
    result = await service.classify_email_text(email_text)

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
