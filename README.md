# Intent Classification

Python utilities for classifying the business intent of email text extracted from PDF files. The package returns a business-friendly intent label and maps that label to the application `EntityType` used by downstream workflows.

## What It Does

This project supports a simple classification pipeline:

```text
PDF email
  -> text extraction
  -> text cleanup
  -> intent classification
  -> business label
  -> EntityType mapping
  -> workflow routing
```

Example output:

```json
{
  "entityType": "MiscRequest",
  "intentLabel": "Miscellaneous Request",
  "confidence": null,
  "topPredictions": [
    {
      "label": "Miscellaneous Request",
      "score": null
    }
  ]
}
```

Some local providers return only the winning label. In those cases `confidence` and `score` are `null`.

## Supported Intent Labels

The default label set is mapped in `intent_classification/entity_types.py` and currently includes:

- Statement of Affairs
- Payment Request
- Cheque Deposit Request
- Travel Application
- Insurance Policy
- Vehicle Documents
- Real Estate
- Company Shares
- Case Trustee
- Case Business
- Business Entity
- Case Creditor
- Asset
- Cash Asset
- Prospect
- General Case
- Email
- Miscellaneous Request

## Project Structure

```text
intent_classification/
  classifiers/              Provider adapters
  config.py                 Runtime configuration
  entity_types.py           Label to EntityType mapping
  factory.py                Classifier factory
  pdf_text_extractor.py     Text-based PDF extraction
  service.py                Intent classification service
scripts/
  classify_pdf_email.py     CLI for classifying a PDF
  install_t5_dependencies.ps1
tests/                      Unit tests
```

## Requirements

- Python 3.10 or newer
- Windows PowerShell for the included install helper
- Text-based PDFs for direct PDF classification

Scanned or image-only PDFs need OCR before classification. After OCR, pass the extracted text into `IntentClassificationService.classify_email_text(...)`.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the default PDF/T5 runtime:

```powershell
python -m pip install -r requirements.txt
```

Or use the Windows helper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_t5_dependencies.ps1
```

For the OpenAI provider, install the optional OpenAI dependency:

```powershell
python -m pip install -r requirements.openai-intent-classification.txt
```

## Classify A PDF

Run this from the project root:

```powershell
python scripts\classify_pdf_email.py "C:\path\to\email.pdf"
```

Example with your local PDF:

```powershell
python scripts\classify_pdf_email.py "C:\Users\Admin\Downloads\Vehicle - 2 (1).pdf"
```

Use the larger FLAN-T5 model when accuracy matters more than speed:

```powershell
python scripts\classify_pdf_email.py "C:\path\to\email.pdf" --provider T5 --model flan-t5-base
```

Other supported provider names:

```powershell
python scripts\classify_pdf_email.py "C:\path\to\email.pdf" --provider Embedding
python scripts\classify_pdf_email.py "C:\path\to\email.pdf" --provider SmolLM2
python scripts\classify_pdf_email.py "C:\path\to\email.pdf" --provider OpenAI --model gpt-4.1-mini
```

## Use In Python

```python
import asyncio

from intent_classification import IntentClassificationService, IntentClassifierOptions
from intent_classification.factory import create_intent_classifier


async def main() -> None:
    options = IntentClassifierOptions(
        provider="T5",
        model="small",
        top_k=3,
    )
    classifier = create_intent_classifier(options)
    service = IntentClassificationService(classifier, options=options)

    result = await service.classify_email_text("Extracted email text goes here")
    print(result.to_dict())


asyncio.run(main())
```

Configuration can also be loaded from a dictionary:

```python
settings = {
    "IntentClassifier": {
        "Provider": "T5",
        "Model": "small",
        "TopK": 3,
        "ProviderOptions": {
            "device": "cpu"
        }
    }
}

options = IntentClassifierOptions.from_dict(settings)
```

## Testing

The unit tests use mocked classifiers, so they do not require downloading model packages.

```powershell
python -m unittest discover -v
```

Optional syntax check:

```powershell
python -m compileall intent_classification tests
```

## Configuration

Default configuration:

```json
{
  "IntentClassifier": {
    "Provider": "T5",
    "Model": "small",
    "TopK": 3
  }
}
```

Available options:

- `Provider`: `T5`, `Embedding`, `OpenAI`, or `SmolLM2`
- `Model`: provider-specific model name
- `TopK`: number of candidates to keep, must be at least `1`
- `FailOnUnknownLabel`: fail when a provider returns an unmapped label, defaults to `true`
- `ProviderOptions`: provider-specific settings such as `{ "device": "cpu" }` or `{ "api_key": "..." }`

For OpenAI, you can also set `OPENAI_API_KEY` in your environment and omit `api_key` from `ProviderOptions`.

## Add Or Change Labels

Business labels are defined in `intent_classification/entity_types.py`.

To add a label:

1. Add the business-friendly label to `LABEL_TO_ENTITY_TYPE`.
2. Map it to the correct `EntityType`.
3. Add or update tests if the label is part of a required workflow.

No model retraining is required for the default dynamic-label providers.

## Notes

- `open-intent-classifier==0.0.2` is pinned for the T5/PDF path because it avoids a newer optional dependency chain that can require Rust/Cargo on Windows.
- Direct PDF classification works best for PDFs that already contain selectable text.
- Keep secrets such as API keys out of Git. Use environment variables or local-only `.env` files.
