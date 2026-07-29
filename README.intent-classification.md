# Email Intent Classification

This module classifies already-extracted PDF email text into a business-friendly intent label and maps that label back to the application `EntityType`.

## Pipeline

```text
PDF
  -> OCR / text extraction
  -> text cleaning
  -> intent classification
  -> predicted business label
  -> EntityType mapping
  -> business workflow
```

## Default configuration

```json
{
  "IntentClassifier": {
    "Provider": "T5",
    "Model": "small",
    "TopK": 3
  }
}
```

Use the larger FLAN-T5 model when accuracy matters more than latency:

```json
{
  "IntentClassifier": {
    "Provider": "T5",
    "Model": "flan-t5-base",
    "TopK": 3,
    "ProviderOptions": {
      "device": "cpu"
    }
  }
}
```

## Usage

From the project root, install the T5/PDF packages once:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_t5_dependencies.ps1
```

If you prefer to run the install manually:

```powershell
python -m pip install -r requirements.intent-classification.txt
```

Use `open-intent-classifier==0.0.2` for this T5 path on Windows. Newer releases can pull an optional DSPy/LiteLLM dependency chain that requires Rust/Cargo.

```python
from intent_classification import IntentClassifierOptions, IntentClassificationService
from intent_classification.factory import create_intent_classifier

settings = {
    "IntentClassifier": {
        "Provider": "T5",
        "Model": "small",
        "TopK": 3,
    }
}

options = IntentClassifierOptions.from_dict(settings)
classifier = create_intent_classifier(options)
service = IntentClassificationService(classifier, options=options)

result = await service.classify_email_text(extracted_email_text)
payload = result.to_dict()
```

To classify a text-based PDF directly from the terminal:

```powershell
python scripts/classify_pdf_email.py "C:\path\to\email.pdf" --provider T5 --model small
```

For the larger FLAN-T5 model:

```powershell
python scripts/classify_pdf_email.py "C:\path\to\email.pdf" --provider T5 --model flan-t5-base
```

If the PDF is scanned or image-only, run OCR first and pass the extracted text into `classify_email_text(...)`.

Example response:

```json
{
  "entityType": "StatementOfAffairs",
  "intentLabel": "Statement of Affairs",
  "confidence": 0.94,
  "topPredictions": [
    {
      "label": "Statement of Affairs",
      "score": 0.94
    }
  ]
}
```

The default T5 adapter uses `open_intent_classifier.model.IntentClassifier`. The package currently returns a predicted label from its documented `predict(text, labels)` API, so confidence and ranked candidates are returned only when the selected provider exposes them.

## Adding labels

Add a business-friendly label to `LABEL_TO_ENTITY_TYPE` in `intent_classification/entity_types.py`, mapped to the correct `EntityType`. No retraining is required because labels are supplied dynamically at prediction time.

For tenant-specific or runtime-managed labels, provide a custom `IIntentLabelProvider` implementation and inject it into `IntentClassificationService`.

## Dependencies

Install only the providers you enable:

For T5 on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_t5_dependencies.ps1
```

For the OpenAI provider:

```powershell
python -m pip install -r requirements.openai-intent-classification.txt
```

The tests use mocked classifiers and do not require model packages.

On Windows, prefer `open-intent-classifier==0.0.2` for the T5 path. Newer releases may pull optional DSPy/LiteLLM dependencies that require Rust/Cargo during install.
