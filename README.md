# Local PDF Intent Classification

Classify business emails stored as PDF files into one of 18 predefined intents. The application extracts the PDF text, runs an open-source zero-shot model locally, and maps the selected business label to the application's `EntityType`.

No paid LLM API or API key is required. An internet connection is needed only when the model is downloaded for the first time.

## How It Works

```text
PDF email
  -> extract selectable text
  -> normalize whitespace
  -> compare text with 18 detailed label definitions
  -> rank labels with confidence scores
  -> map the winning label to EntityType
  -> return JSON
```

The default model is [`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0). It is designed for zero-shot classification, which means labels can be added or refined without fine-tuning the model.

## Intent Labels

| Business label | EntityType | Typical content |
| --- | --- | --- |
| Statement of Affairs | `StatementOfAffairs` | Insolvency financial disclosures, assets, and liabilities |
| Payment Request | `PaymentRequest` | Payment, reimbursement, invoice, or approval requests |
| Cheque Deposit Request | `ChequeDepositRequest` | Depositing, clearing, or recording a cheque |
| Travel Application | `TravelApplication` | Travel approval, itinerary, accommodation, or expenses |
| Insurance Policy | `InsurancePolicy` | Policies, coverage, premiums, renewals, or claims |
| Vehicle Documents | `Vehicle` | Registration, COE, ownership, licensing, or vehicle records |
| Real Estate | `RealEstate` | Land, property, title, mortgage, sale, or valuation |
| Company Shares | `CompanyShares` | Shares, securities, certificates, dividends, or ownership |
| Case Trustee | `CaseTrustee` | Trustee appointment, authority, identity, or instructions |
| Case Business | `CaseBusiness` | Business operations or records associated with a case |
| Business Entity | `BusinessEntity` | Company registration, ownership, directors, or structure |
| Case Creditor | `CaseCreditor` | Proof of debt, creditor claims, or creditor details |
| Asset | `Asset` | General non-cash assets without a more specific category |
| Cash Asset | `AssetCash` | Cash, bank accounts, balances, deposits, or savings |
| Prospect | `Prospect` | New client enquiries, leads, or onboarding opportunities |
| General Case | `Case` | Case administration without a more specific case category |
| Email | `Email` | Email delivery, forwarding, mailbox, or attachment issues |
| Miscellaneous Request | `MiscRequest` | Requests that do not match another available category |

The detailed model-facing definitions live in `intent_classification/label_definitions.py`. The label-to-entity mappings live in `intent_classification/entity_types.py`.

## Requirements

- Python 3.10 or newer
- Windows PowerShell for the commands below
- A text-based PDF with selectable text
- Internet access for the first model download

The model runs on CPU by default. A compatible CUDA GPU can improve speed but is not required.

## Installation

From PowerShell:

```powershell
cd "C:\Users\Admin\OneDrive - Zest Labs\Documents\Data\intent_classification"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Classify A PDF

Run the classifier from the project root:

```powershell
python scripts\classify_pdf_email.py "C:\path\to\email.pdf"
```

Vehicle document example:

```powershell
python scripts\classify_pdf_email.py "C:\Users\Admin\Downloads\Vehicle - 2 (1).pdf"
```

Proof-of-debt example:

```powershell
python scripts\classify_pdf_email.py "C:\Users\Admin\Downloads\POD -2 (1).pdf"
```

The first classification downloads the model from Hugging Face. Later classifications use the cached local model.

## Output

```json
{
  "entityType": "Vehicle",
  "intentLabel": "Vehicle Documents",
  "confidence": 0.91,
  "topPredictions": [
    {
      "label": "Vehicle Documents",
      "score": 0.91
    },
    {
      "label": "Asset",
      "score": 0.05
    },
    {
      "label": "General Case",
      "score": 0.02
    }
  ]
}
```

The scores rank labels for the current document. They are useful for comparison, but they should not be treated as perfectly calibrated probabilities.

Use `--top-k` to change the number of returned candidates:

```powershell
python scripts\classify_pdf_email.py "C:\path\to\email.pdf" --top-k 5
```

Select a device explicitly when needed:

```powershell
python scripts\classify_pdf_email.py "C:\path\to\email.pdf" --device cpu
python scripts\classify_pdf_email.py "C:\path\to\email.pdf" --device cuda:0
```

## Python Usage

```python
import asyncio

from intent_classification import IntentClassificationService, IntentClassifierOptions
from intent_classification import ZeroShotIntentClassifier


async def main() -> None:
    options = IntentClassifierOptions(top_k=3, device="cpu")
    classifier = ZeroShotIntentClassifier(
        model_name=options.model,
        device=options.device,
    )
    service = IntentClassificationService(classifier, options=options)

    result = await service.classify_email_text(
        "Please provide the COE renewal and registration details for vehicle SMD4125Y."
    )
    print(result.to_dict())


asyncio.run(main())
```

Configuration can also be loaded from a dictionary:

```python
options = IntentClassifierOptions.from_dict(
    {
        "IntentClassifier": {
            "Model": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
            "TopK": 3,
            "Device": "cpu"
        }
    }
)
```

## Add Or Improve A Label

1. Add or update the label mapping in `intent_classification/entity_types.py`.
2. Add or improve its business description in `intent_classification/label_definitions.py`.
3. Add representative test cases for the new intent.
4. Test against a collection of real documents before deploying the change.

Clear descriptions matter. Include the business meaning, common document types, and distinguishing terminology. Keep `Email` and `Miscellaneous Request` narrow so they do not override more specific categories.

## Testing

The unit tests use an in-memory fake model, so they do not download model weights:

```powershell
python -m unittest discover -v
```

Optional syntax check:

```powershell
python -m compileall intent_classification tests scripts
```

## Project Structure

```text
intent_classification/
  classifiers/
    base.py                Classifier interface
    zero_shot.py           Local Hugging Face classifier
  config.py                Model, device, and TopK settings
  entity_types.py          Label-to-EntityType mapping
  label_definitions.py     Detailed meaning of every label
  labels.py                Label provider abstraction
  models.py                Result data structures
  pdf_text_extractor.py    PDF text extraction
  service.py               Validation and workflow coordination
  text_cleaning.py         Input normalization
scripts/
  classify_pdf_email.py    Command-line entry point
tests/                     Unit tests
requirements.txt           Runtime dependencies
```

## Troubleshooting

**No text could be extracted**

The PDF is probably scanned or image-only. Run OCR first, then classify the searchable PDF or pass the OCR text to `classify_email_text(...)`.

**First run is slow**

The model is being downloaded and loaded. Later runs use the local cache, although loading the model still takes some time.

**A generic label wins**

Review the extracted text and improve the relevant entry in `label_definitions.py`. Accuracy should be evaluated using real examples from every category.

**Out of memory**

Close other memory-heavy applications, run with `--device cpu`, or configure a smaller compatible zero-shot classification model with `--model`.
