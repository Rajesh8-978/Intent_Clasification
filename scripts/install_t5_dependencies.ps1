$ErrorActionPreference = "Stop"

Write-Host "Installing Open Intent Classifier T5 release..."
python -m pip install open-intent-classifier==0.0.2

Write-Host "Installing PDF runtime dependency..."
python -m pip install pypdf

Write-Host "Done. You can now run scripts/classify_pdf_email.py with --provider T5."
