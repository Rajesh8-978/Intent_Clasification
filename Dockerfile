# syntax=docker/dockerfile:1

# A slim CPU image is portable across local machines and cloud container services.
FROM python:3.12-slim

ARG MODEL_ID=MoritzLaurer/deberta-v3-base-zeroshot-v2.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    INTENT_MODEL_ID=${MODEL_ID}

WORKDIR /app

# Install the CPU-only PyTorch wheel first to avoid unnecessary GPU libraries.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
    && python -m pip install -r requirements.txt

# Copy only the runtime package and command-line entry point.
COPY intent_classification ./intent_classification
COPY scripts ./scripts

# Run as an unprivileged user in deployed environments.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Cache the default model in the image so runtime classification needs no download.
RUN python -c "import os; from transformers import AutoModelForSequenceClassification, AutoTokenizer; model = os.environ['INTENT_MODEL_ID']; AutoTokenizer.from_pretrained(model); AutoModelForSequenceClassification.from_pretrained(model)"

# Prevent optional Hugging Face metadata checks after the model is embedded.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["python", "scripts/classify_pdf_email.py"]
CMD ["--help"]
