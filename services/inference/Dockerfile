FROM python:3.11-slim AS runtime

# INSTALL_ML=false (default) builds a small mock-mode image without torch.
# Set to true once real checkpoints exist to pull the full training stack.
ARG INSTALL_ML=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY services/inference/requirements.txt ./services-requirements.txt
COPY requirements.txt ./ml-requirements.txt
RUN pip install --upgrade pip \
 && pip install -r services-requirements.txt \
 && if [ "$INSTALL_ML" = "true" ]; then pip install -r ml-requirements.txt; fi

COPY services/inference/app ./app
# Preprocessing is imported from the training code rather than reimplemented,
# so the discretizer/normalizer stay byte-for-byte identical (Section 6).
COPY medpatch ./medpatch

# Checkpoints and normalizer state are mounted at runtime, never baked in.
RUN useradd --create-home --uid 1001 inference && chown -R inference:inference /app
USER inference

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
