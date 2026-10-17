# PneumoVision inference service.
#
#   INSTALL_ML=false (default): mock mode only -- FastAPI, no torch (~200 MB).
#   INSTALL_ML=true: CPU-only torch + the serving subset of the training stack
#     (requirements-serve.txt), with the BioBERT and timm ViT-small/16-384
#     pretrained caches baked into HF_HOME=/opt/hf and used offline. Serves
#     INFERENCE_MODE=readers or fusion. No CUDA libraries: the target is a
#     2-vCPU EC2 instance.
#
# Checkpoints are never baked in (CLAUDE.md rule 2): they are mounted read-only
# at /app/checkpoints. .dockerignore keeps MIMIC-derived files out of the
# build context altogether.

ARG PYTHON_IMAGE=python:3.11-slim

# ── deps: a venv with everything installed, plus the pretrained caches ──────
FROM ${PYTHON_IMAGE} AS deps
ARG INSTALL_ML=false
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH=/opt/venv/bin:$PATH

RUN python -m venv /opt/venv && mkdir -p /opt/hf
COPY services/inference/requirements.txt services/inference/requirements-serve.txt /tmp/req/
# pytest is a dev dependency of requirements.txt; the image does not need it.
RUN grep -v '^pytest' /tmp/req/requirements.txt > /tmp/req/service.txt \
 && pip install -r /tmp/req/service.txt \
 && if [ "$INSTALL_ML" = "true" ]; then \
      pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1 \
      && pip install -r /tmp/req/requirements-serve.txt; \
    fi

COPY services/inference/app/offline.py /tmp/bake/app/offline.py
COPY services/inference/scripts/bake_caches.py /tmp/bake/scripts/bake_caches.py
RUN touch /tmp/bake/app/__init__.py \
 && if [ "$INSTALL_ML" = "true" ]; then python /tmp/bake/scripts/bake_caches.py; fi \
 && find /opt/venv -name '__pycache__' -prune -exec rm -rf {} + \
 && rm -rf /opt/hf/xet /opt/hf/hub/.locks

# ── runtime ──────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

COPY --from=deps /opt/venv /opt/venv
COPY --from=deps /opt/hf /opt/hf

WORKDIR /app
COPY services/inference/app ./app
# Preprocessing is imported from the training code rather than reimplemented,
# so the discretizer/normalizer stay byte-for-byte identical (Section 6).
COPY medpatch ./medpatch

RUN useradd --create-home --uid 1001 inference \
 && chown -R inference:inference /app \
 && chmod -R a+rX /opt/hf
USER inference

EXPOSE 8000
# 200 only when the configured models are loaded and warmed up; 503 while
# degraded (e.g. a checkpoint missing or failing its sha256 check).
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health').status==200 else 1)"

# One worker: the models are loaded once and are not fork-safe to copy.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
