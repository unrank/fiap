# =============================================================================
# Tech Challenge - Fase 1 | Sistema inteligente de suporte ao diagnostico
#
# Imagem reproduzivel do pipeline de Machine Learning.
#
#   Build:   docker build -t medai-fase1 .
#   Run:     docker run --rm -v "$(pwd)/reports:/app/reports" \
#                             -v "$(pwd)/models:/app/models" medai-fase1
#
# O dataset (Breast Cancer Wisconsin) ja vem embarcado no scikit-learn, logo
# o container roda o pipeline completo SEM acesso a internet.
# =============================================================================
FROM python:3.12-slim AS base

LABEL org.opencontainers.image.title="medai - Tech Challenge Fase 1"
LABEL org.opencontainers.image.description="Diagnostico de cancer de mama com Machine Learning"
LABEL org.opencontainers.image.version="1.0.0"

# Python previsivel dentro do container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app/src \
    OMP_NUM_THREADS=4

WORKDIR /app

# libgomp1 e exigido pelo scikit-learn/numba (OpenMP) na imagem slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Camada de dependencias separada: so e reconstruida quando o requirements muda.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Codigo da aplicacao.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY data/ ./data/

# Diretorios de saida (montaveis como volume para recuperar os resultados).
RUN mkdir -p reports/figures reports/metrics models data/raw

# Executa como usuario sem privilegios.
RUN useradd --create-home --uid 1000 medai \
    && chown -R medai:medai /app
USER medai

# Verifica que o ambiente esta integro antes de dar a imagem por pronta.
RUN python -c "import sklearn, pandas, shap, matplotlib; print('ambiente OK')"

# Padrao: roda o pipeline completo e escreve tudo em reports/ e models/.
# Para outro comando:
#   docker run --rm medai-fase1 python scripts/predict.py --demo
CMD ["python", "scripts/run_pipeline.py"]


# =============================================================================
# Estagio OPCIONAL - entregavel EXTRA de visao computacional (CNN + PyTorch).
# Imagem bem maior; construa apenas se for rodar a CNN:
#
#   docker build --target extra -t medai-fase1-cnn .
#   docker run --rm -v "$(pwd)/reports:/app/reports" medai-fase1-cnn
# =============================================================================
FROM base AS extra

USER root
# As versoes de referencia estao em requirements-extra.txt; aqui usamos o indice
# oficial de wheels CPU-only do PyTorch, bem menor que o pacote padrao com CUDA.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir medmnist \
    && chown -R medai:medai /app
USER medai

CMD ["python", "scripts/run_cnn.py", "--epochs", "12"]
