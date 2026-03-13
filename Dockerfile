FROM python:3.11-slim

# Diretório de trabalho
WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python primeiro (cache eficiente)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código e modelo
COPY src/ ./src/
COPY models/ ./models/

# Usuário não-root por segurança
RUN useradd -m appuser
USER appuser

# Expor porta
EXPOSE 8000

# Iniciar API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
