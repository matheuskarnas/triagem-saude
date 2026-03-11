"""
api.py
------
API REST de triagem médica.
Endpoints:
    POST /predict  — classifica relato de sintomas
    GET  /health   — verifica status da API e modelo
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
from src.classifier import classifier


# ─── Modelos Pydantic ─────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    texto: str

    @field_validator("texto")
    @classmethod
    def texto_nao_vazio(cls, v):
        if not v or not v.strip():
            raise ValueError("O campo 'texto' não pode ser vazio.")
        if len(v.strip()) < 3:
            raise ValueError("O campo 'texto' deve ter pelo menos 3 caracteres.")
        return v.strip()


class PredictResponse(BaseModel):
    label:                  str
    label_num:              int
    confianca:              float
    alerta:                 str
    threshold_emergencia:   float


class HealthResponse(BaseModel):
    status: str
    model:  str


# ─── Lifespan — carrega modelo na inicialização ───────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo ao iniciar a API."""
    try:
        classifier.load()
        print("✅ Modelo carregado com sucesso!")
    except FileNotFoundError as e:
        print(f"❌ Erro ao carregar modelo: {e}")
    yield


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Triagem Saúde API",
    description="Sistema inteligente de triagem médica — classifica sintomas em EMERGENCIA, URGENTE ou NAO_URGENTE.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health():
    """Verifica se a API e o modelo estão funcionando."""
    return {
        "status": "ok",
        "model":  "loaded" if classifier._loaded else "not loaded",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Triagem"])
def predict(request: PredictRequest):
    """
    Classifica um relato de sintomas em nível de urgência.

    - **EMERGENCIA**: Risco imediato de vida — procure emergência agora
    - **URGENTE**: Atenção em até 24h
    - **NAO_URGENTE**: Consulta agendada
    """
    if not classifier._loaded:
        raise HTTPException(
            status_code=503,
            detail="Modelo não disponível. Tente novamente em instantes."
        )

    try:
        resultado = classifier.predict(request.texto)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")
