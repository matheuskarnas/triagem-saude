"""
classifier.py
-------------
Carrega o modelo BERTimbau fine-tuned e realiza predições com threshold
ajustado para priorizar recall da classe EMERGENCIA.
"""

import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing import limpar_texto

# ─── Configuração ─────────────────────────────────────────────────────────────

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
BERT_MODEL_DIR = MODELS_DIR / "bertimbau_triagem"

# Threshold de decisão para EMERGENCIA
# Valor menor = mais sensível (menos falsos negativos)
# Padrão: 0.35 — ajustável via variável de ambiente
URGENTE_THRESHOLD = float(os.getenv("URGENTE_THRESHOLD", "0.35"))

# Mapeamento índice → label (deve coincidir com o treinamento)
ID2LABEL = {0: "LEVE", 1: "MODERADO", 2: "URGENTE"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

ALERTAS = {
    "URGENTE":  "🔴 Procure atendimento de emergência imediatamente!",
    "MODERADO": "🟡 Procure atendimento médico em até 24 horas.",
    "LEVE":     "🟢 Agende uma consulta médica.",
}

# Usar GPU se disponível
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Classe do Classificador ──────────────────────────────────────────────────

class TriagemClassifier:
    """Classificador de triagem médica com BERTimbau e threshold ajustável."""

    def __init__(self):
        self.model     = None
        self.tokenizer = None
        self._loaded   = False

    def load(self):
        """Carrega modelo BERTimbau e tokenizer do disco."""
        if not BERT_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Modelo BERTimbau não encontrado em: {BERT_MODEL_DIR}\n"
                "Execute o fine-tuning antes de iniciar a API."
            )

        print(f"🤖 Carregando BERTimbau de {BERT_MODEL_DIR}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(BERT_MODEL_DIR))
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            str(BERT_MODEL_DIR)
        )
        self.model.to(DEVICE)
        self.model.eval()
        self._loaded = True
        print(f"✅ Modelo carregado no dispositivo: {DEVICE}")

    def predict(self, texto: str) -> dict:
        """
        Classifica um relato de sintomas.

        Args:
            texto: Texto livre com sintomas do paciente.

        Returns:
            Dicionário com label, label_num, confiança e alerta.
        """
        if not self._loaded:
            raise RuntimeError("Modelo não carregado. Chame load() primeiro.")

        texto_limpo = limpar_texto(texto)

        # Tokenizar
        inputs = self.tokenizer(
            texto_limpo,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Inferência sem gradiente
        with torch.no_grad():
            outputs = self.model(**inputs)
            proba = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        # Mapear probabilidades para dict
        proba_dict = {ID2LABEL[i]: float(p) for i, p in enumerate(proba)}

        # Aplicar threshold especial para EMERGENCIA
        if proba_dict.get("URGENTE", 0) >= URGENTE_THRESHOLD:
            label = "URGENTE"
        else:
            label = ID2LABEL[int(np.argmax(proba))]

        return {
            "label":                label,
            "label_num":            LABEL2ID[label],
            "confianca":            round(proba_dict[label], 4),
            "alerta":               ALERTAS[label],
            "threshold_urgente": URGENTE_THRESHOLD,
        }


# Instância global — carregada uma vez na inicialização da API
classifier = TriagemClassifier()
