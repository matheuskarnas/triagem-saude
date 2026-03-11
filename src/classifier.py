"""
classifier.py
-------------
Carrega o modelo treinado e realiza predições com threshold ajustado
para priorizar recall da classe EMERGENCIA.
"""

import os
import joblib
import numpy as np
from pathlib import Path
from src.preprocessing import limpar_texto

# ─── Configuração ─────────────────────────────────────────────────────────────

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

# Threshold de decisão para EMERGENCIA
# Valor menor = mais sensível (menos falsos negativos)
# Padrão: 0.35 — ajustável via variável de ambiente
EMERGENCIA_THRESHOLD = float(os.getenv("EMERGENCIA_THRESHOLD", "0.35"))

CLASSES = ["EMERGENCIA", "URGENTE", "NAO_URGENTE"]

ALERTAS = {
    "EMERGENCIA":  "🔴 Procure atendimento de emergência imediatamente!",
    "URGENTE":     "🟡 Procure atendimento médico em até 24 horas.",
    "NAO_URGENTE": "🟢 Agende uma consulta médica.",
}


# ─── Classe do Classificador ──────────────────────────────────────────────────

class TriagemClassifier:
    """Classificador de triagem médica com threshold ajustável para EMERGENCIA."""

    def __init__(self):
        self.model      = None
        self.vectorizer = None
        self._loaded    = False

    def load(self):
        """Carrega modelo e vectorizer do disco."""
        model_path      = MODELS_DIR / "classifier.pkl"
        vectorizer_path = MODELS_DIR / "vectorizer.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer não encontrado em: {vectorizer_path}")

        self.model      = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self._loaded    = True

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
        vetor = self.vectorizer.transform([texto_limpo])

        # Probabilidades por classe
        proba = self.model.predict_proba(vetor)[0]
        classes_model = self.model.classes_

        # Mapear probabilidades para dict
        proba_dict = {c: float(p) for c, p in zip(classes_model, proba)}

        # Aplicar threshold especial para EMERGENCIA
        # Se a probabilidade de EMERGENCIA supera o threshold, classifica como tal
        if proba_dict.get("EMERGENCIA", 0) >= EMERGENCIA_THRESHOLD:
            label = "EMERGENCIA"
        else:
            label = classes_model[np.argmax(proba)]

        return {
            "label":      label,
            "label_num":  CLASSES.index(label),
            "confianca":  round(proba_dict[label], 4),
            "alerta":     ALERTAS[label],
            "threshold_emergencia": EMERGENCIA_THRESHOLD,
        }


# Instância global — carregada uma vez na inicialização da API
classifier = TriagemClassifier()
