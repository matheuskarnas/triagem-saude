# 🏥 triagem-saude

Sistema inteligente de triagem médica que classifica relatos de sintomas em três níveis de urgência.

> Desenvolvido como solução ao desafio técnico de Engenheiro de IA — 2026.

---

## 📋 Níveis de Urgência

| Classe | Descrição | Exemplo |
|--------|-----------|---------|
| 🔴 EMERGENCIA | Risco imediato de vida | Dor no peito, falta de ar severa, AVC |
| 🟡 URGENTE | Atenção em até 24h | Febre alta, dor intensa, tontura |
| 🟢 NAO_URGENTE | Consulta agendada | Resfriado, dor leve, cansaço |

---

## 🚀 Como Rodar

### Pré-requisitos
- Docker e Docker Compose instalados

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/triagem-saude.git
cd triagem-saude
```

### 2. Configure as variáveis de ambiente
```bash
cp .env.example .env
```

### 3. Suba com Docker Compose
```bash
docker compose up --build
```

A API estará disponível em `http://localhost:8000`.

---

## 📡 Endpoints da API

### `POST /predict`
Classifica um relato de sintomas.

**Request:**
```json
{
  "texto": "dor no peito, falta de ar, tontura"
}
```

**Response:**
```json
{
  "label": "EMERGENCIA",
  "label_num": 2,
  "confianca": 0.94,
  "alerta": "⚠️ Procure atendimento imediato!"
}
```

### `GET /health`
Verifica se a API está no ar.

```json
{ "status": "ok", "model": "loaded" }
```

---

## 🗂️ Estrutura do Projeto

```
triagem-saude/
├── src/
│   ├── api.py            # FastAPI — endpoints
│   ├── classifier.py     # Lógica de predição
│   └── preprocessing.py  # Limpeza e preparação do texto
├── models/               # Modelo serializado
├── tests/                # Testes unitários
├── data/
│   └── processed/        # Dataset limpo (sem dados brutos)
├── docs/                 # Documentação adicional
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 🧪 Testes

```bash
pip install pytest
pytest tests/ -v
```

---

## ⚠️ Decisão de Threshold

Por se tratar de um sistema de saúde, o modelo prioriza **recall da classe EMERGENCIA** acima de acurácia global. O threshold de decisão foi ajustado para minimizar falsos negativos críticos (paciente urgente classificado como leve).

Detalhes no [relatório técnico](docs/relatorio_tecnico.md).

---

## 🔗 Repositório de Experimentação

Todo o processo de EDA, treinamento e avaliação de modelos está documentado em:
👉 [triagem-saude-lab](https://github.com/seu-usuario/triagem-saude-lab)
