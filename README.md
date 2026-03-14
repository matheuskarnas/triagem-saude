# 🏥 Triagem Saúde — API de Produção

API REST para classificação de urgência médica baseada em BERTimbau fine-tunado.

> Para visão geral do projeto e relatório técnico, veja o [repo principal](https://github.com/matheuskarnas/triagem-saude-projeto).
> Para reproduzir os experimentos e retreinar o modelo, veja o [repo de laboratório](https://github.com/matheuskarnas/triagem-saude-lab).

---

## 📁 Estrutura

```
triagem-saude/
├── src/
│   ├── preprocessing.py   ← pré-processamento de texto
│   ├── classifier.py      ← classificador com threshold adaptativo
│   └── api.py             ← endpoints FastAPI
├── models/
│   ├── README.md          ← instruções para obter o modelo
│   ├── classifier.pkl     ← modelo baseline TF-IDF + LR
│   └── vectorizer.pkl     ← vetorizador TF-IDF
├── tests/                 ← testes pytest
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## ⚙️ Requisitos

- Docker 20.10+ **ou** Python 3.10+
- 2GB de RAM disponível
- Modelo BERTimbau (~417MB) — veja [Como obter o modelo](#como-obter-o-modelo)

---

## 🚀 Setup e Execução

### Opção 1 — Docker (Recomendado)

```bash
git clone https://github.com/matheuskarnas/triagem-saude
cd triagem-saude

# Obtenha o modelo primeiro (veja seção abaixo)

# Cria o arquivo de ambiente (necessário para o docker-compose)
touch .env

# Sobe a API
docker-compose up --build
```

> ⚠️ Se aparecer erro `ContainerConfig` ao rodar `--build`, execute:
> ```bash
> docker-compose down && docker-compose up
> ```

### Opção 2 — Local

```bash
git clone https://github.com/matheuskarnas/triagem-saude
cd triagem-saude

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Obtenha o modelo primeiro (veja seção abaixo)

uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

A API estará disponível em `http://localhost:8000`.
Documentação interativa (Swagger) em `http://localhost:8000/docs`.

---

## 📦 Como Obter o Modelo

O modelo BERTimbau fine-tunado (~417MB) não está versionado no Git por limitações de tamanho.

**Opção 1 — Retreinar (~15 minutos com GPU):**

```bash
git clone https://github.com/matheuskarnas/triagem-saude-lab
cd triagem-saude-lab

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 src/train_bert.py \
  --data data/processed/dataset_balanceado.csv \
  --output models/ \
  --epochs 5 \
  --batch_size 16

cp -r models/bertimbau_triagem ../triagem-saude/models/
```

**Opção 2 — Contato direto:** solicite o modelo serializado ao autor.

---

## 🔌 API Reference

### `POST /predict`

Classifica um relato de sintomas.

**Request:**
```json
{ "texto": "dor no peito e falta de ar" }
```

**Response:**
```json
{
  "label": "URGENTE",
  "label_num": 2,
  "confianca": 0.9081,
  "alerta": "🔴 Procure atendimento de emergência imediatamente!",
  "threshold_urgente": 0.35
}
```

| Campo | Descrição |
|---|---|
| `label` | `URGENTE`, `MODERADO` ou `LEVE` |
| `label_num` | 2, 1 ou 0 |
| `confianca` | Probabilidade da classe predita (0–1) |
| `alerta` | Mensagem orientativa para o paciente |
| `threshold_urgente` | Threshold usado para a classe URGENTE |

**Lógica de threshold:**

```
se P(URGENTE) >= 0.35 → classifica como URGENTE
senão               → argmax das probabilidades
```

Threshold 0.35 foi escolhido para maximizar o Recall da classe URGENTE (92.9%), priorizando segurança do paciente sobre precisão.

### `GET /health`

```json
{ "status": "ok", "model": "loaded" }
```

---

## ✅ Verificar funcionamento

```bash
# Health check
curl http://localhost:8000/health

# Teste URGENTE
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "dor no peito e falta de ar"}'

# Teste LEVE
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "tosse leve há dois dias sem febre"}'
```

---

## 📊 Métricas do Modelo

| Métrica | Valor |
|---|---|
| Acurácia geral | **87%** |
| Recall URGENTE | **92.9%** |
| F1 URGENTE | **0.90** |
| Threshold URGENTE | **0.35** |
