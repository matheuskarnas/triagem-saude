# Modelos

O modelo BERTimbau fine-tunado (~417MB) não está versionado no Git por limitações de tamanho.

## Como obter o modelo

**Opção 1 — Retreinar do zero:**
```bash
cd ../triagem-saude-lab
python3 src/train_bert.py \
  --data data/processed/dataset_balanceado.csv \
  --output models/ \
  --epochs 5 \
  --batch_size 16

cp -r models/bertimbau_triagem ../triagem-saude/models/
```

**Opção 2 — Contato:**
Solicite o modelo serializado diretamente ao autor.

## Métricas do modelo atual

| Métrica | Valor |
|---|---|
| Acurácia geral | 87% |
| Recall URGENTE | 92.9% |
| F1 URGENTE | 0.90 |
| Threshold URGENTE | 0.35 |
