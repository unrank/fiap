# Dados

## Tarefa principal - Breast Cancer Wisconsin (Diagnostic)

O arquivo `raw/breast_cancer_wisconsin.csv` e gravado automaticamente na primeira
execucao do pipeline. A ordem de busca implementada em `src/medai/data.py` e:

1. o proprio CSV local (se ja existir);
2. a copia oficial embarcada no scikit-learn (`sklearn.datasets.load_breast_cancer`) -
   identica a do UCI e disponivel **offline**, inclusive dentro do container Docker;
3. download direto do UCI.

Fontes originais:

- UCI ML Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

Formato: 569 linhas, 31 colunas (`diagnosis` + 30 preditoras numericas).
`diagnosis`: `M` = maligno, `B` = benigno.

## EXTRA - PneumoniaMNIST

Baixado automaticamente para `medmnist/` na primeira execucao de
`scripts/run_cnn.py` (poucos MB). Nao versionado no Git.

- MedMNIST v2: https://medmnist.com/
- Base original (Chest X-Ray Pneumonia, Kermany et al.):
  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
