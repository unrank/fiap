# Modelos treinados

Esta pasta guarda os artefatos gerados pelo treinamento. Eles **nao sao versionados**
no Git (ver `.gitignore`), por dois motivos:

1. arquivos `joblib`/`pickle` sao sensiveis a versao — um modelo salvo com
   scikit-learn 1.9 pode nao carregar em outra versao, e um artefato quebrado e
   pior do que artefato nenhum;
2. eles sao 100% reproduziveis a partir do codigo e da semente fixa (42).

## Como gerar

```bash
python scripts/run_pipeline.py      # -> models/modelo_diagnostico.joblib
python scripts/run_cnn.py           # -> models/cnn_pneumonia.pt   (EXTRA)
```

Depois disso, `python scripts/predict.py --demo` ja funciona.

## Conteudo de `modelo_diagnostico.joblib`

Um dicionario com o `Pipeline` completo (pre-processamento + estimador), o limiar
clinico calibrado, a ordem esperada das colunas, os hiperparametros escolhidos e as
metricas de teste registradas no momento do treino.
