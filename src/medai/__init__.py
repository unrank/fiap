"""
medai - Base do sistema inteligente de suporte ao diagnostico.

Pacote desenvolvido para o Tech Challenge - Fase 1 (POSTECH).

Modulos
-------
config         Caminhos, constantes e semente aleatoria do projeto.
data           Carga, auditoria e persistencia do dataset clinico tabular.
eda            Analise exploratoria e geracao de figuras.
preprocessing  Pipeline de pre-processamento (numerico + categorico).
models         Catalogo de modelos e grades de hiperparametros.
evaluate       Metricas, curvas, matriz de confusao e ajuste de limiar.
interpret      Feature importance, permutation importance e SHAP.
cnn            EXTRA - classificacao de radiografias de torax com CNN.
"""

__version__ = "1.0.0"
__all__ = [
    "config",
    "data",
    "eda",
    "preprocessing",
    "models",
    "evaluate",
    "interpret",
]
