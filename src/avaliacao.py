"""
Avaliação dos modelos: métricas, matriz de confusão, relatório e curvas de treinamento.
"""

import matplotlib

matplotlib.use("Agg")  # gera as figuras sem abrir janelas
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import CLASSE_POSITIVA

NOMES_CLASSES = ["comestível (e)", "venenoso (p)"]  # índice 0 e 1


def calcular_metricas(y_real, y_previsto, prefixo):
    """
    Calcula as métricas considerando 'venenoso' (1) como classe positiva.
    prefixo = "val" (validação) ou "test" (teste final). Ex.: val_f1, test_recall.
    """
    return {
        f"{prefixo}_accuracy": accuracy_score(y_real, y_previsto),
        f"{prefixo}_precision": precision_score(y_real, y_previsto, pos_label=CLASSE_POSITIVA),
        f"{prefixo}_recall": recall_score(y_real, y_previsto, pos_label=CLASSE_POSITIVA),
        f"{prefixo}_f1": f1_score(y_real, y_previsto, pos_label=CLASSE_POSITIVA),
    }


def gerar_matriz_confusao(y_real, y_previsto, titulo):
    """Cria a figura da matriz de confusão (linhas = real, colunas = previsto)."""
    figura, eixo = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_real, y_previsto, display_labels=NOMES_CLASSES, cmap="Blues", ax=eixo, colorbar=False
    )
    eixo.set_title(titulo)
    eixo.set_xlabel("Previsto")
    eixo.set_ylabel("Real")
    figura.tight_layout()
    return figura


def gerar_relatorio(y_real, y_previsto):
    """Relatório de classificação (precisão, recall e F1 de cada classe) em texto."""
    return classification_report(y_real, y_previsto, target_names=NOMES_CLASSES, digits=4)


def gerar_curvas(historico, titulo):
    """Curvas de treinamento: loss (treino x validação) e acurácia por época."""
    epocas = [h["epoca"] for h in historico]
    figura, (eixo_loss, eixo_acc) = plt.subplots(1, 2, figsize=(10, 4))

    eixo_loss.plot(epocas, [h["train_loss"] for h in historico], label="train_loss")
    eixo_loss.plot(epocas, [h["val_loss"] for h in historico], label="val_loss")
    eixo_loss.set_title("Loss por época")
    eixo_loss.set_xlabel("época")
    eixo_loss.legend()

    eixo_acc.plot(epocas, [h["train_accuracy"] for h in historico], label="train_accuracy")
    eixo_acc.plot(epocas, [h["val_accuracy"] for h in historico], label="val_accuracy")
    eixo_acc.set_title("Acurácia por época")
    eixo_acc.set_xlabel("época")
    eixo_acc.legend()

    figura.suptitle(titulo)
    figura.tight_layout()
    return figura
