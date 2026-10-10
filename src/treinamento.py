"""
Treinamento dos modelos.

A MLP é treinada época por época: a cada época medimos o erro (loss) no treino
e na validação, e a acurácia na validação. Essas métricas são registradas no
MLflow com step = número da época, gerando as curvas de treinamento.
"""

import mlflow
from sklearn.metrics import accuracy_score, log_loss


def treinar_mlp_por_epocas(modelo, X_treino, y_treino, X_validacao, y_validacao, epocas):
    """
    Treina a MLP por `epocas` épocas com partial_fit (1 chamada = 1 passada pelo treino).
    Devolve o histórico com as métricas de cada época.
    """
    historico = []
    for epoca in range(1, epocas + 1):
        modelo.partial_fit(X_treino, y_treino, classes=[0, 1])

        prob_treino = modelo.predict_proba(X_treino)
        prob_validacao = modelo.predict_proba(X_validacao)
        metricas = {
            # loss = entropia cruzada (log loss): quanto menor, melhor
            "train_loss": log_loss(y_treino, prob_treino, labels=[0, 1]),
            "val_loss": log_loss(y_validacao, prob_validacao, labels=[0, 1]),
            "train_accuracy": accuracy_score(y_treino, prob_treino.argmax(axis=1)),
            "val_accuracy": accuracy_score(y_validacao, prob_validacao.argmax(axis=1)),
        }
        mlflow.log_metrics(metricas, step=epoca)
        historico.append({"epoca": epoca, **metricas})
    return historico


def treinar_modelo_simples(modelo, X_treino, y_treino):
    """Árvore e Random Forest: treinam de uma vez só (não existem épocas)."""
    modelo.fit(X_treino, y_treino)
