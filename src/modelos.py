"""
Criação dos modelos e do Pipeline final (pré-processamento + modelo).
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Liga o nome usado no config.py à classe do scikit-learn
MODELOS_DISPONIVEIS = {
    "MLPClassifier": MLPClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
}


def criar_modelo(nome_modelo, hiperparametros, semente):
    """Cria o modelo do scikit-learn com os hiperparâmetros do config.py."""
    classe_modelo = MODELOS_DISPONIVEIS[nome_modelo]
    if nome_modelo == "MLPClassifier":
        # solver="adam": otimizador que usa a taxa de aprendizado (learning_rate_init)
        # activation="relu": função de ativação da camada oculta
        return classe_modelo(solver="adam", activation="relu",
                             random_state=semente, **hiperparametros)
    return classe_modelo(random_state=semente, **hiperparametros)


def montar_pipeline(preprocessador, modelo):
    """
    Junta o pré-processador (já ajustado com o treino) e o modelo (já treinado)
    num único objeto. É ele que é salvo no MLflow: recebe os dados originais
    (letras) e devolve a previsão.
    """
    return Pipeline([
        ("preprocessamento", preprocessador),
        ("modelo", modelo),
    ])
