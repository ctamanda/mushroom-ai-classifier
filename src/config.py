"""
Configuração central do projeto.

Para mudar um experimento, altere os valores aqui (ou use as opções
de linha de comando do main.py). O pipeline não precisa ser reescrito.
"""

from pathlib import Path

# Pasta raiz do projeto (a pasta que contém o main.py)
PASTA_PROJETO = Path(__file__).resolve().parent.parent

# Arquivo de dados original (nunca é alterado pelo código)
CAMINHO_DADOS = PASTA_PROJETO / "data" / "agaricus-lepiota.data"
ORIGEM_DADOS = "UCI Machine Learning Repository - Mushroom (agaricus-lepiota.data)"

# ---------------- Divisão treino / validação / teste ----------------
# 60% treino | 20% validação (escolha do modelo) | 20% teste (avaliação final)
PROPORCAO_VALIDACAO = 0.2
PROPORCAO_TESTE = 0.2
SEMENTE = 42            # semente fixa -> mesma divisão e mesmos resultados a cada execução

# ---------------- MLflow ----------------
# Usamos sempre o MESMO banco mlflow.db que já existe na raiz do projeto
URI_RASTREAMENTO = f"sqlite:///{(PASTA_PROJETO / 'mlflow.db').as_posix()}"
NOME_EXPERIMENTO_MLFLOW = "mushroom-aula08"

# Classe positiva das métricas binárias: 1 = venenoso (p)
# (errar um cogumelo venenoso é o erro mais perigoso)
CLASSE_POSITIVA = 1

# Descrição do pré-processamento (registrada em todas as runs)
PREPROCESSAMENTO = {
    "coluna_removida": "veil-type (valor único em todas as linhas)",
    "valores_ausentes": "'?' em stalk-root -> categoria 'ausente'",
    "codificacao": "OneHotEncoder(handle_unknown='ignore')",
    "ajustado_com": "somente o conjunto de treino",
}

# ---------------- Seleção do modelo ----------------
# A escolha usa SOMENTE a validação. O teste fica reservado para o final.
# 1º maior val_f1  ->  2º (desempate) menor val_loss  ->  3º ordem abaixo
CRITERIO_SELECAO = "maior val_f1; desempate: menor val_loss"

# ---------------- Experimentos ----------------
# As três MLPs partem da mesma configuração base e cada uma muda UM parâmetro,
# para que a comparação mostre o efeito daquele parâmetro.
# Árvore e Random Forest ficam como modelos de referência (não treinam por épocas
# e por isso não têm curvas train_loss/val_loss; não entram na seleção).
MLP_BASE = {
    "hidden_layer_sizes": (32,),   # arquitetura: 1 camada oculta com 32 neurônios
    "learning_rate_init": 0.001,   # taxa de aprendizado (otimizador Adam)
    "alpha": 0.0001,               # regularização L2
    "batch_size": 64,              # exemplos por atualização dos pesos
}

EXPERIMENTOS = {
    "mlp-base": {
        "nome_execucao": "mlp-base",
        "modelo": "MLPClassifier",
        "candidato": True,
        "epocas": 30,
        "pergunta": "Qual desempenho uma MLP simples alcança na validação?",
        "justificativa": "Configuração de referência para as outras MLPs.",
        "hiperparametros": dict(MLP_BASE),
    },
    "mlp-lr-alta": {
        "nome_execucao": "mlp-lr-alta",
        "modelo": "MLPClassifier",
        "candidato": True,
        "epocas": 30,
        "pergunta": "Uma taxa de aprendizado 10x maior acelera ou atrapalha o treino?",
        "justificativa": "Igual à base, mudando só a taxa de aprendizado (0.001 -> 0.01).",
        "hiperparametros": {**MLP_BASE, "learning_rate_init": 0.01},
    },
    "mlp-l2-alta": {
        "nome_execucao": "mlp-l2-alta",
        "modelo": "MLPClassifier",
        "candidato": True,
        "epocas": 30,
        "pergunta": "Uma regularização L2 mais forte muda o desempenho na validação?",
        "justificativa": "Igual à base, mudando só a regularização L2 (0.0001 -> 0.1).",
        "hiperparametros": {**MLP_BASE, "alpha": 0.1},
    },
    # Investigação do resultado inesperado (todas as runs chegam a 100%):
    # hipótese = o atributo "odor" sozinho já separa quase todas as classes.
    # Mesma configuração da mlp-base, mas SEM a coluna odor. Não entra na seleção.
    "mlp-sem-odor": {
        "nome_execucao": "investigacao-mlp-sem-odor",
        "modelo": "MLPClassifier",
        "candidato": False,
        "papel": "investigacao",
        "epocas": 30,
        "remover_colunas": ["odor"],
        "pergunta": "Os 100% dependem do atributo odor? O que acontece sem ele?",
        "justificativa": "Igual à mlp-base, removendo só a coluna odor dos dados.",
        "hiperparametros": dict(MLP_BASE),
    },
    "arvore": {
        "nome_execucao": "ref-arvore-decisao",
        "modelo": "DecisionTreeClassifier",
        "candidato": False,
        "pergunta": "Como uma Árvore de Decisão se sai como modelo de referência?",
        "justificativa": "Modelo simples e interpretável, usado como comparação.",
        "hiperparametros": {
            "criterion": "gini",       # medida usada para escolher as divisões
            "max_depth": 5,            # profundidade máxima (limita o tamanho da árvore)
        },
    },
    "floresta": {
        "nome_execucao": "ref-random-forest",
        "modelo": "RandomForestClassifier",
        "candidato": False,
        "pergunta": "Como uma Random Forest se sai como modelo de referência?",
        "justificativa": "Conjunto de árvores, usado como comparação.",
        "hiperparametros": {
            "n_estimators": 100,       # quantidade de árvores na floresta
            "max_depth": None,         # None = árvores crescem sem limite
            "max_features": "sqrt",    # atributos sorteados em cada divisão
        },
    },
}
