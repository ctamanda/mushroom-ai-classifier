"""
Carregamento e exploração dos dados (base Mushroom - UCI).

Como executar (a partir da pasta raiz do projeto, com o .venv ativado):
    python -m src.exploracao
"""

import pandas as pd

from src.config import CAMINHO_DADOS

# Nomes oficiais das colunas (arquivo agaricus-lepiota.names da UCI):
# 1 coluna alvo ("class") + 22 atributos = 23 colunas
NOMES_COLUNAS = [
    "class",                     # alvo: e = comestível (edible), p = venenoso (poisonous)
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",                # contém valores ausentes representados por "?"
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]


def carregar_dados(caminho=CAMINHO_DADOS):
    """Lê o arquivo .data (sem cabeçalho) e devolve um DataFrame com os nomes oficiais."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado: {caminho}\n"
            "Baixe agaricus-lepiota.data da UCI e coloque na pasta data/."
        )
    return pd.read_csv(
        caminho,
        header=None,            # o arquivo não tem linha de cabeçalho
        names=NOMES_COLUNAS,    # usamos os nomes oficiais
        dtype=str,              # todos os atributos são categóricos (letras)
    )


def validar_dados(df):
    """Confere se o arquivo foi lido como esperado. Interrompe a execução se algo estiver errado."""
    if df.shape[1] != len(NOMES_COLUNAS):
        raise ValueError(f"Esperadas {len(NOMES_COLUNAS)} colunas, encontradas {df.shape[1]}.")
    if df.empty:
        raise ValueError("O arquivo de dados está vazio.")
    classes = set(df["class"].unique())
    if classes != {"e", "p"}:
        raise ValueError(f"Classes inesperadas na coluna 'class': {classes}")
    return {
        "registros": len(df),
        "colunas": df.shape[1],
        "valores_ausentes_?": int((df == "?").sum().sum()),
        "venenosos_%": round((df["class"] == "p").mean() * 100, 2),
    }


def main():
    df = carregar_dados()

    print("=" * 60)
    print("1) DIMENSÕES")
    print("=" * 60)
    linhas, colunas = df.shape
    print(f"Linhas: {linhas} | Colunas: {colunas}")
    print("Registros esperados: 8124 ->", "OK" if linhas == 8124 else "DIFERENTE!")
    print("Colunas esperadas: 23 (1 alvo + 22 atributos) ->", "OK" if colunas == 23 else "DIFERENTE!")

    print("\n" + "=" * 60)
    print("2) PRIMEIRAS LINHAS")
    print("=" * 60)
    print(df.head())

    print("\n" + "=" * 60)
    print("3) TIPOS DE DADOS")
    print("=" * 60)
    print(df.dtypes.value_counts())

    print("\n" + "=" * 60)
    print("4) VALORES AUSENTES")
    print("=" * 60)
    print("Ausentes no formato padrão (NaN):", int(df.isna().sum().sum()))
    interrogacoes = (df == "?").sum()
    print("Valores '?' por coluna (apenas colunas com '?'):")
    print(interrogacoes[interrogacoes > 0])

    print("\n" + "=" * 60)
    print("5) DISTRIBUIÇÃO DAS CLASSES (e = comestível, p = venenoso)")
    print("=" * 60)
    contagem = df["class"].value_counts()
    percentual = df["class"].value_counts(normalize=True) * 100
    print(pd.DataFrame({"quantidade": contagem, "percentual (%)": percentual.round(2)}))

    print("\n" + "=" * 60)
    print("6) QUANTIDADE DE CATEGORIAS DIFERENTES POR COLUNA")
    print("=" * 60)
    print(df.nunique().sort_values())


if __name__ == "__main__":
    main()
