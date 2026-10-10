"""
Preparação dos dados (base Mushroom - UCI).

Como executar (a partir da pasta raiz do projeto, com o .venv ativado):
    python -m src.preparacao
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import PROPORCAO_TESTE, PROPORCAO_VALIDACAO, SEMENTE
from src.exploracao import carregar_dados


def preparar_dados(df):
    """Remove colunas inúteis e separa X (atributos) de y (alvo)."""
    # veil-type tem o mesmo valor em todas as linhas: não ajuda a diferenciar nada
    df = df.drop(columns=["veil-type"])  # drop devolve uma cópia: o original não muda

    # X = características do cogumelo | y = o que queremos prever
    X = df.drop(columns=["class"])
    # 1 = venenoso (p) -> classe positiva | 0 = comestível (e)
    y = (df["class"] == "p").astype(int)
    return X, y


def dividir_dados(X, y, proporcao_validacao=PROPORCAO_VALIDACAO,
                  proporcao_teste=PROPORCAO_TESTE, semente=SEMENTE):
    """
    Divide em treino, validação e teste, mantendo a proporção de venenosos/comestíveis.
      - treino:    o modelo aprende com ele
      - validação: compara as configurações e escolhe a melhor
      - teste:     usado UMA vez, só no final, com o modelo já escolhido
    """
    # 1º separa o teste e o deixa guardado
    X_resto, X_teste, y_resto, y_teste = train_test_split(
        X, y, test_size=proporcao_teste, stratify=y, random_state=semente
    )
    # 2º divide o restante em treino e validação
    # (ex.: 0.2 / 0.8 = 25% do restante = 20% do total)
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(
        X_resto, y_resto,
        test_size=proporcao_validacao / (1 - proporcao_teste),
        stratify=y_resto,       # preserva a proporção das classes
        random_state=semente,
    )
    return X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste


def criar_preprocessador(colunas):
    """
    Cria o pré-processamento aplicado a todas as colunas categóricas:
      1) troca "?" (valor ausente) pela categoria "ausente";
      2) OneHotEncoder: transforma cada categoria (letra) em colunas de 0 e 1.
    Ele só aprende alguma coisa quando chamamos fit() com os dados de TREINO.
    """
    tratamento_categorico = Pipeline([
        # "?" significa "não se sabe". Em vez de apagar ~30% das linhas,
        # tratamos como uma categoria própria chamada "ausente".
        ("trata_ausentes", SimpleImputer(
            missing_values="?", strategy="constant", fill_value="ausente"
        )),
        # handle_unknown="ignore": se aparecer na validação/teste uma categoria que
        # não existia no treino, o codificador não dá erro (preenche tudo com 0)
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("categoricas", tratamento_categorico, list(colunas)),
    ])


def main():
    df = carregar_dados()
    X, y = preparar_dados(df)

    print("=" * 60)
    print("1) SEPARAÇÃO X (atributos) e y (alvo)")
    print("=" * 60)
    print(f"X: {X.shape[0]} linhas x {X.shape[1]} colunas (sem 'class' e sem 'veil-type')")
    print(f"y: {y.shape[0]} valores (1 = venenoso, 0 = comestível)")

    X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste = dividir_dados(X, y)

    print("\n" + "=" * 60)
    print("2) DIVISÃO TREINO / VALIDAÇÃO / TESTE")
    print("=" * 60)
    for nome, parte in [("completo", y), ("treino", y_treino),
                        ("validação", y_validacao), ("teste", y_teste)]:
        print(f"{nome:<10} {len(parte):>5} linhas | {parte.mean() * 100:.2f}% venenosos")

    # O pré-processador aprende as categorias APENAS com o treino (fit),
    # depois só aplica (transform) na validação e no teste -> evita vazamento de dados
    preprocessador = criar_preprocessador(X.columns)
    X_treino_cod = preprocessador.fit_transform(X_treino)
    X_validacao_cod = preprocessador.transform(X_validacao)

    print("\n" + "=" * 60)
    print("3) TRATAMENTO DO '?' + CODIFICAÇÃO ONE-HOT")
    print("=" * 60)
    print(f"Treino codificado:    {X_treino_cod.shape[0]} linhas x {X_treino_cod.shape[1]} colunas")
    print(f"Validação codificada: {X_validacao_cod.shape[0]} linhas x {X_validacao_cod.shape[1]} colunas")
    nomes = preprocessador.get_feature_names_out()
    print("Colunas criadas para stalk-root:", [n for n in nomes if "stalk-root" in n])


if __name__ == "__main__":
    main()
