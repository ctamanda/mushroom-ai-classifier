"""
Pipeline de treino, validação e teste com MLflow (base Mushroom - UCI).

Como executar (a partir da pasta raiz do projeto, com o .venv ativado):
    python main.py                                  -> todas as runs + seleção + teste final
    python main.py --experimentos mlp-base mlp-lr-alta
    python main.py --epocas 50                      -> muda as épocas de todas as MLPs
    python main.py --sem-teste-final                -> só treino e validação
    python main.py --help                           -> mostra todas as opções

Fluxo:
  1) cada configuração vira uma run: treina com TREINO e mede na VALIDAÇÃO
  2) a melhor MLP é escolhida SOMENTE pela validação
  3) só a escolhida é avaliada no TESTE, numa run separada "avaliacao-final"
"""

import argparse
import copy
import json
import platform
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn

from src import config
from src.avaliacao import calcular_metricas, gerar_curvas, gerar_matriz_confusao, gerar_relatorio
from src.exploracao import carregar_dados, validar_dados
from src.modelos import criar_modelo, montar_pipeline
from src.preparacao import criar_preprocessador, dividir_dados, preparar_dados
from src.treinamento import treinar_mlp_por_epocas, treinar_modelo_simples

# O MLflow salva o modelo no formato seguro "skops" e pede para declarar
# os tipos internos usados pelos modelos. É seguro: nós mesmos treinamos o modelo.
TIPOS_CONFIAVEIS_SKOPS = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
]

# Avisos informativos do MLflow que não afetam os resultados
warnings.filterwarnings("ignore", message="The specified dataset source can be interpreted")
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")


# ---------------------------------------------------------------- configuração
def ler_argumentos():
    """Opções de linha de comando. Sem opções, usa os valores do src/config.py."""
    parser = argparse.ArgumentParser(description="Treina, seleciona e avalia modelos na base Mushroom com MLflow.")
    parser.add_argument("--experimentos", nargs="+", choices=list(config.EXPERIMENTOS),
                        default=list(config.EXPERIMENTOS),
                        help="quais configurações executar (padrão: todas)")
    parser.add_argument("--epocas", type=int, help="número de épocas das MLPs")
    parser.add_argument("--lr", type=float, help="taxa de aprendizado das MLPs escolhidas")
    parser.add_argument("--l2", type=float, help="regularização L2 (alpha) das MLPs escolhidas")
    parser.add_argument("--arquitetura", help="neurônios por camada oculta das MLPs, ex.: 64,32")
    parser.add_argument("--semente", type=int, default=config.SEMENTE,
                        help=f"semente aleatória (padrão: {config.SEMENTE})")
    parser.add_argument("--proporcao-validacao", type=float, default=config.PROPORCAO_VALIDACAO,
                        help=f"fração dos dados para validação (padrão: {config.PROPORCAO_VALIDACAO})")
    parser.add_argument("--proporcao-teste", type=float, default=config.PROPORCAO_TESTE,
                        help=f"fração dos dados para teste (padrão: {config.PROPORCAO_TESTE})")
    parser.add_argument("--sem-teste-final", action="store_true",
                        help="não avalia no teste (útil enquanto se testa configurações)")
    return parser.parse_args()


def montar_experimentos(args):
    """Copia os experimentos do config.py e aplica as alterações vindas da linha de comando."""
    alteracoes = {}
    if args.epocas is not None:
        alteracoes["epocas"] = args.epocas
    if args.lr is not None:
        alteracoes["learning_rate_init"] = args.lr
    if args.l2 is not None:
        alteracoes["alpha"] = args.l2
    if args.arquitetura is not None:
        alteracoes["hidden_layer_sizes"] = tuple(int(n) for n in args.arquitetura.split(","))

    experimentos = []
    for chave in args.experimentos:
        exp = copy.deepcopy(config.EXPERIMENTOS[chave])
        exp["alteracoes_cli"] = {}
        if exp["modelo"] == "MLPClassifier":
            for nome, valor in alteracoes.items():
                if nome == "epocas":
                    exp["epocas"] = valor
                else:
                    exp["hiperparametros"][nome] = valor
                exp["alteracoes_cli"][nome] = valor
        experimentos.append(exp)
    return experimentos


# ---------------------------------------------------------------- etapas
def preparar_divisao(args):
    """Carrega, valida, separa X/y e divide em treino/validação/teste (sempre a mesma divisão)."""
    df = carregar_dados()
    resumo = validar_dados(df)
    X, y = preparar_dados(df)
    partes = dividir_dados(X, y, args.proporcao_validacao, args.proporcao_teste, args.semente)
    return df, resumo, partes


def registrar_dados(df, resumo, partes, args):
    """Registra no MLflow a origem dos dados, tamanhos, proporções e pré-processamento."""
    X_treino, X_validacao, X_teste = partes[:3]
    mlflow.log_params({
        "dados_origem": config.ORIGEM_DADOS,
        "registros_total": resumo["registros"],
        "registros_treino": len(X_treino),
        "registros_validacao": len(X_validacao),
        "registros_teste": len(X_teste),
        "proporcao_validacao": args.proporcao_validacao,
        "proporcao_teste": args.proporcao_teste,
        "semente": args.semente,
        "classe_positiva": "p (venenoso)",
        **{f"preproc_{k}": v for k, v in config.PREPROCESSAMENTO.items()},
    })
    # Registra o dataset (só esquema e resumo; o arquivo em si não é copiado)
    dataset = mlflow.data.from_pandas(df, source=str(config.CAMINHO_DADOS),
                                      name="mushroom-uci", targets="class")
    mlflow.log_input(dataset, context="dataset_completo")


def resumo_reproducao(exp, args, id_execucao):
    """Tudo o que é preciso para repetir esta run (vira o artefato configuracao.json)."""
    return {
        "id_execucao": id_execucao,
        "comando": "python " + " ".join(sys.argv),
        "experimento": {k: v for k, v in exp.items()},
        "semente": args.semente,
        "proporcao_validacao": args.proporcao_validacao,
        "proporcao_teste": args.proporcao_teste,
        "preprocessamento": config.PREPROCESSAMENTO,
        "criterio_selecao": config.CRITERIO_SELECAO,
        "versoes": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "mlflow": mlflow.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }


def executar_experimento(exp, args, id_execucao):
    """Uma run do MLflow: treina com TREINO e avalia na VALIDAÇÃO (o teste não é usado aqui)."""
    nome = exp["nome_execucao"]
    eh_mlp = exp["modelo"] == "MLPClassifier"

    with mlflow.start_run(run_name=nome) as run:
        mlflow.set_tags({
            "mlflow.note.content": f"{exp['pergunta']}\n\n{exp['justificativa']}",
            "pergunta": exp["pergunta"],
            "justificativa": exp["justificativa"],
            "id_execucao": id_execucao,
            "papel": exp.get("papel", "candidato" if exp["candidato"] else "referencia"),
            "modelo": exp["modelo"],
        })
        remover = exp.get("remover_colunas", [])
        parametros = {"modelo": exp["modelo"], **exp["hiperparametros"]}
        if eh_mlp:
            parametros.update({"epocas": exp["epocas"], "solver": "adam", "activation": "relu"})
        if remover:
            parametros["colunas_removidas"] = ",".join(remover)
        if exp["alteracoes_cli"]:
            parametros["alteracoes_cli"] = json.dumps(exp["alteracoes_cli"])
        mlflow.log_params(parametros)

        # Trace: cada etapa vira um "span", com duração, entradas e saídas
        with mlflow.start_span(name=f"pipeline:{nome}") as raiz:
            raiz.set_inputs({"experimento": nome, "parametros": {k: str(v) for k, v in parametros.items()}})

            with mlflow.start_span(name="preparacao_dados") as span:
                df, resumo, partes = preparar_divisao(args)
                X_treino, X_validacao, _, y_treino, y_validacao, _ = partes
                registrar_dados(df, resumo, partes, args)
                span.set_outputs({"treino": len(X_treino), "validacao": len(X_validacao),
                                  "teste_reservado": len(partes[2])})

            with mlflow.start_span(name="preprocessamento") as span:
                # Colunas retiradas de propósito (usado na investigação sem "odor")
                X_treino = X_treino.drop(columns=remover)
                X_validacao = X_validacao.drop(columns=remover)
                # Aprende as categorias SOMENTE com o treino -> sem vazamento de dados
                preprocessador = criar_preprocessador(X_treino.columns)
                X_treino_cod = preprocessador.fit_transform(X_treino)
                X_validacao_cod = preprocessador.transform(X_validacao)
                resumo_preproc = {
                    **config.PREPROCESSAMENTO,
                    "colunas_removidas_no_experimento": remover,
                    "atributos_de_entrada": X_treino.shape[1],
                    "colunas_apos_one_hot": X_treino_cod.shape[1],
                }
                span.set_outputs({"colunas_apos_one_hot": X_treino_cod.shape[1]})

            with mlflow.start_span(name="treinamento") as span:
                modelo = criar_modelo(exp["modelo"], exp["hiperparametros"], args.semente)
                historico = None
                if eh_mlp:
                    historico = treinar_mlp_por_epocas(modelo, X_treino_cod, y_treino,
                                                       X_validacao_cod, y_validacao, exp["epocas"])
                    span.set_outputs({"epocas": exp["epocas"],
                                      "train_loss_final": historico[-1]["train_loss"]})
                else:
                    treinar_modelo_simples(modelo, X_treino_cod, y_treino)
                    span.set_outputs({"treinado": True})

            with mlflow.start_span(name="validacao") as span:
                y_previsto = modelo.predict(X_validacao_cod)
                metricas = calcular_metricas(y_validacao, y_previsto, "val")
                if eh_mlp:
                    # val_accuracy já foi registrada época a época; a última época é a final
                    del metricas["val_accuracy"]
                    # Em que época a acurácia de validação chegou ao seu máximo (velocidade)
                    melhor = max(h["val_accuracy"] for h in historico)
                    metricas["epoca_val_accuracy_maxima"] = next(
                        h["epoca"] for h in historico if h["val_accuracy"] == melhor)
                mlflow.log_metrics(metricas)
                span.set_outputs({k: round(v, 4) for k, v in metricas.items()})

            raiz.set_outputs({"val_f1": metricas["val_f1"]})

        # Artefatos
        figura = gerar_matriz_confusao(y_validacao, y_previsto, f"{nome} (validação)")
        mlflow.log_figure(figura, "matriz_confusao_validacao.png")
        plt.close(figura)
        mlflow.log_text(gerar_relatorio(y_validacao, y_previsto), "relatorio_validacao.txt")
        if eh_mlp:
            figura = gerar_curvas(historico, nome)
            mlflow.log_figure(figura, "curvas_treinamento.png")
            plt.close(figura)
            mlflow.log_dict(historico, "historico_epocas.json")
        mlflow.log_dict(resumo_reproducao(exp, args, id_execucao), "configuracao.json")
        mlflow.log_dict(resumo_preproc, "preprocessamento.json")

        pipeline = montar_pipeline(preprocessador, modelo)
        mlflow.sklearn.log_model(
            pipeline,
            name="modelo",
            input_example=X_treino.head(3),
            pip_requirements=[f"scikit-learn=={sklearn.__version__}", "skops"],
            skops_trusted_types=TIPOS_CONFIAVEIS_SKOPS,
        )

    resultado = {
        "nome": nome,
        "run_id": run.info.run_id,
        "candidato": exp["candidato"],
        "modelo": exp["modelo"],
        "pipeline": pipeline,
        "exp": exp,
        "val_loss": historico[-1]["val_loss"] if eh_mlp else None,
        "val_accuracy": historico[-1]["val_accuracy"] if eh_mlp else metricas["val_accuracy"],
        "epoca_val_accuracy_maxima": metricas.get("epoca_val_accuracy_maxima"),
        **{k: v for k, v in metricas.items() if k.startswith("val_") and k != "val_accuracy"},
    }
    texto_loss = f" | val_loss={resultado['val_loss']:.6f}" if eh_mlp else ""
    print(f"  {nome:<27} val_accuracy={resultado['val_accuracy']:.4f} | "
          f"val_f1={resultado['val_f1']:.4f}{texto_loss}")
    return resultado


def tabela_comparacao(resultados):
    """Tabela com as métricas de VALIDAÇÃO de todas as runs."""
    colunas = ["nome", "modelo", "candidato", "val_accuracy", "val_precision",
               "val_recall", "val_f1", "val_loss", "epoca_val_accuracy_maxima", "run_id"]
    return pd.DataFrame([{c: r[c] for c in colunas} for r in resultados])


def selecionar_modelo(resultados):
    """
    Escolhe a configuração usando SOMENTE a validação:
    maior val_f1; empate -> menor val_loss; empate -> ordem do config.py.
    """
    candidatos = [r for r in resultados if r["candidato"]]
    if not candidatos:
        return None
    # sorted é estável: em empate total, mantém a ordem do config.py
    return sorted(candidatos, key=lambda r: (-r["val_f1"], r["val_loss"]))[0]


def justificar(escolhido, resultados):
    """Texto com a justificativa da escolha, baseado nos números da validação."""
    def descrever(r):
        return (f"{r['nome']} (val_f1={r['val_f1']:.4f}, val_loss={r['val_loss']:.6f}, "
                f"val_accuracy máxima na época {r['epoca_val_accuracy_maxima']})")

    linhas = [f"Critério: {config.CRITERIO_SELECAO} (somente validação).",
              f"Escolhida: {descrever(escolhido)}."]
    for r in resultados:
        if r["candidato"] and r is not escolhido:
            linhas.append(f"Comparada: {descrever(r)}.")
    return "\n".join(linhas)


def avaliacao_final(escolhido, resultados, args, id_execucao):
    """Run separada: avalia UMA vez no TESTE a configuração escolhida pela validação."""
    justificativa = justificar(escolhido, resultados)
    tabela = tabela_comparacao(resultados)

    with mlflow.start_run(run_name="avaliacao-final"):
        mlflow.set_tags({
            "mlflow.note.content": justificativa,
            "id_execucao": id_execucao,
            "papel": "avaliacao_final",
            "run_escolhida_id": escolhido["run_id"],
            "run_escolhida_nome": escolhido["nome"],
            "criterio_selecao": config.CRITERIO_SELECAO,
        })
        mlflow.log_params({
            "run_escolhida_nome": escolhido["nome"],
            "run_escolhida_id": escolhido["run_id"],
            "modelo": escolhido["modelo"],
            "epocas": escolhido["exp"]["epocas"],
            **escolhido["exp"]["hiperparametros"],
        })

        with mlflow.start_span(name="avaliacao_final") as raiz:
            raiz.set_inputs({"run_escolhida": escolhido["nome"], "criterio": config.CRITERIO_SELECAO})
            with mlflow.start_span(name="preparacao_dados") as span:
                # Mesma semente e mesma função -> exatamente o mesmo conjunto de teste
                df, resumo, partes = preparar_divisao(args)
                X_teste, y_teste = partes[2], partes[5]
                registrar_dados(df, resumo, partes, args)
                span.set_outputs({"teste": len(X_teste)})
            with mlflow.start_span(name="teste") as span:
                y_previsto = escolhido["pipeline"].predict(X_teste)
                metricas = calcular_metricas(y_teste, y_previsto, "test")
                mlflow.log_metrics(metricas)
                span.set_outputs({k: round(v, 4) for k, v in metricas.items()})
            raiz.set_outputs({"test_f1": metricas["test_f1"]})

        figura = gerar_matriz_confusao(y_teste, y_previsto, f"{escolhido['nome']} (teste)")
        mlflow.log_figure(figura, "matriz_confusao_teste.png")
        plt.close(figura)
        relatorio = gerar_relatorio(y_teste, y_previsto)
        mlflow.log_text(relatorio, "relatorio_teste.txt")
        mlflow.log_text(tabela.to_csv(index=False), "comparacao_validacao.csv")
        mlflow.log_dict({"run_escolhida": escolhido["nome"], "run_escolhida_id": escolhido["run_id"],
                         "justificativa": justificativa}, "selecao.json")

    # Marca a run escolhida para ser fácil de achar na interface
    mlflow.MlflowClient().set_tag(escolhido["run_id"], "selecionada", "sim")
    return metricas, relatorio, justificativa


# ---------------------------------------------------------------- principal
def main():
    args = ler_argumentos()
    experimentos = montar_experimentos(args)
    id_execucao = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Confere os dados antes de criar qualquer run
    df, resumo, partes = preparar_divisao(args)
    print("=" * 64)
    print("DADOS CARREGADOS E VALIDADOS")
    print("=" * 64)
    for chave, valor in resumo.items():
        print(f"  {chave}: {valor}")
    print(f"  treino: {len(partes[0])} | validação: {len(partes[1])} | teste: {len(partes[2])} "
          f"(semente = {args.semente})")

    mlflow.set_tracking_uri(config.URI_RASTREAMENTO)
    mlflow.set_experiment(config.NOME_EXPERIMENTO_MLFLOW)

    print("\n" + "=" * 64)
    print(f"TREINO + VALIDAÇÃO  (id da execução: {id_execucao})")
    print("=" * 64)
    resultados = [executar_experimento(exp, args, id_execucao) for exp in experimentos]

    escolhido = selecionar_modelo(resultados)
    print("\n" + "=" * 64)
    print("SELEÇÃO (somente validação)")
    print("=" * 64)
    if escolhido is None:
        print("  Nenhuma MLP candidata foi executada: seleção e teste final não realizados.")
        return
    print(justificar(escolhido, resultados))

    if args.sem_teste_final:
        print("\n--sem-teste-final: o conjunto de teste não foi usado.")
        return

    metricas, relatorio, _ = avaliacao_final(escolhido, resultados, args, id_execucao)
    print("\n" + "=" * 64)
    print(f"AVALIAÇÃO FINAL NO TESTE: {escolhido['nome']}")
    print("=" * 64)
    print("  " + " | ".join(f"{k}={v:.4f}" for k, v in metricas.items()))
    print(relatorio)
    print(f"Resultados no experimento MLflow '{config.NOME_EXPERIMENTO_MLFLOW}'.")
    print("Para ver na interface:  mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1")


if __name__ == "__main__":
    try:
        main()
    except Exception as erro:
        print(f"\nERRO: a execução não foi concluída.\n  {type(erro).__name__}: {erro}", file=sys.stderr)
        print("  Runs interrompidas ficam com status FAILED no MLflow.", file=sys.stderr)
        sys.exit(1)
