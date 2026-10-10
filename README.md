# Pipeline de treino e avaliação com MLflow: base Mushroom (UCI)

Classificação de cogumelos em **comestíveis (e)** ou **venenosos (p)** a partir de 22 características (cheiro, cor, formato...). O projeto implementa um pipeline completo de **treino → validação → seleção → teste final**, com todas as execuções rastreadas no **MLflow**.

## Dataset

- **Origem:** [UCI Machine Learning Repository: Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)
- **Arquivo:** `data/agaricus-lepiota.data` (sem cabeçalho; nomes das colunas definidos em `src/exploracao.py`)
- **Tamanho:** 8124 registros, 22 atributos categóricos + 1 coluna alvo (`class`)
- **Classes:** 4208 comestíveis (51,8%) e 3916 venenosos (48,2%)
- **Valores ausentes:** 2480 `?` na coluna `stalk-root`

## Estrutura

```
├── data/agaricus-lepiota.data      dados originais (nunca alterados)
├── src/
│   ├── config.py                   configurações: divisão, semente, experimentos
│   ├── exploracao.py               leitura, validação e exploração dos dados
│   ├── preparacao.py               X/y, divisão treino/validação/teste, pré-processamento
│   ├── modelos.py                  criação dos modelos (MLP, Árvore, Random Forest)
│   ├── treinamento.py              treino da MLP época por época
│   └── avaliacao.py                métricas, matriz de confusão, curvas
├── main.py                         pipeline completo com MLflow
└── notebooks/apresentacao_mushroom.ipynb   apresentação (lê os resultados do MLflow)
```

## Ambiente e instalação

Python 3.12, Windows/PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Como executar

```powershell
python main.py                                    # todas as runs + seleção + teste final
python main.py --help                             # todas as opções
python -m src.exploracao                          # só a exploração dos dados
python -m src.preparacao                          # só a preparação/divisão
```

Exemplos de execuções alterando configurações **sem editar o código**:

```powershell
python main.py --experimentos mlp-base mlp-lr-alta    # só algumas configurações
python main.py --epocas 50                            # épocas de todas as MLPs
python main.py --experimentos mlp-base --arquitetura 64,32 --sem-teste-final
python main.py --semente 7                            # outra divisão aleatória
```

As configurações padrão ficam em `src/config.py`. As alterações feitas pela linha de comando ficam registradas no parâmetro `alteracoes_cli` da run.

## Pipeline

1. **Dados:** leitura, validação (23 colunas, classes `e`/`p`), remoção de `veil-type` (valor único).
2. **Divisão estratificada** com semente 42: **60% treino (4874), 20% validação (1625), 20% teste (1625)**.
3. **Pré-processamento** ajustado **somente com o treino**: `?` vira a categoria `ausente`, e o One-Hot gera 116 colunas.
4. **Treino:**
   - **MLP** (`MLPClassifier`, Adam, ReLU) treinada **época por época** com `partial_fit`. A cada época, registra `train_loss`, `val_loss`, `train_accuracy` e `val_accuracy` (step = época).
   - **Árvore de Decisão e Random Forest** como **referência**. Treinam de uma vez (sem épocas), então não têm curvas.
5. **Seleção, somente pela validação**, entre as MLPs: maior `val_f1`; desempate pelo menor `val_loss`.
6. **Teste final:** só a configuração escolhida é avaliada no teste, numa run separada `avaliacao-final`.

Métricas binárias usam **venenoso (p) = classe positiva**, porque o erro mais perigoso é classificar um venenoso como comestível.

## Configurações dos experimentos

| Run | Modelo | Arquitetura | Taxa de aprendizado | L2 (alpha) | Épocas | O que testa |
|---|---|---|---|---|---|---|
| `mlp-base` | MLP | (32,) | 0.001 | 0.0001 | 30 | configuração de referência |
| `mlp-lr-alta` | MLP | (32,) | **0.01** | 0.0001 | 30 | efeito da taxa de aprendizado |
| `mlp-l2-alta` | MLP | (32,) | 0.001 | **0.1** | 30 | efeito da regularização L2 |
| `investigacao-mlp-sem-odor` | MLP | (32,) | 0.001 | 0.0001 | 30 | igual à base, **sem a coluna `odor`** (investigação) |
| `ref-arvore-decisao` | Árvore | max_depth=5 | – | – | – | referência |
| `ref-random-forest` | Random Forest | 100 árvores | – | – | – | referência |

Todas as MLPs usam `batch_size=64` e semente 42. Cada uma muda **um único parâmetro** em relação à base. A run de investigação não participa da seleção.

## MLflow

- **Armazenamento:** `mlflow.db` (SQLite) para o histórico e `mlruns/` para os artefatos
- **Experimento:** `mushroom-aula08` (os experimentos antigos `mushroom-classificacao` e `teste-mushroom` foram preservados)

O que cada run registra:

| | Runs de treino | Run `avaliacao-final` |
|---|---|---|
| Parâmetros | modelo, hiperparâmetros, épocas, semente, proporções, tamanhos, origem dos dados, pré-processamento | configuração escolhida e id da run escolhida |
| Métricas | `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy` por época (MLP); `val_precision`, `val_recall`, `val_f1` | `test_accuracy`, `test_precision`, `test_recall`, `test_f1` |
| Artefatos | curvas, matriz de confusão e relatório (validação), histórico por época, `preprocessamento.json`, `configuracao.json` (reprodução), modelo | matriz de confusão e relatório (teste), `comparacao_validacao.csv`, `selecao.json` |
| Traces | `preparacao_dados → preprocessamento → treinamento → validacao` | `preparacao_dados → teste` |
| Tags | `papel` (candidato/investigacao/referencia), `id_execucao`, `selecionada=sim` na escolhida, commit do git | `run_escolhida_nome`, `run_escolhida_id` |

Para abrir a interface:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```

Depois acesse http://127.0.0.1:5000. Os traces ficam na aba **Traces** do experimento.

Se uma execução falhar, o terminal mostra uma mensagem de erro e a run fica com status **FAILED** no MLflow.

## Resultados (execução `2026-10-10_20-38-41`, commit `9a0aaa5`)

**Validação:**

| Run | val_accuracy | val_f1 | val_loss (final) | Época em que a validação chegou ao máximo |
|---|---|---|---|---|
| mlp-base | 1.0000 | 1.0000 | 0.000351 | 4 |
| **mlp-lr-alta** | 1.0000 | 1.0000 | **0.000016** | **2** |
| mlp-l2-alta | 1.0000 | 1.0000 | 0.005519 | 4 |
| investigacao-mlp-sem-odor | 1.0000 | 1.0000 | 0.001062 | 9 |
| ref-arvore-decisao | 1.0000 | 1.0000 | – | – |
| ref-random-forest | 1.0000 | 1.0000 | – | – |

**Escolha:** todas as MLPs empataram em `val_f1 = 1.0`. Pelo critério de desempate, a escolhida foi a **`mlp-lr-alta`**, que teve o menor `val_loss` e atingiu 100% de acurácia de validação já na 2ª época.

**Teste final (`mlp-lr-alta`):** accuracy = precision = recall = F1 = **1.0000** (842 comestíveis e 783 venenosos, zero erros).

**Investigação do resultado inesperado:**
- **Sintoma:** todas as runs chegam a 100%.
- **Hipótese:** o atributo `odor` sozinho separa as classes.
- **Nova run:** `investigacao-mlp-sem-odor`.
- **Resultado:** a hipótese foi **refutada**. Sem o `odor`, a MLP ainda chega a 100% na validação, mas mais devagar (época 9, contra época 4) e com `val_loss` cerca de 3x maior. Outros atributos, como `spore-print-color`, carregam informação parecida: a base tem atributos redundantes.

**Limitações:** a base Mushroom é quase perfeitamente separável, então todos os modelos chegam a 100%. A comparação depende do `val_loss` e da velocidade de convergência, não da acurácia. Uma `val_loss` muito baixa também indica um modelo muito confiante. Com dados de outra origem, o desempenho pode ser menor.
