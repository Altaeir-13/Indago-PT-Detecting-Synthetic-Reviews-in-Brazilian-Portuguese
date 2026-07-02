# Indago PT: Detecting Synthetic Reviews in Brazilian Portuguese

Experimento reproduzivel para a 3a avaliacao de Aprendizado Profundo:
classificacao binaria de reviews em portugues brasileiro usando uma CNN 1D em
PyTorch, comparada com baselines classicos de PLN.

Repositorio do projeto:

https://github.com/Altaeir-13/Indago-PT-Detecting-Synthetic-Reviews-in-Brazilian-Portuguese

## Objetivo

Classificar uma review individual como:

- `0`: falsa/sintetica.
- `1`: genuina.

O projeto trata a unidade de analise como a review isolada. Ele nao detecta
astroturfing completo, campanhas coordenadas, comportamento de usuarios,
padroes temporais ou grafos de contas, porque o dataset principal nao contem
usuario, data, rating, produto especifico nem relacoes entre contas.

## Codigo principal

O experimento foi consolidado em um unico notebook autocontido:

    notebooks/01_experimento_fake_reviews.ipynb

A pasta `src/` foi removida porque toda a logica de carregamento, EDA, preprocessamento, split, baselines, CNN, avaliacao e geracao de outputs esta no notebook.

## Dataset

Use o Fake Reviews PT-BR Dataset:

https://github.com/cristianomg10/fake-reviews-ptbr-dataset

O arquivo esperado e:

```text
data/raw/true_fake_dataset_top15.csv
```

Para preparar o dataset:

1. Acesse https://github.com/cristianomg10/fake-reviews-ptbr-dataset.
2. Baixe o arquivo `true_fake_dataset_top15.csv`.
3. Coloque o arquivo em `data/raw/true_fake_dataset_top15.csv`.
4. Abra `notebooks/01_experimento_fake_reviews.ipynb` e execute as celulas em ordem.

A base e derivada do dataset publico da Olist:

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Colunas reais detectadas na execucao:

- texto: `review_comment_message`
- categoria: `product_category_name`
- rotulo: `label`

O mapeamento tambem foi salvo em `outputs/tables/column_mapping.json`.

## Instalacao

```bash
python -m pip install -r requirements.txt
```

## Como executar

O codigo principal do experimento esta em:

    notebooks/01_experimento_fake_reviews.ipynb

Para reproduzir o experimento completo:

1. Instale as dependencias com `python -m pip install -r requirements.txt`.
2. Coloque o dataset em `data/raw/true_fake_dataset_top15.csv`.
3. Abra o notebook no Jupyter Notebook, JupyterLab ou VS Code.
4. Mantenha `USE_SMOKE_TEST = False`.
5. Execute todas as celulas em ordem (`Run All`).

Smoke test rapido:

1. Abra o notebook.
2. Altere `USE_SMOKE_TEST = True` na secao de configuracao global.
3. Execute todas as celulas.
4. Os resultados de smoke sao gravados em `outputs/smoke/`, que e ignorado pelo Git.

O notebook e autocontido e nao depende da pasta `src/`.

## Estrutura

```text
.
|-- README.md
|-- requirements.txt
|-- data/
|   |-- README.md
|   `-- raw/
|-- notebooks/
|   `-- 01_experimento_fake_reviews.ipynb
|-- outputs/
|   |-- figures/
|   |-- tables/
|   `-- models/
|-- report/
|   |-- relatorio.md
|   `-- slides.md
`-- tests/
    `-- fixtures/
        `-- smoke_reviews.csv
```

## Pipeline

1. Carrega o CSV local.
2. Infere colunas de texto, categoria e rotulo.
3. Faz limpeza minima: textos vazios sao removidos, espacos e quebras de linha
   sao normalizados.
4. Executa EDA inicial.
5. Divide em treino, validacao e teste com estratificacao `70/15/15` e
   `random_state=42`.
6. Treina TF-IDF + Regressao Logistica.
7. Treina TF-IDF + SVM Linear.
8. Treina CNN 1D em PyTorch com early stopping.
9. Avalia todos os modelos no teste.
10. Salva metricas, graficos, modelos, historico de treino, resumo do split e analise de erros em `outputs/`.

## CNN 1D em PyTorch

Arquitetura:

```text
Embedding(vocab_size, embedding_dim, padding_idx=0)
-> Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=kernel_size)
-> ReLU
-> GlobalMaxPool1d
-> Dropout
-> Linear(filters, 1)
```

Treinamento padrao:

- Loss: `BCEWithLogitsLoss`.
- Otimizador: Adam.
- Learning rate: `0.001`.
- Batch size: `64`.
- Epocas: ate `30`.
- Early stopping por `val_loss`, `patience=4`.
- Melhor modelo salvo em `outputs/models/cnn1d.pt`.

## Resultados reais

Os resultados abaixo foram gerados pelo pipeline real e estao em
`outputs/tables/model_comparison.csv`.

| Modelo | Accuracy | Precision fake label 0 | Recall fake label 0 | F1 fake label 0 | F1 macro | AUC-ROC fake label 0 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + Regressao Logistica | 0.885086 | 0.866385 | 0.910667 | 0.887974 | 0.885009 | 0.949770 |
| TF-IDF + SVM Linear | 0.897533 | 0.882428 | 0.917333 | 0.899542 | 0.897492 | 0.962379 |
| CNN 1D PyTorch | 0.949544 | 0.950557 | 0.948444 | 0.949499 | 0.949544 | 0.987807 |

Leituras principais:

- A CNN 1D PyTorch teve o melhor desempenho geral.
- Entre os baselines classicos, o SVM linear superou a regressao logistica.
- A CNN teve ganho de aproximadamente 5,2 pontos percentuais de acuracia em
  relacao ao SVM linear.
- A CNN atingiu F1 macro de aproximadamente 94,95%.
- A AUC-ROC da CNN para a classe falsa/sintetica foi aproximadamente 98,78%.
- A alta AUC sugere boa separacao entre reviews sinteticas e genuinas no
  dataset avaliado.

Esses resultados nao devem ser interpretados como deteccao completa de
astroturfing. A classe falsa e sintetica/operacional, gerada por GPT-2, e nao
fraude humana comprovada.

## EDA real

Arquivos de EDA gerados:

- `outputs/tables/eda_summary.json`
- `outputs/tables/class_distribution.csv`
- `outputs/tables/category_distribution.csv`
- `outputs/tables/review_length_summary.csv`
- `outputs/tables/missing_values.csv`
- `outputs/tables/examples_by_class.csv`

Resumo observado:

- 29.988 amostras apos limpeza.
- Classes: 15.000 falsas/sinteticas e 14.988 genuinas.
- Sem valores ausentes nas colunas canonicas depois do carregamento.
- 683 linhas duplicadas e 1.537 textos duplicados foram identificados na EDA.
- Tamanho medio das reviews: 82,95 caracteres e 13,98 tokens.
- Mediana: 92 caracteres e 15 tokens.

Figuras de EDA:

- `outputs/figures/class_distribution.png`
- `outputs/figures/category_distribution.png`
- `outputs/figures/review_char_length_hist.png`
- `outputs/figures/review_token_length_hist.png`
- `outputs/figures/review_token_length_by_class.png`

## Figuras e analise de erros

Figuras de resultado:

- `outputs/figures/cnn_loss_curve.png`
- `outputs/figures/cnn_accuracy_curve.png`
- `outputs/figures/confusion_matrix_cnn1d.png`
- `outputs/figures/confusion_matrix_linear_svm.png`
- `outputs/figures/confusion_matrix_logistic_regression.png`

A analise qualitativa da CNN esta em:

```text
outputs/tables/cnn_error_analysis.csv
```

Esse arquivo lista falsos positivos, isto e, reviews genuinas preditas como
falsas/sinteticas, e falsos negativos, isto e, reviews falsas/sinteticas
preditas como genuinas. A discussao desses casos deve considerar que reviews
curtas, genericas, negativas, pouco especificas ou muito padronizadas podem
ficar proximas da fronteira de decisao.

## Limitacoes

- A classe falsa/sintetica e gerada por GPT-2 no dataset. Ela nao representa
  necessariamente fraude humana comprovada.
- O projeto classifica reviews individuais.
- Nao ha metadados para detectar astroturfing completo, campanhas coordenadas,
  grafos de usuarios ou padroes temporais.
- Resultados dependem do recorte do dataset e devem ser interpretados como
  classificacao supervisionada no dominio operacional da base.

## Relacao com o TCC

Este projeto pode ser reaproveitado no TCC como:

- um modelo neural intermediario entre baselines TF-IDF e uma comparacao futura
  com BERTimbau;
- um pipeline experimental reutilizavel para carregamento, EDA, treino,
  avaliacao e analise de erros;
- uma base para discutir ameacas a validade, principalmente a diferenca entre
  texto sintetico de dataset e fraude humana real.

BERTimbau permanece como extensao futura ou comparacao posterior, nao como
modelo principal desta avaliacao.
## Exportacao para PDF

A fonte oficial do relatorio e o arquivo LaTeX:

    report/relatorio.tex

Compile o relatorio a partir da pasta report/ com LaTeX/BibTeX, por exemplo:

    cd report
    pdflatex relatorio.tex
    bibtex relatorio
    pdflatex relatorio.tex
    pdflatex relatorio.tex

O PDF final esperado e report/relatorio.pdf. Verifique visualmente o PDF antes
de entregar para garantir que tabelas, quebras de pagina, figuras e referencias
estejam legiveis.
