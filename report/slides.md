## Slide 1 - Titulo e objetivo

**Deteccao de reviews falsas/sinteticas em portugues brasileiro usando CNN 1D**

- Tarefa: classificacao binaria de reviews individuais.
- Classe `0`: falsa/sintetica.
- Classe `1`: genuina.
- Modelo principal: CNN 1D em PyTorch.
- Comparacao: TF-IDF + Regressao Logistica e TF-IDF + SVM Linear.

Fala sugerida: apresentar o problema como classificacao textual supervisionada e
antecipar que o trabalho nao detecta astroturfing completo.

## Slide 2 - Problema, motivacao e escopo

- Reviews influenciam confianca e decisao de compra em e-commerce.
- Opinion spam e fake reviews sao problemas relevantes em ambientes digitais.
- O experimento investiga sinais textuais em reviews de portugues brasileiro.
- Escopo: review individual.
- Fora do escopo: campanhas coordenadas, grafos de usuarios, padroes temporais,
  rating e produto especifico.

Fala sugerida: explicar que astroturfing envolve coordenacao, mas o dataset nao
permite observar campanhas ou relacoes entre contas.

## Slide 3 - Dataset e EDA

Fonte: Fake Reviews PT-BR Dataset, derivado da base Olist.

- Arquivo usado: `data/raw/true_fake_dataset_top15.csv`.
- Texto: `review_comment_message`.
- Categoria: `product_category_name`.
- Rotulo: `label`.
- 29.988 amostras apos limpeza.
- 15.000 falsas/sinteticas e 14.988 genuinas.
- Media: 82,95 caracteres e 13,98 tokens por review.

Figuras:

- `outputs/figures/class_distribution.png`
- `outputs/figures/category_distribution.png`
- `outputs/figures/review_token_length_hist.png`

## Slide 4 - Trabalhos relacionados

- Borges et al. (2025): benchmark e dataset de fake reviews em portugues
  brasileiro.
- Kim (2014): CNNs para classificacao de sentencas.
- Devlin et al. (2019): BERT e modelos Transformer pre-treinados.
- Souza, Nogueira e Lotufo (2020): BERTimbau para portugues brasileiro.
- Goodfellow, Bengio e Courville (2016): fundamentos de deep learning.

Fala sugerida: destacar que BERTimbau e citado como extensao futura, nao como
modelo principal desta avaliacao.

## Slide 5 - Pipeline experimental

1. Carregamento do CSV.
2. Inferencia das colunas reais.
3. Limpeza minima do texto.
4. EDA e salvamento de tabelas/figuras.
5. Divisao estratificada 70/15/15, `random_state=42`.
6. Treino dos baselines TF-IDF.
7. Treino da CNN 1D PyTorch.
8. Avaliacao no conjunto de teste.
9. Analise qualitativa de erros.

Fala sugerida: destacar que todas as metricas foram geradas automaticamente pelo
codigo e salvas em `outputs/`.

## Slide 6 - Arquitetura e configuracao da CNN 1D

```text
Embedding(vocab_size, embedding_dim, padding_idx=0)
-> Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=kernel_size)
-> ReLU
-> GlobalMaxPool1d
-> Dropout
-> Linear(filters, 1)
```

Treinamento:

- `BCEWithLogitsLoss`.
- Adam, learning rate 0.001.
- Batch size 64.
- Ate 30 epocas.
- Early stopping por `val_loss`, patience 4.
- Melhor modelo: `outputs/models/cnn1d.pt`.

## Slide 7 - Curvas de treinamento

Arquivos:

- `outputs/figures/cnn_loss_curve.png`
- `outputs/figures/cnn_accuracy_curve.png`
- `outputs/tables/cnn_training_history.csv`

Pontos para comentar:

- Melhor `val_loss` observado na epoca 7.
- Treino encerrado na epoca 11 pelo early stopping.
- As curvas ajudam a discutir overfitting e estabilidade do treinamento.

## Slide 8 - Resultados comparativos

Fonte: `outputs/tables/model_comparison.csv`.

| Modelo | Accuracy | F1 fake | F1 macro | AUC-ROC classe 0 |
|---|---:|---:|---:|---:|
| TF-IDF + Regressao Logistica | 0.885086 | 0.887974 | 0.885009 | 0.949770 |
| TF-IDF + SVM Linear | 0.897533 | 0.899542 | 0.897492 | 0.962379 |
| CNN 1D PyTorch | 0.949544 | 0.949499 | 0.949544 | 0.987807 |

Mensagens principais:

- CNN 1D PyTorch teve o melhor desempenho geral.
- SVM linear foi o melhor baseline classico.
- CNN superou o SVM em aproximadamente 5,2 pontos percentuais de acuracia.
- F1 macro da CNN: aproximadamente 94,95%.
- AUC-ROC da CNN para a classe falsa/sintetica: aproximadamente 98,78%.

## Slide 9 - Analise de erros e limitacoes

Figuras e tabelas:

- `outputs/figures/confusion_matrix_cnn1d.png`
- `outputs/figures/confusion_matrix_linear_svm.png`
- `outputs/figures/confusion_matrix_logistic_regression.png`
- `outputs/tables/cnn_error_analysis.csv`

Limitacoes:

- A classe falsa e sintetica/operacional, gerada por GPT-2.
- O resultado nao comprova fraude humana real.
- O trabalho nao detecta astroturfing completo.
- Nao ha usuario, data, rating, produto especifico ou rede de contas.

Fala sugerida: comentar falsos positivos e falsos negativos sem generalizar para
fraude real.

## Slide 10 - Conclusao e relacao com o TCC

Conclusoes:

- A CNN 1D PyTorch foi o melhor modelo no teste.
- O SVM linear superou a regressao logistica entre os baselines classicos.
- A alta AUC sugere boa separacao entre reviews sinteticas e genuinas no dataset.
- O resultado nao significa deteccao completa de astroturfing.

Aproveitamento no TCC:

- Modelo neural intermediario entre TF-IDF e BERTimbau.
- Pipeline experimental reutilizavel.
- Base para analise de erros e ameacas a validade.
- BERTimbau como extensao futura ou comparacao futura, nao como modelo principal
  desta avaliacao.