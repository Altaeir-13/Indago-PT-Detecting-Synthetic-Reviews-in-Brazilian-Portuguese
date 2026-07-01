# Dados

Coloque aqui o arquivo CSV do Fake Reviews PT-BR Dataset.

Arquivo esperado:

```text
data/raw/true_fake_dataset_top15.csv
```

Fonte do dataset:

- Fake Reviews PT-BR Dataset: https://github.com/cristianomg10/fake-reviews-ptbr-dataset
- Base original Olist: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

O codigo nao baixa o dataset automaticamente. Se o CSV nao existir, `src.run_experiment`
encerra com uma mensagem clara de instrucao de download.

Interpretacao dos rotulos usada no experimento:

- `0`: review falsa/sintetica, gerada por GPT-2 no contexto operacional do dataset.
- `1`: review genuina.

Essa classe `0` nao deve ser interpretada como fraude humana comprovada.

