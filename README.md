
---

# Previsão de Preços de Imóveis – CDMX (Cidade do México)

Este projeto utiliza Machine Learning supervisionado para prever o preço de imóveis na Cidade do México.
O pipeline completo — desde o carregamento dos dados até a previsão final — está implementado em um único arquivo: **`src/Main.py`**.

---

## Estrutura do Projeto

```
project/
│
├── data/
│   ├── processed/
│   │   └── housing_data_CDMX_cleaned.csv
│   ├── raw/
│   │   └── housing_data_CDMX.csv
│
├── notebooks/
│   └── data_exploration.ipynb
│
├── src/
│   └── Main.py
│
├── image.png
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Objetivo do Projeto

Criar um pipeline completo de aprendizado de máquina capaz de:

1. Carregar e inspecionar os dados.
2. Limpar e tratar inconsistências.
3. Criar novas features relevantes.
4. Realizar análise exploratória de dados.
5. Treinar modelos de regressão (Linear Regression e Random Forest).
6. Avaliar métricas como MSE, RMSE, MAE e R².
7. Gerar previsões para novos imóveis.

---

## Pipeline do Projeto

Todo o fluxo é executado dentro do arquivo **`Main.py`**, seguindo as etapas:

### 1. Carregamento e Preparo dos Dados

* Leitura do arquivo `housing_data_CDMX_cleaned.csv`.
* Exibição de amostras, estatísticas e informações gerais do dataset.
* Tratamento de valores ausentes.
* Identificação de variáveis numéricas e categóricas.

### 2. Engenharia de Features

O script cria automaticamente variáveis como:

* `area_difference` — diferença entre área total e área coberta
* `area_ratio` — proporção entre área coberta e total
* Codificação numérica de colunas categóricas
* Seleção automatizada das colunas de entrada (features)

### 3. Análise Exploratória de Dados (EDA)

São gerados:

* histogramas
* boxplots
* heatmap de correlação
* scatterplots
* análise de outliers
* gráficos de distribuição do preço

### 4. Treinamento dos Modelos

Modelos utilizados:

* Linear Regression (com StandardScaler)
* Random Forest Regressor

As métricas calculadas incluem:

* MSE
* RMSE
* MAE
* R²
* R² com cross-validation (5-fold)

### 5. Visualização dos Resultados

São produzidos gráficos como:

* previsões vs valores reais
* resíduos
* importância das variáveis (Random Forest)

---

## Predição de Novo Imóvel

Exemplo de dicionário utilizado no código:

```python
novo_imovel = {
    "surface_total_in_m2": 100,
    "surface_covered_in_m2": 90,
    "price": 0,
    "price_aprox_local_currency": 0,
    "price_aprox_usd": 0,
    "price_per_m2": 0,
    "latitude": 19.4,
    "longitude": -99.1,
    "difference": 10,
    "area_ratio": 0.9,
    "property_type_encoded": 1,
    "places_encoded": 2,
    "currency_encoded": 0
}
```

---

## Como Executar

1. Instale as dependências:

```
pip install -r requirements.txt
```

2. Certifique-se de que o arquivo processado esteja localizado em:

```
data/processed/housing_data_CDMX_cleaned.csv
```

3. Execute o script principal:

```
python src/Main.py
```

---

## Observações

* Toda a lógica está centralizada em **Main.py**, cobrindo ETL, EDA, engenharia de features, treinamento e avaliação.
* O notebook em `notebooks/` é opcional e serve apenas para exploração manual dos dados.

---
