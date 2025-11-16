from etl.loader import carregar_dataset
from etl.clean import limpar_dados
from etl.features import criar_features

from analysis.stats import mostrar_estatisticas
from analysis.plots import plot_distribuicoes

from ml.preprocess import preparar_treino
from ml.train import treinar_modelos
from ml.evaluate import avaliar
from ml.predict import prever


def main():

    # 1) Carregar
    df = carregar_dataset("data/housing_data_CDMX.csv")

    # 2) Limpar
    df = limpar_dados(df)

    # 3) Feature Engineering
    df = criar_features(df)

    # 4) Estatísticas e Gráficos
    mostrar_estatisticas(df)
    plot_distribuicoes(df)

    # 5) Treino
    X_train, X_test, y_train, y_test = preparar_treino(df)

    modelos = treinar_modelos(X_train, y_train)

    # 6) Avaliação
    resultados = avaliar(modelos, X_test, y_test)

    print("\nResultados dos modelos:")
    for m, r in resultados.items():
        print(m, r)

    # 7) Previsão de novo imóvel
    novo = {
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

    preco_previsto = prever(modelos["RandomForest"], novo)
    print("\nPreço previsto:", preco_previsto)


if __name__ == "__main__":
    main()
