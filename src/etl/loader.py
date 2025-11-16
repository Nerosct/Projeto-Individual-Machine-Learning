import pandas as pd


def carregar_dataset(caminho):
    df = pd.read_csv(caminho)
    print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


if __name__ == "__main__":
    caminho = 'data/processed/housing_data_CDMX_cleaned.csv'
    print("Loader module carregado com sucesso.")
    carregar_dataset(caminho)
