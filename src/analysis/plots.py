import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribuicoes(df):
    plt.figure(figsize=(8, 4))
    sns.histplot(df["price"], bins=50)
    plt.title("Distribuição de Preço")
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.histplot(df["price_per_m2"], bins=50)
    plt.title("Distribuição Preço por m²")
    plt.show()
