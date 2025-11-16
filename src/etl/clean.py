def limpar_dados(df):
    df = df.drop_duplicates()
    df = df.dropna()
    df = df[df["price"] > 0]
    df = df[df["surface_total_in_m2"] > 0]
    df = df[df["surface_covered_in_m2"] > 0]
    return df
