from sklearn.model_selection import train_test_split

def preparar_treino(df, coluna_alvo="price"):
    X = df.drop(columns=[coluna_alvo])
    y = df[coluna_alvo]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
