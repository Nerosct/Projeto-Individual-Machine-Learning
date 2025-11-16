import pandas as pd

def prever(modelo, dados_dict):
    df = pd.DataFrame([dados_dict])
    return modelo.predict(df)[0]
