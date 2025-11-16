from sklearn.metrics import mean_squared_error, r2_score

def avaliar(modelos, X_test, y_test):
    resultados = {}

    for nome, modelo in modelos.items():
        pred = modelo.predict(X_test)

        resultados[nome] = {
            "MSE": mean_squared_error(y_test, pred),
            "R2": r2_score(y_test, pred)
        }

    return resultados
