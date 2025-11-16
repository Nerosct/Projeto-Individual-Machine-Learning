from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def treinar_modelos(X_train, y_train):
    modelos = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200)
    }

    pipelines = {}

    for nome, modelo in modelos.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("modelo", modelo)
        ])

        pipe.fit(X_train, y_train)
        pipelines[nome] = pipe

    return pipelines
