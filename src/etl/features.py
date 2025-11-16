def criar_features(df):
    df["price_per_m2"] = df["price"] / df["surface_total_in_m2"]
    df["difference"] = df["surface_total_in_m2"] - df["surface_covered_in_m2"]
    df["area_ratio"] = df["surface_covered_in_m2"] / df["surface_total_in_m2"]

    df["property_type_encoded"] = df["property_type"].astype("category").cat.codes
    df["places_encoded"] = df["place_with_parent_names"].astype("category").cat.codes
    df["currency_encoded"] = df["currency"].astype("category").cat.codes

    return df
