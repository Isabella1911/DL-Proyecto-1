import pandas as pd, numpy as np, torch, torch.nn as nn, joblib, json

cfg = json.load(open("model_config.json"))
preprocessor = joblib.load("preprocessor.pkl")
ridge = joblib.load("ridge_final.pkl")

class MLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def preprocess_new(test):
    test = test.copy()
    none_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
                 "FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond",
                 "PoolQC","Fence","MiscFeature","MasVnrType"]
    for c in none_cols:
        if c in test.columns: test[c] = test[c].fillna("None")
    test["GarageYrBlt"] = test["GarageYrBlt"].fillna(test["YearBuilt"])
    test["LotFrontage"] = test["LotFrontage"].fillna(test["LotFrontage"].median())
    for c in ["MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
              "BsmtFullBath","BsmtHalfBath","GarageCars","GarageArea"]:
        if c in test.columns: test[c] = test[c].fillna(0)
    test["TotalSF"]      = test["TotalBsmtSF"] + test["1stFlrSF"] + test["2ndFlrSF"]
    test["TotalBath"]    = test["FullBath"] + 0.5*test["HalfBath"] + test["BsmtFullBath"] + 0.5*test["BsmtHalfBath"]
    test["HouseAge"]     = test["YrSold"] - test["YearBuilt"]
    test["RemodAge"]     = test["YrSold"] - test["YearRemodAdd"]
    test["GarageAge"]    = test["YrSold"] - test["GarageYrBlt"]
    test["IsNew"]        = (test["YrSold"] == test["YearBuilt"]).astype(int)
    test["TotalPorchSF"] = test["OpenPorchSF"] + test["EnclosedPorch"] + test["3SsnPorch"] + test["ScreenPorch"]
    return test

test = pd.read_csv("test_features-1.csv")
ids = test["Id"]
test = preprocess_new(test)
X_test = test.drop(columns=["Id"], errors="ignore")
X_test_t = preprocessor.transform(X_test)
if hasattr(X_test_t, "toarray"): X_test_t = X_test_t.toarray()
X_test_ten = torch.tensor(X_test_t, dtype=torch.float32)

mlp_preds = []
for s in range(cfg["n_seeds"]):
    model = MLP(cfg["n_features"], cfg["hidden"])
    model.load_state_dict(torch.load(f"mlp_seed{s}.pt"))
    model.eval()
    with torch.no_grad():
        mlp_preds.append(model(X_test_ten).numpy())
mlp_pred = np.mean(mlp_preds, axis=0)
ridge_pred = ridge.predict(X_test_t)

w = cfg["blend_weight_mlp"]
final_pred_log = w*mlp_pred + (1-w)*ridge_pred
final_pred = np.expm1(final_pred_log)

pd.DataFrame({"Id": ids, "Prediction": final_pred}).to_csv("predictions.csv", index=False)

if "Prediction" in test.columns:
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(test["Prediction"], final_pred))
    print(f"RMSE: {rmse:,.0f}")