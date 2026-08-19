# requirements: pandas numpy scikit-learn torch joblib
import pandas as pd, numpy as np, torch, torch.nn as nn, joblib, json
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ---------- 1. CARGA + OUTLIERS CONOCIDOS DE AMES ----------
df = pd.read_csv("train.csv")
df = df[~((df["GrLivArea"] > 4000) & (df["Prediction"] < 300000))].reset_index(drop=True)

# ---------- 2. NAs QUE SIGNIFICAN "NO TIENE" ----------
none_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
             "FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond",
             "PoolQC","Fence","MiscFeature","MasVnrType"]
for c in none_cols: df[c] = df[c].fillna("None")
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda s: s.fillna(s.median()))
for c in ["MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
          "BsmtFullBath","BsmtHalfBath","GarageCars","GarageArea"]:
    df[c] = df[c].fillna(0)

# ---------- 3. FEATURE ENGINEERING ----------
df["TotalSF"]      = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["TotalBath"]    = df["FullBath"] + 0.5*df["HalfBath"] + df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"]
df["HouseAge"]     = df["YrSold"] - df["YearBuilt"]
df["RemodAge"]     = df["YrSold"] - df["YearRemodAdd"]
df["GarageAge"]    = df["YrSold"] - df["GarageYrBlt"]
df["IsNew"]        = (df["YrSold"] == df["YearBuilt"]).astype(int)
df["TotalPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]

y = np.log1p(df["Prediction"].values.astype(np.float32))
X = df.drop(columns=["Id","Prediction"])

qual_map = ["None","Po","Fa","TA","Gd","Ex"]
ordinal_cols = [c for c in ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC",
                "KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"] if c in X.columns]
for c in ordinal_cols: X[c] = X[c].fillna("None")
remaining_cat = [c for c in X.select_dtypes(exclude=np.number).columns if c not in ordinal_cols]
num_cols = X.select_dtypes(include=np.number).columns.tolist()

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), num_cols),
        ("ord", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="None")),
                           ("enc", OrdinalEncoder(categories=[qual_map]*len(ordinal_cols),
                                                   handle_unknown="use_encoded_value", unknown_value=-1))]),
         ordinal_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("ohe", OneHotEncoder(handle_unknown="ignore"))]), remaining_cat),
    ])

def rmse_real(y_true_log, y_pred_log):
    return np.sqrt(mean_squared_error(np.expm1(y_true_log), np.expm1(y_pred_log)))

# ---------- 4. MLP ----------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64,32)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def to_t(a): return torch.tensor(a, dtype=torch.float32)

def train_mlp(Xtr_t, ytr, Xva_t, yva, seed, hidden=(64,32), wd=1e-4, lr=1e-3,
              epochs=400, patience=50, batch_size=32):
    torch.manual_seed(seed)
    model = MLP(Xtr_t.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.MSELoss()
    Xtr_ten, ytr_ten = to_t(Xtr_t), to_t(ytr)
    Xva_ten, yva_ten = to_t(Xva_t), to_t(yva)
    n = Xtr_ten.shape[0]
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            loss = crit(model(Xtr_ten[idx]), ytr_ten[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = crit(model(Xva_ten), yva_ten).sqrt().item()
        if va < best_val:
            best_val, best_state, bad = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    with torch.no_grad():
        pred = model(Xva_ten).numpy()
    return pred

# ---------- 5. VALIDACIÓN CRUZADA (usa esto para la sección 2.3 y 2.4 del informe) ----------
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
ridge_scores, mlp_scores, ens_scores = [], [], []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    Xtr_df, Xva_df = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    pre = make_preprocessor()
    Xtr_t = pre.fit_transform(Xtr_df); Xva_t = pre.transform(Xva_df)
    if hasattr(Xtr_t, "toarray"): Xtr_t, Xva_t = Xtr_t.toarray(), Xva_t.toarray()

    ridge = Ridge(alpha=10.0).fit(Xtr_t, ytr)
    ridge_pred = ridge.predict(Xva_t)

    # bagging de 5 semillas: reduce varianza del MLP en datasets chicos
    mlp_preds = [train_mlp(Xtr_t, ytr, Xva_t, yva, seed=s) for s in range(5)]
    mlp_pred = np.mean(mlp_preds, axis=0)

    ens_pred = 0.5*mlp_pred + 0.5*ridge_pred

    r_rmse, m_rmse, e_rmse = rmse_real(yva, ridge_pred), rmse_real(yva, mlp_pred), rmse_real(yva, ens_pred)
    ridge_scores.append(r_rmse); mlp_scores.append(m_rmse); ens_scores.append(e_rmse)
    print(f"fold {fold}: Ridge=${r_rmse:,.0f}  MLP(bagged)=${m_rmse:,.0f}  Ensemble=${e_rmse:,.0f}")

print(f"\nMEAN Ridge:    ${np.mean(ridge_scores):,.0f}")
print(f"MEAN MLP:      ${np.mean(mlp_scores):,.0f}")
print(f"MEAN Ensemble: ${np.mean(ens_scores):,.0f}")

# ---------- 6. ENTRENAMIENTO FINAL SOBRE TODO EL DATASET (para competencia) ----------
pre_final = make_preprocessor()
X_all_t = pre_final.fit_transform(X)
if hasattr(X_all_t, "toarray"): X_all_t = X_all_t.toarray()

ridge_final = Ridge(alpha=10.0).fit(X_all_t, y)

# 5 MLPs finales (bagging) entrenados con un split interno solo para early stopping
from sklearn.model_selection import train_test_split
Xtr_f, Xva_f, ytr_f, yva_f = train_test_split(X_all_t, y, test_size=0.1, random_state=SEED)
final_mlps_state = []
for s in range(5):
    torch.manual_seed(s)
    model = MLP(X_all_t.shape[1], (64,32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    Xtr_ten, ytr_ten = to_t(Xtr_f), to_t(ytr_f)
    Xva_ten, yva_ten = to_t(Xva_f), to_t(yva_f)
    n = Xtr_ten.shape[0]
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(400):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i+32]
            opt.zero_grad()
            loss = crit(model(Xtr_ten[idx]), ytr_ten[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = crit(model(Xva_ten), yva_ten).sqrt().item()
        if va < best_val:
            best_val, best_state, bad = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 50: break
    final_mlps_state.append(best_state)
    torch.save(best_state, f"mlp_seed{s}.pt")

joblib.dump(preprocessor, "preprocessor.pkl") if False else joblib.dump(pre_final, "preprocessor.pkl")
joblib.dump(ridge_final, "ridge_final.pkl")
json.dump({"n_features": X_all_t.shape[1], "hidden": [64,32], "n_seeds": 5,
           "blend_weight_mlp": 0.5}, open("model_config.json", "w"))