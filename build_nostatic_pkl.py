import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

STATIC = {"ocean_O", "ocean_C", "ocean_E", "ocean_A", "ocean_N",
          "is_ITAdmin", "functional_unit_id"}


def build(src, dst, expect_pos):
    o = pickle.load(open(src, "rb"))
    keep = [i for i, n in enumerate(o["feature_names"]) if n not in STATIC]
    names = [o["feature_names"][i] for i in keep]
    X_raw = o["X_raw"][:, keep]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca = PCA(n_components=16, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)
    assert int(o["y"].sum()) == expect_pos, o["y"].sum()
    assert len(X_pca) == len(o["y"]) == len(o["ids"])
    out = dict(o, X_raw=X_raw.astype(np.float32), X_pca=X_pca, feature_names=names,
               scaler=scaler, pca=pca, n_pca_components=16)
    pickle.dump(out, open(dst, "wb"))
    print(f"{dst}: features {len(o['feature_names'])}->{len(names)}  "
          f"PCA var={pca.explained_variance_ratio_.sum():.3f}  pos={out['y'].sum()}")


if __name__ == "__main__":
    build("data/behavioral_features_fixed.pkl", "data/behavioral_features_fixed_nostatic.pkl", 966)
    build("r5.2/data/behavioral_features_fixed.pkl", "r5.2/data/behavioral_features_fixed_nostatic.pkl", 1307)
