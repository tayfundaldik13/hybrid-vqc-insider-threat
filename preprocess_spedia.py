"""
SPEDIA Annotated Log Feature Extraction
"""
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

INPUT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "logs_SPEDIA_annotated_en.csv")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "spedia_behavioral_features.pkl")


FEATURE_COLS = [
    "is_session", "is_email", "is_http", "is_file", "is_command", "is_device","level", "high_level",
    "hour", "is_after_hours", "day_of_week","has_attachments", "email_size", "n_recipients",
    "is_login_failed", "is_file_modified", "is_file_deleted", "is_file_added", "has_url", "has_command",
]


def count_recipients(row):
    count = 0
    for col in ["To", "Cc", "Bcc"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            count += len(str(val).split(";"))
    return count


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SPEDIA Annotated Log Feature Extraction")
    print("=" * 60)

    print(f"\n[1/5] Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    print(f"Anomaly=1 (threat): {(df['Anomaly']==1).sum():,}")
    print(f"Anomaly=0 (normal): {(df['Anomaly']==0).sum():,}")

    print("\n[2/5] Extracting features ...")

    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["hour"]        = ts.dt.hour.fillna(12).astype(int)
    df["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)

    for act in ["session", "email", "http", "file", "command", "device"]:
        df[f"is_{act}"] = (df["Activity"] == act).astype(int)

    df["level"]      = df["Level"].fillna(0).astype(float)
    df["high_level"] = (df["level"] >= 7).astype(int)

    df["is_after_hours"] = ((df["hour"] < 8) | (df["hour"] >= 18)).astype(int)

    df["has_attachments"] = (df["Attachments"].fillna(0) > 0).astype(int)
    df["email_size"]      = df["Size"].fillna(0).astype(float)
    df["n_recipients"]    = df.apply(count_recipients, axis=1)

    df["is_login_failed"]  = (df["Action"] == "Login Failed").astype(int)
    df["is_file_modified"] = (df["Action"] == "File modified").astype(int)
    df["is_file_deleted"]  = (df["Action"] == "File deleted").astype(int)
    df["is_file_added"]    = (df["Action"] == "File added").astype(int)

    df["has_url"]     = df["Url"].notna().astype(int)
    df["has_command"] = df["Command"].notna().astype(int)

    print(f"Extracted {len(FEATURE_COLS)} features: {FEATURE_COLS}")

    print("\n[3/5] Building feature matrix ...")
    X_raw = df[FEATURE_COLS].values.astype(np.float32)
    y     = df["Anomaly"].values.astype(int)
    ids   = list(zip(df["User"].fillna("unknown"), df["Timestamp"]))

    print(f"X shape: {X_raw.shape}")
    print(f"Labels:  {y.sum():,} anomaly / {(y==0).sum():,} normal")

    print("\n[4/5] StandardScaler + PCA ...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_components = min(16, X_scaled.shape[1])
    pca   = PCA(n_components=n_components, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA: {X_scaled.shape[1]} → {X_pca.shape[1]} components "
          f"({explained:.1f}% variance explained)")

    print("\n[5/5] Saving ...")
    out = {
        "X_raw":           X_scaled.astype(np.float32),
        "X_pca":           X_pca,
        "y":               y,
        "ids":             ids,
        "feature_names":   FEATURE_COLS,
        "scaler":          scaler,
        "pca":             pca,
        "n_pca_components": X_pca.shape[1],
    }
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(out, f)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"X_raw shape:  {X_scaled.shape}")
    print(f"X_pca shape:  {X_pca.shape}")
    print(f"Labels:       {y.sum():,} anomaly / {(y==0).sum():,} normal")

    print(f"\nPCA variance per component:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        cum = pca.explained_variance_ratio_[:i+1].sum()
        print(f"     PC{i+1:2d}: {var*100:5.1f}%  (cumulative: {cum*100:5.1f}%)")

    print("=" * 60)

if __name__ == "__main__":
    main()
