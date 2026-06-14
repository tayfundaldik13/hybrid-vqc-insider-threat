"""
CERT r5.2 Behavioral Feature Extraction
"""
import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

R52_DIR      = "your directory"
ANSWERS_DIR  = "your directory"
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_PATH  = os.path.join(OUTPUT_DIR, "behavioral_features.pkl")

WORK_START   = 8
WORK_END     = 18
CHUNK_SIZE   = 500_000

JOB_SITES = (
    "simplyhired|indeed\\.com|linkedin|glassdoor|monster\\.com|"
    "careerbuilder|job-hunt|jobhuntersbible|hotjobs|aol\\.com/jobs|"
    "dice\\.com|jobsearch|careers\\."
)


def parse_dt(series):
    return pd.to_datetime(series, format="%m/%d/%Y %H:%M:%S", errors="coerce")


def after_hours(hour_series):
    return (hour_series < WORK_START) | (hour_series >= WORK_END)


def get_malicious_users():
    df = pd.read_csv(os.path.join(ANSWERS_DIR, "insiders.csv"))
    r52 = df[df["dataset"] == 5.2]["user"].dropna().unique()
    print(f"Malicious users (r5.2): {len(r52)}")
    return set(r52)


def load_ldap():
    ldap_dir = os.path.join(R52_DIR, "LDAP")
    frames = [pd.read_csv(os.path.join(ldap_dir, f))
              for f in sorted(os.listdir(ldap_dir)) if f.endswith(".csv")]
    df = pd.concat(frames).drop_duplicates(subset="user_id")

    df["is_ITAdmin"] = (df["role"] == "ITAdmin").astype(int)

    def extract_unit(val):
        m = re.match(r"^(\d+)", str(val))
        return int(m.group(1)) if m else 0

    df["functional_unit_id"] = df["functional_unit"].apply(extract_unit)

    return df[["user_id", "is_ITAdmin", "functional_unit_id"]].rename(
        columns={"user_id": "user"}
    )


def load_ocean():
    df = pd.read_csv(os.path.join(R52_DIR, "psychometric.csv"))
    return df[["user_id", "O", "C", "E", "A", "N"]].rename(
        columns={"user_id": "user",
                 "O": "ocean_O", "C": "ocean_C", "E": "ocean_E",
                 "A": "ocean_A", "N": "ocean_N"}
    )


def load_logon():
    print("   Loading logon.csv...")
    df = pd.read_csv(
        os.path.join(R52_DIR, "logon.csv"),
        usecols=["date", "user", "pc", "activity"],
    )
    dt = parse_dt(df["date"])
    df["date_only"] = dt.dt.date
    df["hour"]      = dt.dt.hour
    df["is_logon"]  = (df["activity"] == "Logon").astype(int)
    df["is_logoff"] = (df["activity"] == "Logoff").astype(int)
    df["ah_logon"]  = (df["is_logon"] == 1) & after_hours(df["hour"])

    agg = df.groupby(["user", "date_only"]).agg(
        logon_count      =("is_logon",  "sum"),
        logoff_count     =("is_logoff", "sum"),
        after_hours_logon=("ah_logon",  "sum"),
        unique_pcs       =("pc",        "nunique"),
    ).reset_index()
    print(f"Logon user-days: {len(agg):,}")
    return agg


def load_device():
    print("Loading device.csv...")
    df = pd.read_csv(
        os.path.join(R52_DIR, "device.csv"),
        usecols=["date", "user", "activity"],
    )
    dt = parse_dt(df["date"])
    df["date_only"]   = dt.dt.date
    df["is_connect"]  = (df["activity"] == "Connect").astype(int)
    df["is_disconn"]  = (df["activity"] == "Disconnect").astype(int)

    agg = df.groupby(["user", "date_only"]).agg(
        device_connect   =("is_connect", "sum"),
        device_disconnect=("is_disconn", "sum"),
    ).reset_index()
    print(f"   Device user-days: {len(agg):,}")
    return agg


def load_email():
    print("Loading email.csv...")
    df = pd.read_csv(
        os.path.join(R52_DIR, "email.csv"),
        usecols=["date", "user", "to", "size", "attachments", "activity"],
    )
    df = df[df["activity"] == "Send"].copy()

    dt = parse_dt(df["date"])
    df["date_only"]    = dt.dt.date
    df["hour"]         = dt.dt.hour
    df["ah_email"] = after_hours(df["hour"]).astype(int)
    df["external"] = (~df["to"].str.contains("@dtaa.com", na=False)).astype(int)

    df_attach = df.dropna(subset=["attachments"])

    agg = df.groupby(["user", "date_only"]).agg(
        emails_sent      =("size",    "count"),
        total_email_size =("size",    "sum"),
        after_hours_email=("ah_email","sum"),
        external_emails  =("external","sum"),
    ).reset_index()

    agg_attach = df_attach.groupby(["user", "date_only"]).agg(
        emails_with_attach=("attachments", "count"),
    ).reset_index()

    agg = agg.merge(agg_attach, on=["user", "date_only"], how="left")
    agg["emails_with_attach"] = agg["emails_with_attach"].fillna(0).astype(int)
    print(f"   Email user-days: {len(agg):,}")
    return agg


def load_file():
    print("Loading file.csv...")
    df = pd.read_csv(
        os.path.join(R52_DIR, "file.csv"),
        usecols=["date", "user", "filename", "to_removable_media", "from_removable_media"],
    )
    dt = parse_dt(df["date"])
    df["date_only"] = dt.dt.date
    df["ext"]       = df["filename"].str.extract(r"\.([^.]+)$")[0].fillna("unknown")

    df["to_rem"]   = df["to_removable_media"].map(
        {True: 1, False: 0, "True": 1, "False": 0}
    ).fillna(0).astype(int)
    df["from_rem"] = df["from_removable_media"].map(
        {True: 1, False: 0, "True": 1, "False": 0}
    ).fillna(0).astype(int)

    agg = df.groupby(["user", "date_only"]).agg(
        files_copied              =("filename", "count"),
        unique_file_exts          =("ext",      "nunique"),
        to_removable_media_count  =("to_rem",   "sum"),
        from_removable_media_count=("from_rem", "sum"),
    ).reset_index()
    print(f"File user-days: {len(agg):,}")
    return agg


def load_decoy():
    print("Loading decoy_file.csv...")
    decoy_files = set(
        pd.read_csv(os.path.join(R52_DIR, "decoy_file.csv"))["decoy_filename"].str.lower()
    )

    file_df = pd.read_csv(
        os.path.join(R52_DIR, "file.csv"),
        usecols=["date", "user", "filename"],
    )
    dt = parse_dt(file_df["date"])
    file_df["date_only"] = dt.dt.date
    file_df["is_decoy"]  = file_df["filename"].str.lower().isin(decoy_files).astype(int)

    agg = file_df.groupby(["user", "date_only"]).agg(
        decoy_accesses=("is_decoy", "sum"),
    ).reset_index()
    print(f"Decoy - user-days with any access: {(agg['decoy_accesses'] > 0).sum():,}")
    return agg


def load_http():
    print("Loading http.csv ...")
    http_path = os.path.join(R52_DIR, "http.csv")
    chunks = []

    for i, chunk in enumerate(
        pd.read_csv(http_path, usecols=["date", "user", "url"], chunksize=CHUNK_SIZE)
    ):
        dt= parse_dt(chunk["date"])
        chunk["date_only"] = dt.dt.date
        chunk["hour"]      = dt.dt.hour
        chunk["ah_http"]   = after_hours(chunk["hour"]).astype(int)
        chunk["job_site"]  = chunk["url"].str.contains(
            JOB_SITES, na=False, case=False, regex=True
        ).astype(int)

        agg = chunk.groupby(["user", "date_only"]).agg(
            http_visits     =("url",      "count"),
            after_hours_http=("ah_http",  "sum"),
            job_site_visits =("job_site", "sum"),
        ).reset_index()
        chunks.append(agg)

        if (i + 1) % 10 == 0:
            print(f"chunk {i+1} processed...")

    http_df = (
        pd.concat(chunks)
        .groupby(["user", "date_only"])
        .sum()
        .reset_index()
    )
    print(f"HTTP user-days: {len(http_df):,}")
    return http_df


def merge_features(logon, device, email, file_df, http, decoy, ldap, ocean):
    print("Merging all feature tables...")

    df = logon.copy()

    for other in [device, email, file_df, http]:
        df = df.merge(other, on=["user", "date_only"], how="left")

    df = df.merge(decoy, on=["user", "date_only"], how="left")

    activity_cols = [
        "device_connect", "device_disconnect",
        "emails_sent", "emails_with_attach", "total_email_size",
        "after_hours_email", "external_emails",
        "files_copied", "unique_file_exts",
        "to_removable_media_count", "from_removable_media_count",
        "http_visits", "after_hours_http", "job_site_visits",
        "decoy_accesses",
    ]
    df[activity_cols] = df[activity_cols].fillna(0)

    df = df.merge(ldap,  on="user", how="left")
    df = df.merge(ocean, on="user", how="left")

    ldap_ocean_cols = ["is_ITAdmin", "functional_unit_id",
                       "ocean_O", "ocean_C", "ocean_E", "ocean_A", "ocean_N"]
    for col in ldap_ocean_cols:
        df[col] = df[col].fillna(df[col].median())

    df["total_activity"]       = (df["logon_count"] + df["device_connect"] +
                                   df["emails_sent"] + df["files_copied"] + df["http_visits"])
    df["after_hours_ratio"]    = df["after_hours_logon"] / (df["logon_count"] + 1)
    df["device_to_file_ratio"] = df["device_connect"] / (df["files_copied"] + 1)

    print(f"   Merged: {len(df):,} user-days, {df['user'].nunique()} users")
    return df


FEATURE_COLS = [
    "logon_count", "logoff_count", "after_hours_logon", "unique_pcs",
    "device_connect", "device_disconnect",
    "emails_sent", "emails_with_attach", "total_email_size",
    "after_hours_email", "external_emails",
    "files_copied", "unique_file_exts",
    "to_removable_media_count", "from_removable_media_count",
    "http_visits", "after_hours_http", "job_site_visits",
    "ocean_O", "ocean_C", "ocean_E", "ocean_A", "ocean_N",
    "is_ITAdmin", "functional_unit_id",
    "decoy_accesses",
    "total_activity", "after_hours_ratio", "device_to_file_ratio",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("CERT r5.2 Behavioral Feature Extraction")
    print("=" * 60)

    print("\n[1/10] Loading malicious users...")
    malicious = get_malicious_users()

    print("\n[2/10] Loading LDAP...")
    ldap = load_ldap()

    print("\n[3/10] Loading OCEAN (psychometric)...")
    ocean = load_ocean()

    print("\n[4/10] Loading logon...")
    logon = load_logon()

    print("\n[5/10] Loading device...")
    device = load_device()

    print("\n[6/10] Loading email (Send only)...")
    email = load_email()

    print("\n[7/10] Loading file (+ removable media)...")
    file_df = load_file()

    print("\n[8/10] Loading decoy files...")
    decoy = load_decoy()

    print("\n[9/10] Loading HTTP (slow, chunked)...")
    http = load_http()

    print("\n[10/10] Merging + scaling + PCA...")
    df = merge_features(logon, device, email, file_df, http, decoy, ldap, ocean)

    df["label"] = df["user"].isin(malicious).astype(int)
    print(f"Normal user-days:  {(df['label']==0).sum():,}")
    print(f"Insider user-days: {(df['label']==1).sum():,}")

    X_raw = df[FEATURE_COLS].values.astype(np.float32)
    y     = df["label"].values.astype(int)
    ids   = list(zip(df["user"], df["date_only"]))

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca   = PCA(n_components=16, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)
    explained = pca.explained_variance_ratio_.sum()
    print(f"   PCA: {X_scaled.shape[1]} → {X_pca.shape[1]} components "
          f"(explained variance: {explained:.3f})")

    out = {
        "X_raw":            X_scaled.astype(np.float32),
        "X_pca":            X_pca,
        "y":                y,
        "ids":              ids,
        "feature_names":    FEATURE_COLS,
        "scaler":           scaler,
        "pca":              pca,
        "n_pca_components": X_pca.shape[1],
    }
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(out, f)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"X_raw shape:  {X_scaled.shape}")
    print(f"X_pca shape:  {X_pca.shape}")
    print(f"Labels:       {y.sum()} insider / {(y==0).sum()} normal")
    print("=" * 60)


if __name__ == "__main__":
    main()
