# automate_Amanda-Mahendra.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


def load_data(script_dir, parent_dir):
    """Load dataset Wine dari sklearn dan simpan sebagai raw CSV."""
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    
    # Tentukan path penyimpanan raw data di root Eksperimen_SML_Amanda-Mahendra
    raw_csv_path = os.path.join(parent_dir, 'wine_raw.csv')
    df.to_csv(raw_csv_path, index=False)
    print(f"[INFO] Dataset berhasil dimuat. Shape: {df.shape}")
    print(f"[INFO] Raw data disimpan ke: {raw_csv_path}")
    return df


def preprocess_data(df):
    """
    Melakukan preprocessing data:
    1. Memisahkan fitur dan target
    2. Menangani missing values
    3. Feature scaling dengan StandardScaler
    4. Train-test split
    """
    # Pisahkan fitur dan target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Tangani missing values (jika ada)
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
        print("[INFO] Missing values ditangani dengan median.")
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Train-test split (80:20 secara stratifikasi)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Preprocessing selesai.")
    print(f"[INFO] Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_scaled_df, y, X_train, X_test, y_train, y_test


def save_preprocessed_data(X_scaled_df, y, script_dir):
    """Simpan data yang sudah diproses sebagai CSV di folder preprocessing."""
    df_preprocessed = X_scaled_df.copy()
    df_preprocessed['target'] = y.values
    
    output_path = os.path.join(script_dir, 'wine_preprocessing.csv')
    df_preprocessed.to_csv(output_path, index=False)
    print(f"[INFO] Data preprocessing disimpan ke: {output_path}")
    return df_preprocessed


def main():
    """Fungsi utama untuk menjalankan pipeline preprocessing."""
    print("=" * 50)
    print("AUTOMATED PREPROCESSING - Wine Dataset")
    print("=" * 50)
    
    # Tentukan direktori berdasarkan lokasi skrip ini
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    df = load_data(script_dir, parent_dir)
    X_scaled_df, y, X_train, X_test, y_train, y_test = preprocess_data(df)
    df_preprocessed = save_preprocessed_data(X_scaled_df, y, script_dir)
    
    print("\n[SUCCESS] Preprocessing pipeline selesai!")
    print(f"Output shape: {df_preprocessed.shape}")
    return df_preprocessed


if __name__ == "__main__":
    main()
