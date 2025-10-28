import pandas as pd
import numpy as np
import os
import argparse

# Opsi display
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Abaikan warnings
import warnings
warnings.filterwarnings('ignore')

DEFAULT_INPUT_FILE = os.path.join('..', 'diabetes_raw.csv')
DEFAULT_OUTPUT_DIR = 'diabetes_preprocessing'           
DEFAULT_OUTPUT_FILE = os.path.join(DEFAULT_OUTPUT_DIR, 'diabetes_preprocessing.csv')
TARGET_COL = 'Outcome'

def load_data(file_path):
    """Memuat data CSV dengan penanganan error."""
    try:
        df = pd.read_csv(file_path, delimiter=',')
        print(f"Dataset berhasil dimuat dari '{file_path}'.")
        if 'Unnamed: 0' in df.columns:
            print("Header salah? Mencoba muat ulang dengan header=1...")
            df = pd.read_csv(file_path, delimiter=',', header=1)
        return df
    except FileNotFoundError:
        print(f"ERROR: File tidak ditemukan di '{file_path}'")
        return None
    except Exception as e:
        print(f"Error memuat file '{file_path}': {e}")
        try:
             print("Mencoba delimiter ';'...")
             df = pd.read_csv(file_path, delimiter=';', encoding='latin1', on_bad_lines='skip', header=1) # Asumsi header=1 dari kasus sebelumnya
             print(f"Dataset berhasil dimuat dari '{file_path}' (delimiter ';').")
             return df
        except Exception as e_inner:
             print(f"Gagal memuat file dengan kedua delimiter. Error: {e_inner}")
             return None


def preprocess_data(df):
    """Melakukan preprocessing: drop duplikat, handle outliers."""
    if df is None or df.empty:
        print("DataFrame kosong, preprocessing dibatalkan.")
        return None

    print("\nMemulai Preprocessing...")
    df_prep = df.copy()

    # 1. Drop Duplicates
    len_before_drop = len(df_prep)
    df_prep.drop_duplicates(inplace=True)
    print(f"- Data setelah drop duplikat: {len(df_prep)} (dibuang {len_before_drop - len(df_prep)})")

    # 2. Handle Outliers (IQR) pada Fitur (kecuali target)
    if TARGET_COL in df_prep.columns:
         cols_to_check_outliers = df_prep.drop(TARGET_COL, axis=1).select_dtypes(include=np.number).columns
    else:
         cols_to_check_outliers = df_prep.select_dtypes(include=np.number).columns
         
    print("- Penanganan Outliers (IQR) pada Fitur...")
    count_before_outlier = len(df_prep)
    for col in cols_to_check_outliers:
        if col in df_prep.columns and pd.api.types.is_numeric_dtype(df_prep[col]):
            Q1 = df_prep[col].quantile(0.25)
            Q3 = df_prep[col].quantile(0.75)
            IQR_val = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR_val
            upper_bound = Q3 + 1.5 * IQR_val
            df_prep = df_prep[(df_prep[col] >= lower_bound) & (df_prep[col] <= upper_bound)]
    
    count_after_outlier = len(df_prep)
    print(f"- Data setelah handle outliers: {count_after_outlier} (dibuang {count_before_outlier - count_after_outlier} baris)")

    # 3. Handle Missing Values (jika ada sisa)
    if df_prep.isnull().sum().sum() > 0:
         rows_before_na = len(df_prep)
         df_prep.dropna(inplace=True)
         rows_after_na = len(df_prep)
         print(f"- Menghapus sisa missing values: {rows_after_na} (dibuang {rows_before_na - rows_after_na})")
    else:
         print("- Tidak ada sisa missing values.")

    print("Preprocessing Selesai.")
    return df_prep

def save_data(df, output_path):
    """Menyimpan dataframe ke file CSV."""
    if df is None or df.empty:
        print("Tidak ada data untuk disimpan.")
        return

    try:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Folder '{output_dir}' dibuat.")
            
        df.to_csv(output_path, index=False)
        print(f"Data bersih berhasil disimpan di: '{output_path}'")
    except Exception as e:
        print(f"Gagal menyimpan file CSV: {e}")

# Main execution block
if __name__ == "__main__":
    # Setup argumen parser
    parser = argparse.ArgumentParser(description="Script Preprocessing Data Diabetes")
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_FILE, 
                        help=f"Path file input CSV (default: {DEFAULT_INPUT_FILE})")
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE, 
                        help=f"Path file output CSV (default: {DEFAULT_OUTPUT_FILE})")
    
    args = parser.parse_args()

    # Jalankan pipeline
    raw_df = load_data(args.input)
    cleaned_df = preprocess_data(raw_df)
    save_data(cleaned_df, args.output)