import pandas as pd
from pathlib import Path

def load_stes_data(data_dir: str, years: list) -> pd.DataFrame:
    """Load and concatenate STES data for specified years."""
    dfs = []
    for year in years:
        file_path = Path(data_dir) / f"dat_stes_{year}_pseudonymisiert.csv"
        df = pd.read_csv(file_path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_auxiliary_data(data_dir: str) -> dict:
    """Load auxiliary datasets into a dictionary."""
    return {
        'arbeitsformen': pd.read_csv(Path(data_dir) / 'dat_arbeitsformen_pseudonymisiert.csv'),
        'asal': pd.read_csv(Path(data_dir) / 'dat_asal_pseudonymisiert.csv'),
        'regio': pd.read_csv(Path(data_dir) / 'dat_suchregion_pseudonymisiert.csv'),
        'beruf': pd.read_csv(Path(data_dir) / 'dat_berufe_pseudonymisiert.csv'),
        'outcome': pd.read_csv(Path(data_dir) / 'dat_outcome_pseudonymisiert.csv'),
        'avam_sbn2000': pd.read_csv(Path(data_dir) / 'berufsliste_mstr_avam_sbn2000_short.csv', sep=';'),
    }

def clean_stes_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean STES data by dropping columns, formatting dates, and removing duplicates."""
    df = df.drop(columns=['rav', 'muttersprache', 'num_bfs'], errors='ignore')

    # Save original 'geburtsdatum' as string
    geburt_raw = df['geburtsdatum'].astype(str)

    # Try parsing 'YYYY.MM'
    df['geburtsdatum'] = pd.to_datetime(geburt_raw, format='%Y.%m', errors='coerce')

    # Convert monat to datetime
    df['monat'] = pd.to_datetime(df['monat'], format='%Y%m', errors='coerce')

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df

def clean_asal_data(df: pd.DataFrame) -> pd.DataFrame:
    ## drop unnecessary column
    df = df.drop(columns=['num_rf'])

    ## format monat column in datetime format
    df["monat"] = pd.to_datetime(df["monat"], format= "%Y%m")

    ## format beginn rahmenfrist column in datetime format
    df['beginn_r_frist'] = pd.to_datetime(df['beginn_r_frist'], format = "%Y%m%d")

    ## in line with feedback from MG, remove double rows for (month, PID) pairs.
    # according to MG, these com from double Rahmenfristen -> only use largest beginn_r_frist and only include anspruch == 1.

    ## only keep rows with anspruch == 1
    df = df.loc[df['anspruch'] == 1]

    ## of rows with duplicate values in (month,PID) pairs, keep row with highest value in beginn_r_frist
    df = df.sort_values('beginn_r_frist', ascending=False).drop_duplicates(subset=['monat', 'pseudoPID']).sort_index()

    ## drop rows with tgg == 0
    df = df[df['prozentsatz_tgg'] != 0]

    return df



