import pandas as pd

# auxiliary function to validate size of merged datasets
def validate_merge_size(merged, left, right):
    print(f'shape merged dataset: {merged.shape}')
    print(f'shape left dataset: {left.shape}')
    print(f'shape right dataset: {right.shape}')
    print()

# merge stes and asal data; 
def merge_stes_asal(stes: pd.DataFrame, data_asal: pd.DataFrame) -> pd.DataFrame:
    """
    Merge training data with ASAL data using specified time constraints and ID logic.

    Parameters:
    - train (pd.DataFrame): Training data with 'pseudoPID' and 'monat'
    - data_asal (pd.DataFrame): ASAL data with 'pseudoPID', 'monat', and 'beginn_r_frist'

    Returns:
    - pd.DataFrame: Merged and filtered DataFrame
    """
    stes = stes.copy()
    data_asal = data_asal.copy()

    # Create a reference date 1 year before 'monat'
    stes['monat_t_mj'] = stes['monat'] - pd.DateOffset(years=1)

    # Rename columns to clarify origin
    stes.rename(columns={"monat": "monat_t"}, inplace=True)
    data_asal.rename(columns={"monat": "monat_a"}, inplace=True)

    # Outer join on pseudoPID
    merged = pd.merge(stes, data_asal, on='pseudoPID', how='outer')

    # Filter: outcome time is later than beginn_r_frist - 1 year
    merged = merged[merged['monat_t_mj'] <= merged['beginn_r_frist']]

    # Filter: ASAL info before outcome
    merged = merged[merged['monat_t'] > merged['beginn_r_frist']]

    # Keep earliest 'monat_a' for each (pseudoPID, monat_t) pair
    merged = merged.sort_values('monat_a').drop_duplicates(subset=['pseudoPID', 'monat_t'])

    # Final cleanup
    merged.reset_index(drop=True, inplace=True)
    merged.drop(columns=['monat_t_mj', 'monat_a'], inplace=True)
    merged.rename(columns={'monat_t': 'monat'}, inplace=True)

    return merged





