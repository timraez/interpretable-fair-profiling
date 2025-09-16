import pandas as pd

# two auxiliary functions for feature engineering on beruf / job data

def add_count_column(data, column_name, values, count_column_name):
    db = data[data[column_name].isin(values)]
    col_with_count = db['pseudoSTESID'].value_counts().reset_index()
    col_with_count.columns = ['pseudoSTESID', count_column_name]
    col_with_count[count_column_name] = col_with_count[count_column_name].fillna(0)
    data_new = pd.merge(data, col_with_count,  on='pseudoSTESID', how='left')
    return data_new

def add_count_column_cond(conditions, data, column_name, values, count_column_name):
    dt = data
    for condition in conditions:
        dt = dt[dt[condition[0]] == condition[1]]
    dt = dt[dt[column_name].isin(values)]
    col_with_count = dt['pseudoSTESID'].value_counts().reset_index()
    col_with_count.columns = ['pseudoSTESID', count_column_name]
    col_with_count[count_column_name] = col_with_count[count_column_name].fillna(0)
    data_new = pd.merge(data, col_with_count,  on='pseudoSTESID', how='left')
    return data_new

def engineer_job_features(data_beruf: pd.DataFrame, avam_sbn2000: pd.DataFrame) -> pd.DataFrame:
    """
    Add a set of engineered features to the given job-related DataFrame.
    
    Parameters:
    - data_beruf: pd.DataFrame – Input data with job history and preferences
    - avam_sbn2000: pd.DataFrame – Mapping of AVAM codes to job classes (sbn2000)

    Returns:
    - pd.DataFrame – DataFrame with added feature columns
    """
    db = data_beruf.copy()

    db = add_count_column(db, 'b_ausgeuebt', ['Y'], 'anz_b_ausgeuebt')
    db = add_count_column(db, 'b_ausgeuebt', ['N'], 'anz_b_n_ausgeuebt')
    db = add_count_column(db, 'b_gesucht', ['Y'], 'anz_b_gesucht')
    db = add_count_column(db, 'cod_erfahrung', [20], 'anz_b_erf_3j')
    db = add_count_column(db, 'cod_erfahrung', [10], 'anz_b_erf_1bis3j')
    db = add_count_column(db, 'cod_erfahrung', [5], 'anz_b_erf_1j')
    db = add_count_column(db, 'cod_erfahrung', [1], 'anz_b_erf_0')

    db = add_count_column_cond([['b_gesucht', 'Y']], db, 'cod_erfahrung', [5, 10, 20], 'anz_b_mit_erf_such')
    db = add_count_column_cond([['b_ausgeuebt', 'Y'], ['b_gesucht', 'Y']], db, 'cod_ausbildungsniveau', [130, 131, 132, 133, 134, 135, 136], 'b_ges_ausg_sek')
    db = add_count_column_cond([['b_ausgeuebt', 'Y'], ['b_gesucht', 'Y']], db, 'cod_ausbildungsniveau', [150, 160, 170, 171, 172, 173, 180], 'b_ges_ausg_ter')
    db = add_count_column_cond([['b_gesucht', 'Y']], db, 'cod_qualifikation', [1], 'anz_b_ges_gelernt')
    db = add_count_column_cond([['b_gesucht', 'Y']], db, 'cod_funktion', [11, 21, 22], 'anz_b_ges_exp_quali')
    db = add_count_column_cond([['b_gesucht', 'Y']], db, 'cod_funktion', [24, 42, 43], 'anz_b_ges_exp_lehre')
    db = add_count_column_cond([['b_gesucht', 'Y']], db, 'cod_abschluss', [2], 'anz_b_ges_inl_abs')
    db = add_count_column_cond([['b_gesucht', 'Y']], db, 'cod_abschluss', [3], 'anz_b_ges_ausl_abs')

    db['b_gesucht_zuletzt'] = ((db['b_gesucht'] == 'Y') & (db['b_zuletzt'] == 'Y'))
    db = add_count_column(db, 'b_gesucht_zuletzt', [True], 'anz_b_gesucht_zuletzt')
    db.drop(columns=['b_gesucht_zuletzt'], inplace=True)

    # Merge with job class information
    db = pd.merge(db, avam_sbn2000, left_on='cod_avam', right_on='beruf_avam', how='left')
    db.drop(columns=['beruf_avam'], inplace=True)
    db['stellenklasse'] = db['sbn2000'].astype(str).str[0]

    # Add counts for job classes
    berufskl_dict = {
        '1': [['1'], 'berufskl_1'], '2': [['2'], 'berufskl_2'], '3': [['3'], 'berufskl_3'],
        '4': [['4'], 'berufskl_4'], '5': [['5'], 'berufskl_5'], '6': [['6'], 'berufskl_6'],
        '7': [['7'], 'berufskl_7'], '8': [['8'], 'berufskl_8'], '9': [['9'], 'berufskl_9']
    }
    for kl_value, (kl_list, col_name) in berufskl_dict.items():
        db = add_count_column_cond([['b_gesucht', 'Y']], db, 'stellenklasse', kl_list, col_name)

    # Drop irrelevant columns
    cols_to_drop = [
        'b_ausgeuebt', 'b_gesucht', 'b_zuletzt', 'cod_avam', 'cod_funktion', 'cod_qualifikation',
        'cod_ausbildungsniveau', 'cod_erfahrung', 'cod_abschluss', 'sbn2000', 'stellenklasse'
    ]
    db.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Drop duplicates
    db.drop_duplicates(inplace=True)

    return db



