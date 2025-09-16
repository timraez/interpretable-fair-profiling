import pandas as pd

# binned remaining duration of aufenthaltsgenehmigung (bins: one year max, two years max, more)
def add_aufenthalt_bins(df: pd.DataFrame) -> pd.DataFrame:
    # inner helper
    def calculate_time(future, now):
        return future - now
    
    # Convert dates
    df = df.copy()  # to avoid modifying the original DataFrame
    df["monat"] = pd.to_datetime(df["monat"])
    df["monat_m_j"] = df["monat"] - pd.DateOffset(years=1)
    df["dat_aufenthalt_bis"] = pd.to_datetime(df["dat_aufenthalt_bis"], errors='coerce')
    
    # Compute verbleibende Aufenthalt in days
    df["aufenth_verbl"] = df.apply(
        lambda x: calculate_time(x.dat_aufenthalt_bis, x.monat_m_j),
        axis=1
    ).dt.days  # timedelta in days
    
    # Bin into intervals
    intervals = pd.IntervalIndex.from_tuples(
        [
            (0, 365),
            (365, 730),
            (730, 10000),
        ]
    )
    df["aufenth_verbl_bins"] = pd.cut(df["aufenth_verbl"], bins=intervals)
    return df

# binned noga values (bins correspond to highest hierarchy layer of noga)
def add_noga_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    noga_intervals = pd.IntervalIndex.from_breaks(
        [
            0, 50000, 100000, 350000, 360000, 410000, 450000, 490000, 550000,
            580000, 640000, 680000, 690000, 770000, 840000, 850000, 860000,
            900000, 940000, 970000, 990000, 1000000
        ], 
        closed='left'
    )
    df['noga_bins'] = pd.cut(df['noga_letzter_ag'], bins=noga_intervals)
    return df

# binned languages (bins correspond to: some knowledge = 1-3, no knowledge = 4)
def add_language_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sprachkenntnisse = {
        'de_schriftlich': 'de_sch_bin', 
        'fr_muendlich': 'fr_mue_bin', 
        'fr_schriftlich': 'fr_sch_bin',
        'it_muendlich': 'it_mue_bin',
        'it_schriftlich': 'it_sch_bin',
        'de_ch_muendlich': 'ch_mue_bin',
        'ar_muendlich': 'ar_mue_bin',
        'en_muendlich': 'en_mue_bin',
        'en_schriftlich': 'en_sch_bin',
        'es_muendlich': 'es_mue_bin',
        'pt_muendlich': 'pt_mue_bin',
        'sc_muendlich': 'sc_mue_bin',
        'sq_muendlich': 'sq_mue_bin'
    }
    sprachk_intervals = pd.IntervalIndex.from_breaks([0, 3, 4], closed='right')
    for source_col, target_col in sprachkenntnisse.items():
        df[target_col] = pd.cut(df[source_col], bins=sprachk_intervals)
    return df


# binned ausbildungsniveaus (bins correspond to: primary, secondary, tertiary education, unknown)
def add_ausbildung_bins(df):
    interv_ausbildung = pd.IntervalIndex.from_tuples([(119, 122), (129, 137), (149, 181), (197, 200)])
    df = df.copy()
    df['ausbildung_bins'] = pd.cut(df['cod_ausbildungsniveau'], bins=interv_ausbildung)
    return df

# binned prozentsatz taggelder (bins for two values, 70-75; 76-80; values not 70, 80 should not exist...)
def add_prozentsatz_bins(df):
    interv_proz_tgg = pd.IntervalIndex.from_breaks([60, 75, 90])
    df = df.copy()
    df['prozentsatz_bins'] = pd.cut(df['prozentsatz_tgg'], bins=interv_proz_tgg)
    return df

# binned iv-code (bins: no iv, some association with iv)
def add_iv_code_bins(df):
    interv_iv = pd.IntervalIndex.from_breaks([-1, 0, 10])
    df = df.copy()
    df['iv_code_bins'] = pd.cut(df['iv_code'], bins=interv_iv)
    return df

# binned krankentaggelder (bins: no krankentaggeld, some krankentaggeld)
def add_krankentaggeld_bins(df):
    interv_krankentagg = pd.IntervalIndex.from_breaks([-1, 0, 100])
    df = df.copy()
    df['krankentagg_bins'] = pd.cut(df['krankentaggelder'], bins=interv_krankentagg)
    return df

# binned grund_pauschael (bins: no pauschale, pauschale for any reason)
def add_pauschale_bins(df):
    interv_pausch = pd.IntervalIndex.from_breaks([-2, 0, 50])
    df = df.copy()
    df['pauschale_bins'] = pd.cut(df['grund_pauschale'], bins=interv_pausch)
    return df

# add column with category codes for binary variable gender (1 encodes male, 0 encodes female)
def add_gender_codes(df):
    df = df.copy()
    df['cod_geschlecht'] = df['cod_geschlecht'].astype('category')
    df['cat_geschlecht'] = df['cod_geschlecht'].cat.codes
    return df

# add age variable (at time of of wirkungsereignis)
def add_age_variable(df):
    df = df.copy()
    df['geburtsdatum'] = pd.to_datetime(df['geburtsdatum'])
    df['alter'] = (df['monat'] - df['geburtsdatum']) / pd.Timedelta('365 days')
    df['alter'] = df['alter'].astype(int)
    return df

# add month (12 levels) of wirkungsereignis
def add_monat_kal(df):
    df = df.copy()
    df['monat_kal'] = pd.DatetimeIndex(df['monat']).month
    return df

# binned aufenthaltsstatus (CH and C = 0, rest = 1)
def add_aufenthaltsstatus_bin(df):
    df = df.copy()
    aufenthalt_bins = {'CH': 0, 'C': 0, 'B': 1, 'L': 1, 'F': 1, 'K': 1, 'N': 1, 'G': 1, 'E': 1, 'S': 1}
    df['aufenthalt_bins'] = df['aufenthaltsstatus'].map(aufenthalt_bins)
    return df

# binned taggeld_anspruch, breaks from empirical distribution (most common peaks)
def add_taggeld_anspruch_bin(df):
    df = df.copy()
    int_tgg = pd.IntervalIndex.from_breaks([0, 90, 200, 260, 380, 400, 520, 640, 760, 1000])
    df['taggeld_anspr_bins'] = pd.cut(df['taggeld_anspruch'], bins=int_tgg)
    return df

# binned suchregionen, such that kanton is mapped to grossregion
def add_suchregionen_bin(df):
    df = df.copy()
    kanton_grossreg = {
        'A': '99', 'AG': '3', 'AI': '5', 'AR': '5', 'BE': '2', 'BL': '3', 'BS': '3', 'CH': '98',
        'FL': '5', 'FR': '2', 'GE': '1', 'GL': '5', 'GR': '5', 'JU': '2', 'LU': '6', 'NE': '2',
        'NW': '6', 'OW': '6', 'SG': '5', 'SH': '5', 'SO': '2', 'SZ': '6', 'TG': '5', 'TI': '7',
        'UR': '6', 'VD': '1', 'VS': '1', 'ZG': '6', 'ZH': '4',
        '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '99': '99'
    }
    df['suchreg_gross'] = df['code_suchreg'].map(kanton_grossreg)
    return df

# replace NaNs in count columns with 0
def fillna_count_columns(df):
    df = df.copy()
    count_columns = [
        'anz_b_ausgeuebt', 'anz_b_n_ausgeuebt', 'anz_b_gesucht', 'anz_b_erf_3j',
        'anz_b_erf_1bis3j', 'anz_b_erf_1j', 'anz_b_erf_0', 'anz_b_mit_erf_such',
        'b_ges_ausg_sek', 'b_ges_ausg_ter', 'anz_b_gesucht_zuletzt',
        'berufskl_1', 'berufskl_2', 'berufskl_3', 'berufskl_4', 'berufskl_5',
        'berufskl_6', 'berufskl_7', 'berufskl_8', 'berufskl_9',
        'anz_b_ges_gelernt', 'anz_b_ges_exp_quali', 'anz_b_ges_exp_lehre',
        'anz_b_ges_inl_abs', 'anz_b_ges_ausl_abs', 'code_arbeitsform'
    ]
    for column in count_columns:
        df[column] = df[column].fillna(0).astype(int)
    return df

# drop raw columns that have been engineered above
def drop_raw_columns(db):
    """
    Drop specified columns from the dataframe.
    
    Args:
        db: Input dataframe
        
    Returns:
        dataframe: Dataframe with specified columns removed
    """
    columns_to_drop = [
        'monat', 'cod_geschlecht', 'geburtsdatum', 'dat_aufenthalt_bis',
        'cod_ausbildungsniveau', 'dat_anmeld', 'cod_esb', 'de_muendlich', 'de_schriftlich',
        'fr_muendlich', 'fr_schriftlich', 'it_muendlich', 'it_schriftlich', 'de_ch_muendlich',
        'ar_muendlich', 'en_muendlich', 'en_schriftlich', 'es_muendlich', 'pt_muendlich',
        'sc_muendlich', 'sq_muendlich', 'dat_abmeld', 'prozentsatz_tgg', 'beginn_r_frist',
        'ende_r_frist', 'taggeld_anspruch', 'anspruch', 'iv_code', 'beitragsmonate_in_rf',
        'krankentaggelder', 'cod_arbeitsform2', 'cod_arbeitsform3', 'cod_arbeitsform4', 'monat_m_j',
        'aufenth_verbl_bins', 'pseudoPID', 'noga_letzter_ag', 'grund_pauschale', 'aufenthaltsstatus',
        'ausgesteuert', 'aufenth_verbl', 'code_suchreg'
    ]
    
    return db.drop(columns=columns_to_drop)

# Create dummy variables for categorical columns.
def create_dummies(db):
    columns_for_dummies = [
        'suchreg_gross', 'cod_zivilstand', 'cod_esa', 'code_funktion',
        'cod_mobilitaet', 'code_arbeitsform', 'noga_bins',
        'de_sch_bin', 'fr_mue_bin', 'fr_sch_bin', 'it_mue_bin', 'it_sch_bin', 'ch_mue_bin',
        'ar_mue_bin', 'en_mue_bin', 'en_sch_bin', 'es_mue_bin', 'pt_mue_bin',
        'sc_mue_bin', 'sq_mue_bin', 'ausbildung_bins', 'prozentsatz_bins',
        'iv_code_bins', 'krankentagg_bins', 'pauschale_bins', 'monat_kal', 'taggeld_anspr_bins'
    ]
    
    prefix_for_dummies = [
        'cod_such_gro', 'cod_zivil', 'cod_esa', 'cod_funkt', 'cod_mobil',
        'cod_arbfo', 'noga_bins',
        'de_sch_bin', 'fr_mue_bin', 'fr_sch_bin', 'it_mue_bin', 'it_sch_bin', 'ch_mue_bin',
        'ar_mue_bin', 'en_mue_bin', 'en_sch_bin', 'es_mue_bin', 'pt_mue_bin',
        'sc_mue_bin', 'sq_mue_bin', 'ausbi_bins', 'prozent_bins',
        'iv_code_bins', 'krank_bins', 'pausc_bins', 'monat_kal', 'tgg_ans_bin'
    ]
    
    # Add dummies for categorical variables. First level dropped.
    return pd.get_dummies(
        db, 
        columns=columns_for_dummies, 
        prefix=prefix_for_dummies, 
        dtype=int, 
        drop_first=True
    )

# Convert specified float columns to integers.
def convert_floats_to_int(db):
    numerical_columns = [
        'jahr', 'ausweis_b_b1_be', 'vers_verdienst', 'beschaeftigungsgrad_vorher', 
        'vermittlungsgrad_asal', 'beitragsmonate_vor_rf'
    ]
    
    result_db = db.copy()
    for col in numerical_columns:
        result_db[col] = result_db[col].astype(int)
    
    return result_db

#Create interaction features by multiplying oral and written language proficiency columns, then drop the original separate columns.
def create_language_interaction_features(db):
    result_db = db.copy()
    
    # Create interaction features by multiplying oral and written proficiency
    result_db['fr_bin_(3, 4]'] = result_db['fr_mue_bin_(3, 4]'] * result_db['fr_sch_bin_(3, 4]']
    result_db['it_bin_(3, 4]'] = result_db['it_mue_bin_(3, 4]'] * result_db['it_sch_bin_(3, 4]']
    result_db['en_bin_(3, 4]'] = result_db['en_mue_bin_(3, 4]'] * result_db['en_sch_bin_(3, 4]']
    
    # Drop the original separate oral and written columns
    columns_to_drop = [
        'fr_mue_bin_(3, 4]', 'fr_sch_bin_(3, 4]', 
        'it_mue_bin_(3, 4]', 'it_sch_bin_(3, 4]',
        'en_mue_bin_(3, 4]', 'en_sch_bin_(3, 4]'
    ]
    
    return result_db.drop(columns=columns_to_drop)

# chaining feature engineering together to get clean data with dummies for LR
def feature_engineering_dummies(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Add engineered features and bins
    df = add_aufenthalt_bins(df)
    df = add_noga_bins(df)
    df = add_language_bins(df)
    df = add_ausbildung_bins(df)
    df = add_prozentsatz_bins(df)
    df = add_iv_code_bins(df)
    df = add_krankentaggeld_bins(df)
    df = add_pauschale_bins(df)
    df = add_gender_codes(df)
    df = add_age_variable(df)
    df = add_monat_kal(df)
    df = add_aufenthaltsstatus_bin(df)
    df = add_taggeld_anspruch_bin(df)
    df = add_suchregionen_bin(df)
    
    # Step 2: Handle missing values
    df = fillna_count_columns(df)
    
    # Step 3: Drop raw columns that have been engineered
    df = drop_raw_columns(df)
    
    # Step 4: Create dummy variables
    df = create_dummies(df)
    
    # Step 5: Convert data types
    df = convert_floats_to_int(df)
    
    # Step 6: Create language interaction features
    df = create_language_interaction_features(df)
    
    return df

# chaining feature engineering together to get clean data without dummies for other models
def feature_engineering_no_dummies(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Add engineered features and bins
    df = add_aufenthalt_bins(df)
    df = add_noga_bins(df)
    df = add_language_bins(df)
    df = add_ausbildung_bins(df)
    df = add_prozentsatz_bins(df)
    df = add_iv_code_bins(df)
    df = add_krankentaggeld_bins(df)
    df = add_pauschale_bins(df)
    df = add_gender_codes(df)
    df = add_age_variable(df)
    df = add_monat_kal(df)
    df = add_aufenthaltsstatus_bin(df)
    df = add_taggeld_anspruch_bin(df)
    df = add_suchregionen_bin(df)
    
    # Step 2: Handle missing values
    df = fillna_count_columns(df)
    
    # Step 3: Drop raw columns that have been engineered
    df = drop_raw_columns(df)
    
    return df