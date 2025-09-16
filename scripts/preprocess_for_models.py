

def transform_datatype_for_ebm(data):
    data = data.copy()
    cat_columns = ['cod_zivilstand', 'cod_esa', 'code_funktion', 'cod_mobilitaet',
       'ausweis_b_b1_be', 'code_arbeitsform',
       'noga_bins', 'de_sch_bin', 'fr_mue_bin', 'fr_sch_bin', 'it_mue_bin',
       'it_sch_bin', 'ch_mue_bin', 'ar_mue_bin', 'en_mue_bin', 'en_sch_bin',
       'es_mue_bin', 'ausbildung_bins', 'prozentsatz_bins', 'iv_code_bins',
       'krankentagg_bins', 'pauschale_bins', 'cat_geschlecht',
        'aufenthalt_bins', 'taggeld_anspr_bins', 'suchreg_gross']
    for col in cat_columns:
        data[col] = data[col].astype('category')
    return data

def sanitize_column_names_for_xgb(df):
    """
    Replace forbidden characters in column names:
    [ → {, ] → }, < → ^
    """
    df = df.copy()
    df.columns = df.columns.str.translate(str.maketrans({"[": "{", "]": "}", "<": "^"}))
    return df

def apply_preprocessing_to_folds(folds, preprocessing_fn):
    """
    Apply a preprocessing function to CV_X and holdout_X in each fold.
    
    Parameters:
    - folds: list of dicts, each with keys including 'CV_X' and 'holdout_X'
    - preprocessing_fn: function that takes a DataFrame and returns a transformed DataFrame

    Returns:
    - A new list of folds with transformed 'CV_X' and 'holdout_X'
    """
    new_folds = []
    for fold in folds:
        # Apply the preprocessing function to CV_X and holdout_X
        processed_fold = fold.copy()
        processed_fold['CV_X'] = preprocessing_fn(fold['CV_X'])
        processed_fold['holdout_X'] = preprocessing_fn(fold['holdout_X'])
        new_folds.append(processed_fold)

    return new_folds