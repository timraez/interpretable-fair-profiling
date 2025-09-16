def get_input_output(df, dummies='Y'):
    df = df.copy()
    
    df['f01_zug_lz_al_12_w2z'] = df['f01_zug_lz_al_12_w2z'].astype(int)

    # Always drop these columns
    drop_cols = ['Unnamed: 0', 'pseudoSTESID', 'f01_zug_lz_al_12_w2z', 'jahr']

    # Additional columns to drop depending on dummy setting
    if dummies == 'Y':
        drop_cols += ['pt_mue_bin_(3, 4]', 'sc_mue_bin_(3, 4]', 'sq_mue_bin_(3, 4]']
    else:
        drop_cols += ['pt_mue_bin', 'sc_mue_bin', 'sq_mue_bin']

    X = df.drop(columns=drop_cols)
    Y = df['f01_zug_lz_al_12_w2z']

    return X, Y

# get moving window indices for cv split from dataset, based on jahr column
def get_mw_fold_index(data, target_years):
    fold_index = []
    for i in target_years:
        train_index = data.index[data['jahr'] == (i-1)]
        test_index = data.index[data['jahr'] == i]
        fold_index.append((train_index, test_index))
    return fold_index

# get appropriate train and test folds
def get_tscv_data(X, Y, fold_index):
    folds = []
    for train_index, test_index in fold_index:    
        CV_X = X.iloc[train_index]
        CV_Y = Y.iloc[train_index]
        
        holdout_X = X.iloc[test_index]
        holdout_Y = Y.iloc[test_index]
        
        fold = {'CV_X': CV_X, 'CV_Y': CV_Y,'holdout_X': holdout_X, 'holdout_Y': holdout_Y}
        folds.append(fold)
    return folds

def get_moving_windows_split_train(df, target_years, dummies='Y'):
    df = df.copy()
    X, Y = get_input_output(df, dummies=dummies)
    fold_index = get_mw_fold_index(df, target_years)
    folds = get_tscv_data(X, Y, fold_index)
    return folds


def get_test_split(train, test, target_year, dummies='Y'):
    train = train.copy()
    test = test.copy()

    # Extract last year from training data
    train_last_year = train[train['jahr'] == (target_year - 1)]
    CV_X, CV_Y = get_input_output(train_last_year, dummies=dummies)

    # Process test data
    holdout_X, holdout_Y = get_input_output(test, dummies=dummies)

    return [{'CV_X': CV_X, 'CV_Y': CV_Y, 'holdout_X': holdout_X, 'holdout_Y': holdout_Y}]


def print_folds_shape(folds):
    counter = 0
    for fold in folds:
        print('fold number '+str(counter))
        print('shape of training fold: '+str(fold['CV_X'].shape))
        print('shape of validation fold: '+str(fold['holdout_X'].shape))
        counter = counter +1

# [LEGACY FUNCTION, NOT USED] get times series indices for cv split from dataset, based on jahr column
def get_ts_fold_index(data, target_years):
    fold_index = []
    for i in target_years:
        train_index = data.index[data['jahr'] < i]
        test_index = data.index[data['jahr'] == i]
        fold_index.append((train_index, test_index))
    return fold_index