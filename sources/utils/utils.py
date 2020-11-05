import pandas as pd
import numpy as np
import time
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
import category_encoders


def clean_data(df, missing_values_threshol=50, show_dropped_cols=True):
    """Helper function for data cleansing"""
    def percent_na(x):
        pct = 100*(x.isna().sum() / x.shape[0]).round(2)
        return pct

    df_ = df.loc[:, :'Q105']
    to_drop_1 = ['COUNTRY', 'COUNTRY.BY.REGION.NO',
                 'COUNTRY.BY.REGION', 'RESPNO',
                 'DATEINTR', 'STRTIME', 'THISINT',
                 'PREVINT', 'ENDTIME', 'LENGTH',
                 'CALLS'
                 ]  # Dropping unecessary variables
    to_drop_2 = [col for col in df.columns if 'NOCALL' in col]
    drop_cols = to_drop_1 + to_drop_2
    df_ = df_.drop(drop_cols, axis=1)
    dropped_cols = pd.DataFrame(columns=['Variables',
                                         'Percentage missing data'
                                         ])
    n = 0  # Attributes to drop iterator
    for col in list(df_):
        pmissing = percent_na(df_[col])
        if pmissing >= missing_values_threshol:
            dropped_cols.loc[n, :] = [col, pmissing]
            df_.drop(col, axis=1, inplace=True)
            n += 1
    if show_dropped_cols:
        print('Columns not satisfying the missing data treshold are:')
        print(dropped_cols)
    return df_


def get_age_groups(ages):
    """Helper function for computing age groups"""
    cat_ages = []
    for i in range(ages.shape[0]):
        if ages[i] < 25:
            cat_ages.append('<25')
        elif ages[i] <= 35:
            cat_ages.append('25-35')
        elif ages[i] <= 45:
            cat_ages.append('36-45')
        elif ages[i] <= 55:
            cat_ages.append('46-55')
        elif ages[i] <= 65:
            cat_ages.append('56-65')
        elif ages[i] <= 75:
            cat_ages.append('66-75')
        else:
            cat_ages.append('>76')
    return pd.Series(cat_ages)


def encode_col_values(col, dic):
    """Helper function for variables encoding"""
    return col.map(dic)


def get_most_corruption_type_by_country(df_bribes, countries):
    """Helper function for computing the most type of bribe in each country.

    The computation is as follow for each types:
    For each country compute the compute the frecencies of the answers:
    - Never
    - Once or twice
    - A few times
    - Often
    - No experience with this in the past year

    Of each of the bribe type:
    - Document or permit
    - water or sanitation services
    - Treatment at local health clinic or hospital
    - Avoid problem with police
    - School placement,
    - Election incentives
    """
    df_bribes.columns = ['Document or permit', 'water or sanitation services',
                         'Treatment at local health clinic or hospital',
                         'Avoid problem with police', 'School placement',
                         'Election incentives'
                         ]
    most_corruption_type = []
    for country in countries.unique():
        max_ = 0.0
        type_ = ''
        for col in df_bribes.columns:
            counts = df_bribes.loc[countries == country, col].value_counts()
            if 'At least once' in counts.index:
                n_at_least_once = counts['At least once']
                pertcent = 100*np.round(n_at_least_once/counts.sum(), 3)
                if pertcent > max_:
                    max_ = pertcent
                    type_ = col
        most_corruption_type.append({'country': country,
                                     'type': type_,
                                     'percentage': max_
                                     }
                                    )
    return most_corruption_type


def get_importante_variables(df_cluster, target, n_variables=10):
    rf = ensemble.RandomForestClassifier(random_state=6)
    rf.fit(df_cluster, target)
    f_imp = rf.feature_importances_
    imp_vars_mask = np.argsort(f_imp)[::-1]
    f_imp = f_imp[imp_vars_mask][:n_variables]
    imp_vars = df_cluster.columns[imp_vars_mask][:n_variables]
    return pd.Series(f_imp, index=imp_vars).sort_values()


def count_encoding(df):
    """Helper function for count encoding"""
    for col in list(df):
        counts = df[col].value_counts()
        df[col + '_count'] = df[col].map(counts)
    return df


def target_encode(df, target):
    encoder = category_encoders.TargetEncoder(cols=list(df))
    encoder.fit(df, target)
    df_targ_enc = encoder.transform(df)
    return df_targ_enc


def split(df, target, test_size=0.3):
    """Helper function for splitting the data
       into a train and test sets giving a test set size
    """
    x_train, x_val, y_train, y_val = (model_selection
                                      .train_test_split(df,
                                                        target,
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        random_state=10
                                                        )
                                      )
    return x_train, x_val, y_train, y_val


def initialize_attributes(self_, x_train, x_val, y_train, y_val):
    self_.x_train = x_train
    self_.x_val = x_val
    self_.y_train = y_train
    self_.y_val = y_val


def get_semi_supervised_data(x_train, x_val,
                             y_train, y_val,
                             preds_prob, threshold
                             ):
    """
    Helper function for creating data for semi-supervised learning.
    The process is the following:
    Given the predictions of a model and a threshold, we compute the indexes
    of the predictions having a value greater or equal to the threshold and we
    append the validation data corresponding to the computed indexes to the training set.
    """
    preds = pd.Series(preds_prob)
    mask = (preds >= threshold).values
    indexes = y_val[mask].index
    x_train_new = pd.concat([x_train, x_val.loc[indexes]])
    y_train_new = np.append(y_train, y_val[indexes])
    return x_train_new, y_train_new


def CV(model, train, target, n_splits=5):
    oof = pd.Series(np.zeros(train.shape[0]), index=train.index)
    time_start = time.time()
    kf = model_selection.KFold(n_splits=n_splits, random_state=4)
    for i, (tr_idx, te_idx) in enumerate(kf.split(train, target)):
        x_tr, x_te = train.iloc[tr_idx], train.iloc[te_idx]
        y_tr, y_te = target.iloc[tr_idx], target.iloc[te_idx]
        model.fit(x_tr, y_tr)
        tmp = model.predict_proba(x_te)[:, 1]
        print('Fold {0} auc {1}'.format(i + 1, metrics.roc_auc_score(y_te, tmp)))
        oof.iloc[te_idx] = tmp
    time_end = time.time()
    print('Time compexity {}'.format(time_end - time_start))
    return model, oof
