import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

scale_cols = ['Sales3', 'Sales14', 'Sales27', 'Sales35', 'Sales36', 'Sales48', 'prods',
              'Florfenicol_Family', 'Reproduction_Hormones', 'Total_Bios', 'All_Other', 'Panacur_Safeguard',
              'Total_Implants', 'Zuprevo', 'Mastitis', 'Biologicals', 'Pharmaceuticals']

lab_enc_cols = ['State', 'City']

na_cols = {'pct_ret': 0, 'State': 'unk', 'City': 'unk'}


def _fillna(df_in, na_dict):

    for key, value in na_dict.items():
        df_in[key] = df_in[key].fillna(value)

    return df_in


def _scale_cols(df_in, cols_to_scale):

    for col in cols_to_scale:
        standard_scaler = StandardScaler()
        data = df_in[col].to_numpy().reshape(-1, 1)
        df_in[col] = standard_scaler.fit_transform(data)

    return df_in


def _encode_cols(df_in, cols_to_encode):

    for col in cols_to_encode:
        label_encoder = LabelEncoder()
        df_in[col] = label_encoder.fit_transform(df_in[col])

    return df_in


def _data_quality_tests(df_in):

    if df_in.isnull().any(axis=1).sum() > 0:
        raise AssertionError("NAs in the dataset!\n%s" % df_in.isnull().head())

    for c in df_in.columns:
        if df_in[c].nunique() == 1:
            warn("Column %s is single valued" % c)


def process_features(df):

    df = _fillna(df, na_cols)
    df = _scale_cols(df, scale_cols)
    df = _encode_cols(df, lab_enc_cols)
    _data_quality_tests(df)

    return df
