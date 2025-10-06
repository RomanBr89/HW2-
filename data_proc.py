import pandas as pd


def check_missing(df, columns):
    missing = df[columns].isnull().sum()
    print("Пропущенные значения по колонкам:")
    print(missing)
    return missing

def fill_missing(df, column, method='mean'):
    if method == 'mean':
        df[column] = df[column].fillna(df[column].mean())
    elif method == 'median':
        df[column] = df[column].fillna(df[column].median())
    return df
