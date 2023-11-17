import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    x = df.drop(columns=[df.columns[-1]])
    y = df.iloc[:, -1]
    return x.to_numpy(), y.to_numpy()
