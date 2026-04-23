import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    # Binary classification (toxic vs non-toxic)
    df['toxic_label'] = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].max(axis=1)

    df = df[['comment_text', 'toxic_label']]
    df = df.dropna()

    return df
