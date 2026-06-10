import pandas as pd

def sample_personas(df, sample_sizes, random_state=42):
    return df.sample(n=sample_sizes, random_state=random_state).reset_index(drop=True)