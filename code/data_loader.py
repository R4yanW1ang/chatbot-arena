import os
import hydra
import torch
import pandas as pd
import re

from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="data")
def load_dataset(cfg: DictConfig):
    data_dir = cfg['data_dir']

    # Load CSV using pandas
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'), encoding='utf-8')

    if cfg['debug']:
        df_train = df_train.head(cfg['num_sample'])  # for testing purpose

    # Format the dataset
    df_train = format_df(df_train)

    output_file = os.path.join(data_dir, 'formatted_train.csv')
    df_train.to_csv(output_file, index=False, errors='replace')  # Save without the index column


def process_text(text: str) -> str:
    # Handle encoding issues
    cleaned_text = text.encode('utf-8', 'ignore').decode('utf-8')
    # Remove excessive whitespace and newlines
    return ' '.join(cleaned_text.split())[1:-1]

def format_df(df):
    # Apply process_text to each column in the DataFrame
    prompt = ["<prompt>: " + process_text(t) for t in df["prompt"]]
    response_a = ["\n\n<response_a>: " + process_text(t) for t in df["response_a"]]
    response_b = ["\n\n<response_b>: " + process_text(t) for t in df["response_b"]]

    # Combine the columns into a new 'input' column
    df['input'] = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]

    return df


if __name__ == "__main__":
    load_dataset()
