import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import torch
import sklearn
import numpy as np
import pandas as pd
import os
import hydra

from omegaconf import DictConfig
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast, BitsAndBytesConfig
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from data_loader import format_df
from peft import PeftModel


class Inference:
    def __init__(self, config):
        self.tokenizer = None
        self.dataset_config = config['dataset']
        self.model_config = config['model']
        self.inference_config = config['inference']

    def pre_process(self):
        test = pd.read_csv(os.path.join(self.dataset_config['data_dir'], 'test.csv'))

        cleaned_test = format_df(test)
        cleaned_test['input_ids'], cleaned_test['attention_mask'] = self.tokenize(cleaned_test)
        cleaned_test["length"] = cleaned_test["input_ids"].apply(len)

        cleaned_test = cleaned_test.sort_values("length", ascending=False)

        return cleaned_test

    def tokenize(self, df):
        self.tokenizer = GemmaTokenizerFast.from_pretrained(self.model_config['checkpoint'])
        self.tokenizer.add_eos_token = True
        self.tokenizer.padding_side = "right"

        df['input'] = df.apply(lambda x: self.tokenizer(x['input'],
                                                        max_length=self.model_config['max_length'],
                                                        truncation=True,
                                                        padding=True
                                                        ), axis=1)

        df['input_ids'] = df['input'].apply(lambda x: x['input_ids'])
        df['attention_mask'] = df['input'].apply(lambda x: x['attention_mask'])

        return df['input_ids'].tolist(), df['attention_mask'].tolist()

    def inference(self, df):
        model = Gemma2ForSequenceClassification.from_pretrained(
            self.model_config['checkpoint'],
            num_labels=3,
            device_map="auto",
            use_cache=False
        )

        model = PeftModel.from_pretrained(model, os.path.join(self.inference_config['model_output_dir'],
                                                              'checkpoint-20'))
        st = time.time()

        result_df = self.run(df, model)
        proba = result_df[["winner_model_a", "winner_model_b", "winner_tie"]].values

        print(f"elapsed time: {time.time() - st}")

        result_df.loc[:, "winner_model_a"] = proba[:, 0]
        result_df.loc[:, "winner_model_b"] = proba[:, 1]
        result_df.loc[:, "winner_tie"] = proba[:, 2]
        submission_df = result_df[["id", 'winner_model_a', 'winner_model_b', 'winner_tie']]
        submission_df.to_csv('submission.csv', index=False)


    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def run(self, df, model, device=torch.device("cuda"), batch_size=None, max_length=None):
        batch_size = self.inference_config['batch_size']
        max_length = self.inference_config['max_length']

        a_win, b_win, tie = [], [], []

        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            tmp = df.iloc[start_idx:end_idx]
            input_ids = tmp["input_ids"].to_list()
            attention_mask = tmp["attention_mask"].to_list()
            inputs = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                {"input_ids": input_ids, "attention_mask": attention_mask},
                padding="longest",
                pad_to_multiple_of=None,
                return_tensors="pt",
            )
            outputs = model(**inputs.to(device))
            proba = outputs.logits.softmax(-1).cpu()

            a_win.extend(proba[:, 0].tolist())
            b_win.extend(proba[:, 1].tolist())
            tie.extend(proba[:, 2].tolist())

        df["winner_model_a"] = a_win
        df["winner_model_b"] = b_win
        df["winner_tie"] = tie

        return df


@hydra.main(version_base=None, config_path="../config", config_name="model")
def main(cfg: DictConfig):
    inference = Inference(cfg)
    df_test = inference.pre_process()
    inference.inference(df_test)


if __name__ == "__main__":
    main()
