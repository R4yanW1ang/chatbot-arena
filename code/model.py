import os
import torch
import hydra
import numpy as np

from omegaconf import DictConfig
from datasets import Dataset
from transformers import (
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score

from tokenizer import CustomTokenizer


class ChatbotTrainer:
    def __init__(self, config):
        self.tokenizer = None
        self.dataset_config = config['dataset']
        self.model_config = config['model']
        self.training_config = config['training']
        self.lora_config = config['lora']

    def load_dataset(self):
        df_train = Dataset.from_csv(os.path.join(self.dataset_config['data_dir'], 'formatted_train.csv'))

        # Initiate tokenizer
        self.tokenizer = GemmaTokenizerFast.from_pretrained(self.model_config['checkpoint'])
        self.tokenizer.add_eos_token = True
        self.tokenizer.padding_state = 'right'

        encode = CustomTokenizer(max_length=self.model_config['max_length'], tokenizer=self.tokenizer)
        df_train = df_train.map(encode, batched=True)

        return df_train

    def dataset_split(self, df_train):
        folds = [
            (
                [i for i in range(len(df_train)) if i % self.dataset_config['n_splits'] != fold_idx],
                [i for i in range(len(df_train)) if i % self.dataset_config['n_splits'] == fold_idx]
            )
            for fold_idx in range(self.dataset_config['n_splits'])
        ]

        train_idx, eval_idx = folds[self.dataset_config['fold_idx']]

        train_dataset = df_train.select(train_idx)
        val_dataset = df_train.select(eval_idx)

        return train_dataset, val_dataset

    def training(self, train_dataset, val_dataset):
        training_args = TrainingArguments(
            output_dir="model_output",
            overwrite_output_dir=True,
            report_to="none",
            num_train_epochs=self.training_config['n_epochs'],
            per_device_train_batch_size=self.training_config['per_device_train_batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            per_device_eval_batch_size=self.training_config['per_device_eval_batch_size'],
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="steps",
            save_steps=200,
            optim=self.training_config['optim_type'],
            fp16=False,
            learning_rate=self.training_config['lr'],
            warmup_steps=self.training_config['warmup_steps'],
            max_grad_norm=1.0, ###
        )

        lora_config = LoraConfig(
            r=self.lora_config['lora_r'],
            lora_alpha=self.lora_config['lora_alpha'],
            # only target self-attention
            target_modules=["q_proj", "k_proj", "v_proj"],
            layers_to_transform=[i for i in range(42) if i >= self.training_config['freeze_layers']],
            lora_dropout=self.lora_config['lora_dropout'],
            bias=self.lora_config['lora_bias'],
            task_type=TaskType.SEQ_CLS,
        )

        model = Gemma2ForSequenceClassification.from_pretrained(
            self.model_config['checkpoint'],
            num_labels=3,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        trainer = Trainer(
            args=training_args,
            model=model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )
        trainer.train()


# def compute_metrics(eval_preds: EvalPrediction) -> dict:
#     preds = eval_preds.predictions
#     labels = eval_preds.label_ids
#     probs = torch.from_numpy(preds).float().softmax(-1).numpy()
#     loss = log_loss(y_true=labels, y_pred=probs)
#     acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
#     return {"acc": acc, "log_loss": loss}


def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids

    # Convert logits to probabilities
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()

    # Check for NaNs in logits and probabilities
    if np.any(np.isnan(preds)):
        print("NaNs found in logits.")

    if np.any(np.isnan(probs)):
        print("NaNs found in softmax probabilities.")

    if np.any(np.isnan(labels)):
        print("NaNs found in labels.")

    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))

    return {"acc": acc, "log_loss": loss}


@hydra.main(version_base=None, config_path="../config", config_name="model")
def main(cfg: DictConfig):
    print('Starting process...', flush=True)
    trainer = ChatbotTrainer(cfg)
    df_train = trainer.load_dataset()
    print('Dataset loaded', flush=True)
    train_dataset, val_dataset = trainer.dataset_split(df_train)
    print('Dataset split', flush=True)
    trainer.training(train_dataset, val_dataset)
    print('success', flush=True)


if __name__ == "__main__":
    main()