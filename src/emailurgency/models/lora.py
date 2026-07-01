from __future__ import annotations

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from configs.train_config import LoraConfig, get_lora_config


class LoRA:
    """LoRA sequence classification model."""

    def __init__(self, config: LoraConfig | None = None) -> None:
        """Initialize LoRA."""
        self.config = config or get_lora_config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model,
            num_labels=self.config.num_labels,
        )

    def fit(self, X_train, y_train, X_test=None, y_test=None) -> LoRA:
        """Fine-tune LoRA model."""
        peft_config = PeftLoraConfig(**self.config.peft_params)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        train_dataset = self._dataset(X_train, y_train)
        eval_dataset = None
        training_args = self.config.training_args
        if X_test is not None and y_test is not None:
            eval_dataset = self._dataset(X_test, y_test)
        else:
            training_args = {**training_args, 'eval_strategy': 'no'}

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(**training_args),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
        )
        trainer.train()
        self.save()
        return self

    def transform(self, corpus: list[str]) -> np.ndarray:
        """Return one pooled embedding per text from the current model."""
        if self.model is None:
            raise ValueError('Model is not initialized.')

        self.model.eval()
        encodings = self.tokenizer(
            list(corpus),
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors='pt',
        )
        encodings = {key: value.to(self.model.device) for key, value in encodings.items()}

        with torch.no_grad():
            outputs = self.model(
                **encodings,
                output_hidden_states=True,
                return_dict=True,
            )

        token_embeddings = outputs.hidden_states[-1]
        attention_mask = encodings['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * attention_mask
        lengths = attention_mask.sum(dim=1).clamp(min=1e-9)
        return (masked_embeddings.sum(dim=1) / lengths).cpu().numpy()

    def embeddings(self, corpus: list[str]) -> np.ndarray:
        """Return embeddings using the saved merged LoRA model."""
        if not self.config.merged_path.exists():
            raise FileNotFoundError(f'Merged LoRA model not found: {self.config.merged_path}')

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.merged_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.merged_path)
        return self.transform(corpus)

    def save(self) -> None:
        """Save adapter, merged model, and tokenizer."""
        if self.model is None:
            raise ValueError('Cannot save before fitting the model.')

        self.config.lora_path.mkdir(parents=True, exist_ok=True)
        self.config.merged_path.mkdir(parents=True, exist_ok=True)
        self.config.model_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(self.config.lora_path)
        self.tokenizer.save_pretrained(self.config.model_path)

        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(self.config.merged_path)
        self.tokenizer.save_pretrained(self.config.merged_path)
        self.model = merged_model

    def _dataset(self, X, y) -> Dataset:
        """Tokenize text and build a Hugging Face dataset."""
        encodings = self.tokenizer(
            [str(text) for text in X],
            truncation=True,
            max_length=self.config.max_length,
        )
        return Dataset.from_dict({**encodings, 'labels': np.asarray(y, dtype=int).tolist()})
