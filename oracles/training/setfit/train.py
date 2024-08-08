# RUN: CUDA_VISIBLE_DEVICES=3 python -m oracles.training.setfit.train

import glob
import pandas as pd
import os
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import load_dataset

import traceback
import hydra
import logging
import torch
import glob
from transformers import EarlyStoppingCallback
log = logging.getLogger(__name__)

from oracles.training.setfit.utils import (
    prepare_dataset,
    compute_metrics
)

class SentenceClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_exists = os.path.exists(self.cfg.trainer.output_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load datasets
        self.train_dataset = prepare_dataset(load_dataset('csv', data_files=self.cfg.dataset.train_data_path)['train'].shuffle())
        self.val_dataset = prepare_dataset(load_dataset('csv', data_files=self.cfg.dataset.val_data_path)['train'].shuffle())
        if self.cfg.dataset.max_val_size > 0:
            self.val_dataset = self.val_dataset.shuffle()
            self.val_dataset = self.val_dataset.select(range(self.cfg.dataset.max_val_size))
        self.test_dataset = prepare_dataset(load_dataset('csv', data_files=self.cfg.dataset.test_data_path)['train'].shuffle())

        log.info(f"train_dataset: {self.train_dataset}")
        log.info(f"val_dataset: {self.val_dataset}")
        log.info(f"test_dataset: {self.test_dataset}")

        # log.info(f"train_dataset: {self.train_dataset[0]}")
        # log.info(f"val_dataset: {self.val_dataset[0]}")
        # log.info(f"test_dataset: {self.test_dataset[0]}")
        
        # Configure callbacks
        self.callbacks = []
        escb = EarlyStoppingCallback(
            early_stopping_patience=self.cfg.trainer.early_stopping_patience
        )
        self.callbacks.append(escb)


    def load_trainer(self):
        # checkpoint = get_highest_numeric_subfolder(self.cfg.output_dir)
        # model_name = checkpoint if self.model_exists else self.cfg.model_name_or_path
        model_name = self.cfg.trainer.output_dir if self.model_exists else self.cfg.model.name

        self.model = SetFitModel.from_pretrained(
            model_name,
            cache_dir=self.cfg.model.cache_dir, 
            device=self.device,
        )

        training_args = TrainingArguments(
            output_dir=self.cfg.trainer.output_dir,
            batch_size=(self.cfg.trainer.per_device_train_batch_size, self.cfg.trainer.per_device_eval_batch_size),
            max_steps=self.cfg.trainer.max_steps,
            logging_steps=self.cfg.trainer.logging_steps,
            evaluation_strategy=self.cfg.trainer.evaluation_strategy,
            save_strategy=self.cfg.trainer.save_strategy,
            save_total_limit=self.cfg.trainer.save_total_limit,
            save_steps=self.cfg.trainer.save_steps,
            load_best_model_at_end=self.cfg.trainer.load_best_model_at_end,
            seed=self.cfg.trainer.seed,
            use_amp=self.cfg.trainer.use_amp,
        )
        self.trainer = Trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            args=training_args,
            callbacks=self.callbacks,
            metric=compute_metrics,
        )

    def get_predictions(self):
        # Get predictions for the test dataset
        texts = [item["text"] for item in self.test_dataset]
        pred_labels = self.model.predict(texts)
        return pred_labels

    def evaluate(self):
        self.load_trainer()
        if not self.model_exists:
            self.trainer.train()
            self.trainer.model.save_pretrained(save_directory=self.cfg.trainer.output_dir)
        metrics = self.trainer.evaluate(self.test_dataset)
        log.info("Evaluation metrics:", metrics)

        # # # Get predictions and save to CSV
        # pred_labels = self.get_predictions()
        # df = pd.DataFrame({
        #     "text": [item["text"] for item in self.test_dataset],
        #     "label": pred_labels[:,0],
        # }) 
        # df.to_csv(f"./oracles/training/setfit/results/{self.cfg.model.name.replace('/','--')}_annotations.csv", index=False)

        return metrics

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):

    models = [
        "allenai/longformer-base-4096",
        "google/bigbird-roberta-base",
        "transfo-xl/transfo-xl-wt103",
        "google-bert/bert-base-uncased",
        "FacebookAI/roberta-base",
        "microsoft/deberta-v3-base",
        # "microsoft/deberta-v3-large",
        "FacebookAI/xlm-roberta-large",
    ]

    performance = []
    for m in models:
        cfg.model.name = m

        try: 
            classifier = SentenceClassifier(cfg)
            metrics = classifier.evaluate()
            metrics.update({
                "model": m,
            })
            performance.append(metrics)
            print(f"Metrics: {metrics}")
            df = pd.DataFrame(performance)
            df.to_csv("./oracles/training/setfit/metrics.csv", index=False)
            
        except Exception:
            print(traceback.format_exc())


if __name__ == "__main__":

    main()