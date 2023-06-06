import pandas as pd
import tokenizers
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from evaluate import load as load_metric
import os
import math
import tempfile
from typing import Callable
from . import trial as tl
from share import param
from share import settings
from share import log
from share import model as ml


class __ModelTrainDataset(Dataset):
    """
    モデルの訓練のためのデータセットです。

    与えられたデータフレームから、特定の最大長でトークン化された入力と出力を生成します。

     引数:
         tokenizer (tokenizers.Tokenizer): 入力と出力をトークン化するためのトークナイザー。

         df (pandas.DataFrame): データを含むDataFrame。"Input"と"Output"の列が必要です。

         prefix (str): モデルに使用する、接頭辞
    """

    def __init__(
        self,
        tokenizer: tokenizers.Tokenizer,
        df: pd.DataFrame,
        prefix: str,
    ):
        self.tokenizer = tokenizer
        self.df = df
        self.prefix = prefix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_word = self.prefix + tl.preprocess_text(row["Input"])
        output_word = tl.preprocess_text(row["Output"])
        encoding = self.tokenizer.encode_plus(
            input_word,
            max_length=settings.max_length_src,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer.encode_plus(
            output_word,
            max_length=settings.max_length_target,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": targets["input_ids"].flatten(),
        }


class __LoggingCallback(TrainerCallback):
    """
    訓練中の情報をログに記録するためのコールバック。

    このコールバックは、各ステップの終了時に情報をログに記録します。
    """

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0 and state.log_history:
            log.logger.info(
                f"Step {state.global_step}, Loss: {state.log_history[-1]['loss']}"
            )


# BLEUメトリクスの読み込み
__bleu_metric = load_metric("bleu")


class __TrainHelper:
    """
    トレーニングプロセスを管理するヘルパークラスです。

    このクラスはモデルとトークナイザーを管理し、

    オプティマイザーとスケジューラーを生成します。

    また、評価予測に基づいてメトリクスを計算します。

    引数:
        model (torch.nn.Module): トレーニングするモデル。

        tokenizer (tokenizers.Tokenizer): 使用するトークナイザー。
    """

    def __init__(self, model: torch.nn.Module, tokenizer: tokenizers.Tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.tokenizer = tokenizer

    def get_optimizers(self, learning_rate, warmup_steps, training_steps):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
        )
        return optimizer, scheduler

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 簡単な後処理を施します
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        bleu_score = __bleu_metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        print(f"bleu_score: {bleu_score['score']}")

        return {"bleu_score": bleu_score["score"]}


def execute(
    df: any,
    callback: Callable,
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
) -> bytes:
    """与えられた訓練データから、訓練と評価を行います。訓練結果のモデルとトークナイザは指定されたパスに保存されます。

    引数:
        df (any): 訓練用のデータのDataFrame

        callback (Callable): 進行状況を報告するためのコールバック関数

        tokenizer (T5Tokenizer): T5のトークナイザ

        model (T5ForConditionalGeneration): T5のモデル
    """
    # 訓練評価用データ
    train_df, eval_df = train_test_split(df, test_size=0.2)

    # 訓練パラメータを取得
    params = param.get_train_param()

    # データセットの作成
    prefix = param.get_prefix_param()
    train_dataset = __ModelTrainDataset(tokenizer, train_df, prefix)
    eval_dataset = __ModelTrainDataset(tokenizer, eval_df, prefix)

    # 最大ステップ数を計算(最低１回)
    max_steps = max(
        1,
        math.ceil(len(train_dataset) / params["per_device_train_batch_size"])
        * params["num_train_epochs"],
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 訓練の設定
        training_args = Seq2SeqTrainingArguments(
            logging_steps=1,
            predict_with_generate=True,
            max_steps=max_steps,
            save_strategy="epoch",
            output_dir=os.path.join(tmp_dir, "result"),
            num_train_epochs=params["num_train_epochs"],
            per_device_train_batch_size=params["per_device_train_batch_size"],
            per_device_eval_batch_size=params["per_device_eval_batch_size"],
            weight_decay=params["weight_decay"],
            gradient_accumulation_steps=params["gradient_accumulation_steps"],
            logging_dir=os.path.join(tmp_dir, "log"),
        )
        # ログの取得
        log_call = __LoggingCallback()

        train_helper = __TrainHelper(model, tokenizer)

        optimizers = train_helper.get_optimizers(
            params["learning_rate"],
            params["warmup_steps"],
            max_steps,
        )

        # トレーナーの作成
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[callback, log_call],
            optimizers=optimizers,
            compute_metrics=train_helper.compute_metrics,
        )

        # 訓練
        log.logger.info("訓練を開始します...")
        train_result = trainer.train()

    # 訓練完了時のメトリクスの整形
    result_message = ", ".join(
        f"{key}: {value}" for key, value in train_result.metrics.items()
    )

    # 訓練完了時のメトリクスを表示
    log.logger.info(result_message)

    log.logger.info("学習が終了しました!")

    # 訓練が終了したモデルとトークナイザを保存
    return ml.save(model, tokenizer)
