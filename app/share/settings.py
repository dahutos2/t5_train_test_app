import os
import torch
import copy

__MODEL_SELECTIONS = (
    "任意のモデル",
    "t5-small",
    "t5-base",
    "t5-large",
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
    "retrieva-jp/t5-small-short",
    "retrieva-jp/t5-small-medium",
    "retrieva-jp/t5-small-long",
    "retrieva-jp/t5-base-short",
    "retrieva-jp/t5-base-medium",
    "retrieva-jp/t5-base-long",
    "retrieva-jp/t5-large-short",
    "retrieva-jp/t5-large-medium",
    "retrieva-jp/t5-large-long",
    "retrieva-jp/t5-xl",
    "sonoisa/t5-base-japanese-mC4-Wikipedia",
    "sonoisa/t5-base-english-japanese",
    "sonoisa/sentence-t5-base-ja-mean-tokens",
    "sonoisa/t5-base-japanese-adapt",
    "sonoisa/t5-base-japanese-question-generation",
    "sonoisa/t5-base-japanese-title-generation",
    "sonoisa/t5-qiita-title-generation",
    "sonoisa/t5-base-japanese-article-generation",
)
model_selections = copy.deepcopy(__MODEL_SELECTIONS)
is_deployed = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_length_src = 512
max_length_target = 128


def __get_bool(bool_str: str):
    if bool_str == "True":
        return True
    else:
        return False


def set_setting():
    """環境で変化する設定を定義する"""
    # デプロイ環境かどうかをチェックする
    global is_deployed
    global model_selections
    is_deployed = __get_bool(os.getenv("IS_DEPLOYED"))
    if is_deployed:
        model_selections = (
            "任意のモデル",
            "t5-small",
            "google/flan-t5-small",
            "retrieva-jp/t5-small-short",
            "retrieva-jp/t5-small-medium",
            "retrieva-jp/t5-small-long",
        )
    else:
        model_selections = copy.deepcopy(__MODEL_SELECTIONS)
