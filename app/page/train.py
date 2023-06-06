import streamlit as st
import pandas as pd
from transformers import (
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from typing import Any, Dict
from share import param
from share import settings
from logic import train
from share import log
from share import model as ml


class __StreamlitProgressCallback(TrainerCallback):
    """
    Streamlit の進捗バーを更新するためのコールバックです。

    このクラスはトレーニングの進行に応じて Streamlit の進捗バーを更新します。

    具体的には、各エポックの開始時に進捗バーを初期化し、

    各ステップの終了時に進捗バーを更新します。
    """

    def __init__(self):
        self.progress_bar = None

    def on_epoch_begin(
        self, args: Seq2SeqTrainingArguments, state: Dict[str, Any], control, **kwargs
    ):
        if state.is_local_process_zero and state.global_step == 0:
            self.progress_bar = st.progress(0)
            self.total_steps = args.max_steps

    def on_step_end(
        self, args: Seq2SeqTrainingArguments, state: Dict[str, Any], control, **kwargs
    ):
        if state.is_local_process_zero:
            progress = state.global_step / self.total_steps
            self.progress_bar.progress(progress)


def show() -> None:
    """この関数は、ユーザーからの入力を受け取り、それに基づいてモデルの訓練を行います。

    具体的には、以下の情報をユーザーから受け取ります：

        接頭辞: モデルに使用する、接頭辞。

        訓練エポック数: 全ての訓練データがネットワークを通過する回数。

        デバイスごとの訓練バッチサイズ: 各デバイス（GPUまたはCPU）での訓練バッチサイズ。

        デバイスごとの評価バッチサイズ: 各デバイス（GPUまたはCPU）での評価バッチサイズ。

        ウォームアップステップ数: 学習率スケジューラーのウォームアップステップ数。

        重み減衰率: モデルの重み減衰率。

        学習率: モデルのパラメータ更新のスピードを制御する学習率。

        勾配蓄積ステップ数: バッチサイズを事実上増加させるために使用される勾配蓄積ステップ数。

    これらの情報を受け取った後、ユーザーからCSVファイルがアップロードされると、受け取ったパラメータを使用してモデルの訓練が開始されます。"""
    st.subheader("学習")

    # ユーザーからのハイパーパラメータの入力を受け取る
    default_param = param.get_train_param()

    is_succeeded, model_path = ml.get_model_path(default_param["model_path"])

    is_checked = st.checkbox("訓練を行わない")

    if is_checked:
        __execute_save(is_succeeded, model_path)
        return

    ml.set_prefix()

    params = __set_hyper_parameters(default_param, model_path)

    __execute_train(is_succeeded, params, model_path)


def __set_hyper_parameters(
    default_param: Dict[str, Any], model_path: str
) -> Dict[str, Any]:
    num_train_epochs = st.number_input(
        "訓練エポック数を入力してください。",
        min_value=1,
        max_value=100,
        value=default_param["num_train_epochs"],
        step=1,
        help="""
        \nエポックとは、全ての訓練データがネットワークを通過する回数を指します。
        \nこの値はモデルが過学習しないようにするために調整する必要があります。
        \n一般的には、数エポック（2〜10など）から始め、
        \n過学習の兆候を見つけるために検証セットのパフォーマンスを監視します。
        """,
    )
    per_device_train_batch_size = st.number_input(
        "デバイスごとの訓練バッチサイズを入力してください。",
        min_value=1,
        max_value=100,
        value=default_param["per_device_train_batch_size"],
        step=1,
        help="""
        \n各デバイス（GPUまたはCPU）での訓練バッチサイズです。
        \nバッチサイズとは、同時にネットワークを通過するトレーニングデータの数を指します。
        \nこの値は、利用可能なメモリに応じて適切に調整する必要があります。
        \n一般的には、8から32の範囲が適切な値とされています。
        """,
    )
    per_device_eval_batch_size = st.number_input(
        "デバイスごとの評価バッチサイズを入力してください。",
        min_value=1,
        max_value=100,
        value=default_param["per_device_eval_batch_size"],
        step=1,
        help="""
        \n各デバイス（GPUまたはCPU）での評価バッチサイズです。
        \n訓練バッチサイズ同様、利用可能なメモリに応じて調整する必要があります。
        \n一般的には、訓練バッチサイズと同等またはそれ以上の値が選択されます。
        """,
    )
    warmup_steps = st.number_input(
        "学習率スケジューラーのウォームアップステップ数を入力してください。",
        min_value=0,
        max_value=10000,
        value=default_param["warmup_steps"],
        step=1,
        help="""
        \nウォームアップステップの数は、学習率スケジューリングに影響します。
        \n訓練の初期段階で学習率を徐々に上げることで、訓練の安定性を向上させることができます。
        \n適切な値は訓練ステップの総数の一部で、
        \n一般的にはトータルステップの10%程度が指定されます。
        """,
    )
    weight_decay = st.number_input(
        "重み減衰率を入力してください。",
        min_value=0.0,
        max_value=1.0,
        value=default_param["weight_decay"],
        step=0.01,
        help="""
        \n重み減衰は正則化手法の一つで、モデルの複雑さを制限することで過学習を防ぎます。
        \n一般的には0.0から0.01の範囲が推奨されます。
        """,
    )
    learning_rate = st.number_input(
        "学習率を入力してください。",
        min_value=1e-6,
        max_value=1e-1,
        value=default_param["learning_rate"],
        step=1e-6,
        format="%.6f",
        help="""
        \n学習率はモデルのパラメータ更新のスピードを制御します。
        \n大きすぎると学習が不安定になり、小さすぎると学習が遅くなります。
        \n一般的には0.00001（1e-5）から0.1の範囲で設定します。
        """,
    )
    gradient_accumulation_steps = st.number_input(
        "勾配蓄積ステップ数を入力してください。",
        min_value=1,
        max_value=100,
        value=default_param["gradient_accumulation_steps"],
        step=1,
        help="""
        \n勾配累積ステップ数は、バッチサイズを事実上増加させるために使用されます。
        \nこれにより、ハードウェアのメモリ制限に直面している場合でも、大きなバッチで訓練を行うことができます。
        \n適切な値はハードウェアのメモリとバッチサイズに依存します。
        \n例えば、バッチサイズが8で、メモリがそれ以上のバッチサイズを扱うことができない場合、
        \n勾配累積ステップ数を2に設定することで、実質的なバッチサイズを16にすることができます。
        \nただし、勾配累積は計算時間を増加させる可能性があります。
        """,
    )

    params = {
        "model_path": model_path,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    return params


def __execute_train(
    is_succeeded: bool,
    params: Dict[str, Any],
    model_path: str,
) -> None:
    # CSVファイルのアップロード
    upload_file = st.file_uploader("CSVファイルをアップロードしてください。", type=["csv"], key="train")
    if (
        is_succeeded
        and upload_file is not None
        and st.button("訓練", disabled=settings.is_deployed)
    ):
        param.set_train_param(params)
        callback = __StreamlitProgressCallback()

        # 非活性化
        log.change_enabled(False)

        # トークナイザとモデルの準備
        tokenizer, model = ml.get_load_model(model_path)
        with st.spinner("訓練中..."):
            df = pd.read_csv(upload_file)
            binary = train.execute(df, callback, tokenizer, model)
            # 活性化
            log.change_enabled(True)

        st.success("訓練が終了しました！")
        # プログレスバーを消す
        callback.progress_bar.empty()

        # ダウンロードボタンを表示
        st.download_button(
            "クリックして、モデルをダウンロードして下さい。",
            data=binary,
            file_name="model.zip",
            mime="application/zip",
        )


def __execute_save(
    is_succeeded: bool,
    model_path: str,
) -> None:
    if is_succeeded and st.button("保存"):
        # 非活性化
        log.change_enabled(False)

        # トークナイザとモデルの準備
        tokenizer, model = ml.get_load_model(model_path)
        with st.spinner("保存中..."):
            binary = ml.save(tokenizer, model)
            # 活性化
            log.change_enabled(True)

        st.success("保存が終了しました！")

        # ダウンロードボタンを表示
        st.download_button(
            "クリックして、モデルをダウンロードして下さい。",
            data=binary,
            file_name="model.zip",
            mime="application/zip",
        )
