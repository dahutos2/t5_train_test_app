import streamlit as st
import pandas as pd
from logic import test
from share import param
from share import log
from share import model as ml

__similarity_list = []


def show() -> None:
    """検証を行うための関数です。

    モデルの読み込み、CSVファイルのアップロード、検証の実行、結果の表示などを行います。"""
    st.subheader("検証")

    ml.set_is_fast_text()

    model_param = param.get_model_param()
    is_succeeded, model_path = ml.get_model_path(model_param)

    ml.set_prefix()

    # CSVファイルのアップロード
    upload_file = st.file_uploader("CSVファイルをアップロードしてください。", type=["csv"], key="test")
    if is_succeeded and upload_file is not None and st.button("検証"):
        param.set_model_param(model_path)

        # 非活性化
        log.change_enabled(False)

        # 単語の類似度算出用モデルの準備
        keyed_vectors = ml.get_keyed_vectors()

        # トークナイザとモデルの準備
        tokenizer, model = ml.get_load_model(model_path)
        with st.spinner("検証中..."):
            # プログレスバーの初期化
            progress_bar = st.progress(0)

            df = pd.read_csv(upload_file)
            global __similarity_list
            __similarity_list = test.execute(
                df,
                progress_bar.progress,
                keyed_vectors,
                tokenizer,
                model,
            )
            # プログレスバーを完了状態にする
            progress_bar.progress(1)

            # 活性化
            log.change_enabled(True)

        st.success("検証が終了しました！")
        # プログレスバーを消す
        progress_bar.empty()

    if len(__similarity_list) == 0:
        st.write("データが空です。")
    else:
        st.line_chart(__similarity_list)
        st.write(f"平均類似度: {sum(__similarity_list) / len(__similarity_list)}")
