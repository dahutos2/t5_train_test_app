import streamlit as st
import re
import pandas as pd
from logic import trial
from share import param
from share import log
from share import model as ml


def show() -> None:
    """このメソッドは、指定された入力文字列またはCSVファイルに含まれる複数の入力文字列を元に出力を表示します。"""
    st.subheader("試験")
    option = st.selectbox("試験内容を選択して下さい。", ("T5", "Word2Vec"))
    if option == "T5":
        __trial_t5()
    elif option == "Word2Vec":
        ml.set_is_fast_text()
        __trial_word2_vec()


def __trial_t5():
    model_param = param.get_model_param()
    is_succeeded, model_path = ml.get_model_path(model_param)

    ml.set_prefix()

    option = st.selectbox("入力方法を選択して下さい", ("単体", "複数"))

    if option == "単体":
        word = st.text_area("任意の文字列を入力してください。")
        if is_succeeded and st.button("実行"):
            param.set_model_param(model_path)

            # 非活性化
            log.change_enabled(False)

            # トークナイザとモデルの準備
            tokenizer, model = ml.get_load_model(model_path)

            with st.spinner("モデルを使用して入力文字列から出力を作成中..."):
                generated = trial.execute_t5(
                    word,
                    tokenizer,
                    model,
                )
                # 活性化
                log.change_enabled(True)

            st.write(f"入力: {word} / 出力: {generated}")
    elif option == "複数":
        # CSVファイルのアップロード
        upload_file = st.file_uploader(
            "CSVファイルをアップロードしてください。", type=["csv"], key="upload"
        )
        if is_succeeded and upload_file is not None and st.button("複数実行"):
            param.set_model_param(model_path)

            # 非活性化
            log.change_enabled(False)

            # トークナイザとモデルの準備
            tokenizer, model = ml.get_load_model(model_path)
            with st.spinner("モデルを使用して入力文字列から出力を作成中..."):
                # プログレスバーの初期化
                progress_bar = st.progress(0)

                words_df = pd.read_csv(upload_file)
                words = words_df["Input"].tolist()

                # 入力と出力のペアのリストを作成
                word_pairs = []
                for i, word in enumerate(words):
                    word_pairs.append(
                        (
                            word,
                            trial.execute_t5(
                                word,
                                tokenizer,
                                model,
                            ),
                        )
                    )
                    progress_bar.progress((i + 1) / len(words))

                # DataFrameに変換
                df = pd.DataFrame(word_pairs, columns=["Input", "Output"])
                # 活性化
                log.change_enabled(True)

            # プログレスバーを消す
            progress_bar.empty()

            # 結果をデータフレームで表示する
            st.dataframe(df)

            # DataFrameをCSV形式の文字列に変換します。
            csv = df.to_csv(index=False)
            binary = bytes(csv, "utf-8")

            st.download_button(
                label="クリックして、出力データ(csv)をダウンロードして下さい。",
                data=binary,
                file_name="data.csv",
                mime="text/csv",
            )


def __trial_word2_vec():
    option = st.selectbox("タスクを選択して下さい。", ("単語の類似度の比較", "類似した単語の生成", "単語の足し引き"))
    error_msg = "単語の形式が不正です。"
    if option == "単語の類似度の比較":
        st.write("比較する二つの単語を入力して下さい。")
        inputs = []

        # 列の作成
        while True:
            col1, col2 = st.columns(2)
            with col1:
                word1 = st.text_input("単語1", key=f"word1-{len(inputs)}")
            with col2:
                word2 = st.text_input("単語2", key=f"word2-{len(inputs)}")

            words = [word1, word2]
            if all(__is_word(word) for word in words):
                inputs.append(tuple(words))
                continue
            elif not all(__is_word(word) or word == "" for word in words):
                st.warning(error_msg)

            break

        if st.button("実行", disabled=len(inputs) == 0):
            # 非活性化
            log.change_enabled(False)

            # 単語の類似度算出用モデルの準備
            keyed_vectors = ml.get_keyed_vectors()

            with st.spinner("単語の類似度の比較中..."):
                # プログレスバーの初期化
                progress_bar = st.progress(0)

                similarity_list = []
                succeeded_list = []
                for i, word_tuple in enumerate(inputs):
                    result = trial.execute_word2_vec_compare(word_tuple, keyed_vectors)
                    progress_bar.progress((i + 1) / len(inputs))
                    if result != None:
                        similarity_list.append(result)
                        succeeded_list.append(word_tuple)

                # プログレスバーを消す
                progress_bar.empty()

            # 活性化
            log.change_enabled(True)

            if len(similarity_list) == 0:
                st.write("データが空です。")
            else:
                if len(similarity_list) > 1:
                    st.line_chart(similarity_list)
                    st.write(f"平均類似度: {sum(similarity_list) / len(similarity_list)}")
                st.write("単語の類似度一覧")
                for i in range(len(similarity_list)):
                    st.write(
                        f"「{succeeded_list[i][0]}」と「{succeeded_list[i][1]}」の類似度: {similarity_list[i]}"
                    )

    elif option == "類似した単語の生成":
        topn = st.number_input(
            "生成する単語の数を指定して下さい。",
            min_value=1,
            max_value=100,
            step=1,
            value=10,
        )
        input_select = st.selectbox("入力方法を選択して下さい。", ("単体", "複数"))
        if input_select == "単体":
            input_text = st.text_input("単語を入力して下さい。")
            if input_text != "" and not __is_word(input_text):
                st.warning(error_msg)
            if st.button("実行", disabled=not __is_word(input_text)):
                # 非活性化
                log.change_enabled(False)

                # 単語の類似度算出用モデルの準備
                keyed_vectors = ml.get_keyed_vectors()

                with st.spinner("類似した単語を生成中..."):
                    result = trial.execute_word2_vec_generate(
                        [input_text], [], keyed_vectors, topn
                    )

                # 活性化
                log.change_enabled(True)
                if len(result) == 0:
                    st.warning(f"「{input_text}」がモデルに存在しません。")
                else:
                    st.write(f"「{input_text}」に類似した単語の一覧")
                    for word, similarity in result:
                        st.write(f"「{word}」の類似度: {similarity}")

        elif input_select == "複数":
            # CSVファイルのアップロード
            upload_file = st.file_uploader(
                "CSVファイルをアップロードしてください。", type=["csv"], key="upload"
            )
            if upload_file is not None and st.button("複数実行"):
                # 非活性化
                log.change_enabled(False)

                # 単語の類似度算出用モデルの準備
                keyed_vectors = ml.get_keyed_vectors()

                with st.spinner("類似した単語を生成中..."):
                    # プログレスバーの初期化
                    progress_bar = st.progress(0)

                    words_df = pd.read_csv(upload_file)
                    words = words_df["Input"].tolist()
                    results = []
                    for i, word in enumerate(words):
                        result = trial.execute_word2_vec_generate(
                            [word], [], keyed_vectors, topn
                        )
                        progress_bar.progress((i + 1) / len(words))
                        if len(result) > 0:
                            new_result = [(word,) + tup for tup in result]
                            results += new_result

                # DataFrameに変換
                df = pd.DataFrame(results, columns=["Input", "Output", "Similarity"])

                # 活性化
                log.change_enabled(True)

                # プログレスバーを消す
                progress_bar.empty()

                if len(results) == 0:
                    st.write("データが空です。")
                else:
                    # 結果をデータフレームで表示する
                    st.dataframe(df)

                    # DataFrameをCSV形式の文字列に変換します。
                    csv = df.to_csv(index=False)
                    binary = bytes(csv, "utf-8")

                    st.download_button(
                        label="クリックして、出力データ(csv)をダウンロードして下さい。",
                        data=binary,
                        file_name="data.csv",
                        mime="text/csv",
                    )

    elif option == "単語の足し引き":
        topn = st.number_input(
            "生成する単語の数を指定して下さい。",
            min_value=1,
            max_value=100,
            step=1,
            value=10,
        )

        positive_inputs = []
        while True:
            positive = st.text_input("足す単語", key=f"positive-{len(positive_inputs)}")
            if __is_word(positive):
                positive_inputs.append(positive)
                continue
            elif positive != "" and not __is_word(positive):
                st.warning(error_msg)

            break

        negative_inputs = []
        while True:
            negative = st.text_input("引く単語", key=f"negative-{len(negative_inputs)}")
            if __is_word(negative):
                negative_inputs.append(negative)
                continue
            elif negative != "" and not __is_word(negative):
                st.warning(error_msg)

            break

        if st.button(
            "実行", disabled=len(positive_inputs) == 0 and len(negative_inputs) == 0
        ):
            # 非活性化
            log.change_enabled(False)
            # 単語の類似度算出用モデルの準備
            keyed_vectors = ml.get_keyed_vectors()

            with st.spinner("足し引きした単語を生成中..."):
                result = trial.execute_word2_vec_generate(
                    positive_inputs, negative_inputs, keyed_vectors, topn
                )

            # 活性化
            log.change_enabled(True)
            if len(result) == 0:
                st.warning("足し引きした単語の生成に失敗しました。")
            else:
                st.write("足し引きした単語の一覧")
                for word, similarity in result:
                    st.write(f"「{word}」の類似度: {similarity}")


def __is_word(input: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-zぁ-んァ-ン一-龥ー]+", input))
