import gensim
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from typing import List, Callable
from . import trial as tl
from share import log
from share import param


def execute(
    eval_df: any,
    callback: Callable,
    keyed_vectors: gensim.models.KeyedVectors,
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
) -> List[float]:
    """与えられたモデルを使用して、DataFrameの入力から出力を作成し、正しい出力訳と生成された出力の類似度を計算します。類似度はWord2Vecモデルによって計算されます。

    引数:
        eval_df (any): 評価データのDataFrame

        callback (Callable): 進行状況を報告するためのコールバック関数

        keyed_vectors (gensim.models.KeyedVectors): 単語の類似度を計算するためのキー（単語など）とベクトルとのマッピング

        tokenizer (T5Tokenizer): T5のトークナイザ

        model (T5ForConditionalGeneration): T5のモデル

    戻り値:
        List[float]: 正しい出力と生成された出力の類似度のリスト
    """

    # 評価モード
    model.eval()
    similarities = []
    prefix = param.get_prefix_param()
    for index, row in eval_df.iterrows():
        input_word = row["Input"]
        correct_output = row["Output"]
        output = tl.generate(model, tokenizer, input_word, prefix)
        is_not_found = False
        try:
            similarity = keyed_vectors.similarity(correct_output, output)
        except KeyError:
            is_not_found = True
            log.logger.info(f"単語が見つからないです: {correct_output} または {output} がモデルに存在しません。")
        # 進行状況を計算し、コールバック関数を呼び出す
        progress = (index + 1) / len(eval_df)
        callback(progress)

        # Word2Vecモデルの学習データに含まれていなかった場合は追加しない
        if not is_not_found:
            similarities.append(similarity)

    return similarities
