import re
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from typing import Tuple, Optional, List
import neologdn
import gensim
import torch
import transformers
from share import settings
from share import param
from share import log


def execute_t5(
    input_word: str,
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
) -> str:
    """与えられた入力から出力を生成します。出力の生成には指定されたT5のモデルが使用されます。

    引数:
        input_word (str): 出力に使用する入力文字列

        tokenizer (T5Tokenizer): T5のモデル

        model (T5ForConditionalGeneration): T5のトークナイザ

    戻り値:
        str:出力されたの文字列
    """
    # 評価モード
    model.eval()

    prefix = param.get_prefix_param()

    return generate(model, tokenizer, input_word, prefix)


def execute_word2_vec_compare(
    word_set: Tuple[str],
    keyed_vectors: gensim.models.KeyedVectors,
) -> Optional[float]:
    """複数の単語を引数に、単語の類似度を返す

    引数:
        word_set (Tuple[str]): 比較する単語のセット

        keyed_vectors (gensim.models.KeyedVectors): 単語の類似度を計算するためのキー（単語など）とベクトルとのマッピング

    戻り値:
        float: 類似度
    """
    word1, word2 = word_set
    try:
        similarity = keyed_vectors.similarity(word1, word2)
        return similarity
    except KeyError:
        log.logger.info(f"単語が見つからないです:「{word1}」または「{word2}」がモデルに存在しません。")
        return None


def execute_word2_vec_generate(
    positive_list: List[str],
    negative_list: List[str],
    keyed_vectors: gensim.models.KeyedVectors,
    topn: int,
) -> List[Tuple[str, float]]:
    """複数の単語を引数に、単語の類似度を返す

    引数:
        positive_list (List[str]): 加える単語のリスト

        negative_list (List[str]): 引く単語のリスト

        keyed_vectors (gensim.models.KeyedVectors): 単語の類似度を計算するためのキー（単語など）とベクトルとのマッピング

        topn (int): 生成する単語の数

    戻り値:
        List[Tuple[str, float]]: 単語と類似度のリスト
    """
    result = []
    try:
        result = keyed_vectors.most_similar(
            positive=positive_list,
            negative=negative_list,
            topn=topn,
        )
    except KeyError:
        error_msg = "が" if len(positive_list) + len(negative_list) == 1 else "の中のどれかが"
        log.logger.info(
            f"単語が見つからないです:「{', '.join(positive_list + negative_list)}」{error_msg}モデルに存在しません。"
        )

    return result


def generate(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    input: str,
    prefix: str,
) -> str:
    """与えられた入力から出力を生成します。出力の生成には指定されたモデルが使用されます。

    引数:
        model (torch.nn.Module): 出力に使用するモデル

        tokenizer (transformers.PreTrainedTokenizer): テキストをトークン化するためのトークナイザ

        input (str): 出力に使用する入力文字列

        prefix (str): モデルに使用する、接頭辞

    戻り値:
        str: 出力された文字列
    """
    batch = tokenizer(
        f"{prefix}{input}",
        max_length=settings.max_length_src,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # 生成処理を行う
    outputs = model.generate(
        input_ids=batch["input_ids"].to(settings.device),
        attention_mask=batch["attention_mask"].to(settings.device),
        # 生成されるシーケンスの最大長を定義します。
        # これは問題の性質によって異なりますが、
        # 一般的にはソーステキストの長さの2倍程度を指定することが推奨されます。
        max_length=settings.max_length_target,
        # 同じ文の繰り返し（モード崩壊）へのペナルティを定義します。
        # 値が大きいほど、生成されるテキストの繰り返しを避けることができます。
        # 適切な値は実験によりますが、通常は1.0以上の値を設定します。
        repetition_penalty=8.0,
        # 生成にランダム性を入れる温度パラメータです。
        # 値が小さいほど出力は決定的になり、大きいほど出力はランダムになります。
        # 適切な値は問題に依存しますが、一般的には0.7から1.0の間で設定します。
        # temperature=1.0,
        # ビームサーチの探索幅を定義します。
        # ビームサーチは、生成される各ステップで最良の候補を保持し、
        # それらの候補から次のステップを生成します。
        # num_beamsが大きいほど、より多くの候補が考慮され、
        # 生成されるテキストの質が向上する可能性がありますが、
        # 計算コストも増加します。一般的には2から10の間で設定します。
        # num_beams=10,
        # 生成結果の多様性を生み出すためのペナルティパラメータです。
        # これは特定のトークンが選択されることに対するペナルティを増加させ、
        # 結果として生成されるテキストの多様性を高めます。適切な値は問題と目的に依存します。
        # diversity_penalty=1.0,
        # ビームサーチのグループ数を定義します。
        # これは、ビームを複数のグループに分割し、
        # それぞれのグループで独立にサーチを行うことを可能にします。
        # これにより、出力の多様性が増加します。
        # num_beam_groups=10,
        # 生成する文の数を定義します。
        # このパラメータは、複数の異なる出力を生成したい場合に使用します。
        # この値はnum_beams以上である必要があります。
        # num_return_sequences=1,  # 生成する文の数
    )

    generated_texts = [
        tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs
    ]
    return generated_texts[0]


def preprocess_text(text: str) -> str:
    """与えられたテキストを前処理します。前処理として、改行やタブの削除、テキストの正規化、小文字化、余白の削除などが行われます。

    引数:
        text (str): 前処理するテキスト

    戻り値:
        str: 前処理されたテキスト
    """
    text = re.sub(r"[\r\t\n\u3000]", "", text)
    text = neologdn.normalize(text)
    text = text.lower()
    text = text.strip()
    return text
