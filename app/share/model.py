import streamlit as st
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)
import gensim
from typing import Tuple
import zipfile
import datetime
import tempfile
import os
import sys
import shutil
from . import settings
from . import param
from . import log

__uploaded_file = None
__checked_is_fast_text = False
__keyed_vectors = None
__current_is_fast_text = False

__WORD2_VEC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(sys.argv[0])),
    os.path.join(
        "input",
        "model_word2vec",
        "word2vec.gensim.model",
    ),
)

__FAST_TEXT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(sys.argv[0])),
    os.path.join(
        "input",
        "model_fast_text",
        "model.vec",
    ),
)


def save(
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
) -> bytes:
    """指定のモデルを

    引数:
        tokenizer (T5Tokenizer): T5のトークナイザ

        model (T5ForConditionalGeneration): T5のモデル
    戻り値:
        bytes: モデル保存後のzipファイルのバイナリファイル
    """
    # モデルとトークナイザを保存
    log.logger.info("モデルの圧縮を開始します...")
    binary = get_binary_save_model(model, tokenizer)
    log.logger.info("モデルの圧縮が終了しました!")
    return binary


def get_model_path(
    default_model_path: any,
) -> Tuple[bool, str]:
    """このメソッドは、ユーザーが選択したモデルのパスと取得の成否を返します。

    引数:
        default_model_path (str): デフォルトのモデルパス。

    戻り値:
        Tuple[bool, str]: 選択されたモデルのパスと取得の成否。
    """
    index = (
        settings.model_selections.index(default_model_path)
        if default_model_path in settings.model_selections
        else 0
    )
    option = st.selectbox("モデルを選択して下さい", settings.model_selections, index=index)
    if option == "任意のモデル":
        uploaded_file = st.file_uploader("zip形式でモデルをアップロードして下さい", type=["zip"])

        global __uploaded_file
        if uploaded_file is not None:
            unique_path = f"model_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

            __uploaded_file = uploaded_file
            return True, unique_path
        else:
            st.error("モデルをアップロードして下さい。")
            return False, ""

    else:
        return True, option


def set_prefix() -> None:
    """モデルの役割を決定するための接頭辞を設定する"""
    prefix = param.get_prefix_param()
    prefix = st.text_area("モデルを使用する際の、接頭辞を入力して下さい。", prefix)
    param.set_prefix_param(prefix)


def get_load_model(model_path: str) -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
    """このメソッドは、指定されたパスからトークナイザとモデルを読み込みます。

    一度読み込んだモデルと同じパスが指定された場合、キャッシュに保存されたモデルとトークナイザが使用されます。

    それ以外の場合、新しくモデルとトークナイザを読み込みます。

    引数:
        model_path (str): モデルのパス

    戻り値:
        Tuple[T5Tokenizer, T5ForConditionalGeneration]: 読み込んだトークナイザとモデル
    """
    if model_path not in settings.model_selections:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_model_dir = os.path.join(tmp_dir, model_path)
            with zipfile.ZipFile(__uploaded_file, "r") as zip_ref:
                zip_ref.extractall(tmp_model_dir)

            tokenizer, model = __get_load_cache_model(tmp_model_dir)
    else:
        tokenizer, model = __get_load_cache_model(model_path)

    return tokenizer, model


def get_binary_save_model(
    model: T5ForConditionalGeneration, tokenizer: T5Tokenizer
) -> bytes:
    """このメソッドは、モデルとトークナイザをzip化し、binaryデータとして返します。

    Args:
        model (T5ForConditionalGeneration): 保存するモデル
        tokenizer (T5Tokenizer): 保存するトークナイザ

    Returns:
        bytes: モデルとトークナイザのbinaryに変換したもの
    """
    with tempfile.TemporaryDirectory() as tempdir:
        model.save_pretrained(tempdir)
        tokenizer.save_pretrained(tempdir)

        # フォルダをzipファイルに変換します
        zip_path = os.path.join(tempdir, "model")
        shutil.make_archive(zip_path, "zip", tempdir)

        # zipファイルをバイナリに変換します
        with open(zip_path + ".zip", "rb") as f:
            binary = f.read()

            return binary


@st.cache_resource(
    show_spinner="訓練後モデルを読み込み中...",
    ttl=datetime.timedelta(hours=6),
    max_entries=1,
)
def __get_load_cache_model(
    model_path: str,
) -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
    # トークナイザとモデルの準備
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    return tokenizer, model


def set_is_fast_text() -> None:
    """FastTextを使用するかどうかを選択する"""
    global __checked_is_fast_text
    current_checked = __checked_is_fast_text
    is_checked = st.checkbox(
        "評価にFastTextを使用する",
        value=current_checked,
        help="デフォルトでは、Word2Vecを使用します。",
    )
    __checked_is_fast_text = is_checked


def get_keyed_vectors() -> gensim.models.KeyedVectors:
    """単語の類似度算出用モデルを返す

    戻り値:
        gensim.models.KeyedVectors: キー（単語など）とベクトルとのマッピング
    """
    # 単語の類似度算出用モデルをロードする
    with st.spinner("評価用モデルを読み込み中..."):
        if __checked_is_fast_text:
            return __get_keyed_vectors(__FAST_TEXT_PATH)
        else:
            return __get_keyed_vectors(__WORD2_VEC_PATH)


def __get_keyed_vectors(model_path: str) -> gensim.models.KeyedVectors:
    global __keyed_vectors
    if __keyed_vectors == None:
        if model_path == __FAST_TEXT_PATH:
            __keyed_vectors = __get_fast_text()
        else:
            __keyed_vectors = __get_word2_vec()
    else:
        if not __current_is_fast_text and model_path == __FAST_TEXT_PATH:
            __keyed_vectors = __get_fast_text()
        if __current_is_fast_text and model_path == __WORD2_VEC_PATH:
            __keyed_vectors = __get_word2_vec()

    return __keyed_vectors


def __get_word2_vec():
    global __current_is_fast_text
    __current_is_fast_text = False
    return gensim.models.Word2Vec.load(__WORD2_VEC_PATH).wv


def __get_fast_text():
    global __current_is_fast_text
    __current_is_fast_text = True
    return gensim.models.KeyedVectors.load_word2vec_format(
        __FAST_TEXT_PATH,
        binary=False,
    )
