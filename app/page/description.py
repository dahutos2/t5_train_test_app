from typing import Tuple
import streamlit as st
import base64
import markdown
import sass
import os
import sys

# 実行ファイルの絶対パスを取得
__executable_path = os.path.abspath(sys.argv[0])

# 実行ファイルのディレクトリを取得
__executable_dir = os.path.dirname(__executable_path)

# 説明を保存するフォルダのパス
__DESCRIPTION_FOLDER = os.path.join(__executable_dir, "input", "description")

# 説明のSCSSを保存するファイルパス
__DESCRIPTION_FILE_SCSS = os.path.join(__DESCRIPTION_FOLDER, "description.scss")

# アプリ説明を保存するファイルパス
_DESCRIPTION_APP_FILE = os.path.join(__DESCRIPTION_FOLDER, "description_app.md")

# モデル説明を保存するファイルパス
_DESCRIPTION_MODEL_FILE = os.path.join(__DESCRIPTION_FOLDER, "description_model.md")

# T5説明を保存するファイルパス
_DESCRIPTION_T5_FILE = os.path.join(__DESCRIPTION_FOLDER, "description_t5.md")

# T5数学説明を保存するファイルパス
_DESCRIPTION_MATH_FILE = os.path.join(__DESCRIPTION_FOLDER, "description_math.md")

# 専門用語説明を保存するファイルパス
_DESCRIPTION_DIC_FILE = os.path.join(__DESCRIPTION_FOLDER, "description_dic.md")

# 画像ファイルのパス
__IMAGES_PATH = os.path.join(__executable_dir, "input", "images")


def show():
    """.mdファイルをcssを適用して表示する"""

    # CSS適用
    _show_css_file()

    # アプリ説明
    __show_app_file()

    # モデル説明
    __show_model_file()

    # T5説明
    __show_t5_file()

    # T5数学説明
    __show_math_file()

    # 専門用語説明
    __show_dic_file()


"""アプリ、T5、専門用語"""


def __show_app_file() -> None:
    html = __convert_to_html(_DESCRIPTION_APP_FILE)

    # StreamlitでHTMLを表示
    st.markdown(html, unsafe_allow_html=True)


def __show_t5_file() -> None:
    html = __convert_to_html(_DESCRIPTION_T5_FILE)

    # StreamlitでHTMLを表示
    st.markdown(html, unsafe_allow_html=True)


def __show_dic_file() -> None:
    html = __convert_to_html(_DESCRIPTION_DIC_FILE)

    # StreamlitでHTMLを表示
    st.markdown(html, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def __convert_to_html(file_path: str) -> str:
    md_text = __file_to_text(file_path)

    return __convert_to_styled_html(md_text)


"""CSS"""


def _show_css_file() -> None:
    css = __convert_to_css(__DESCRIPTION_FILE_SCSS)

    # CSSを適用する
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def __convert_to_css(file_path: str) -> str:
    # CSSファイルを読み込む
    scss = __file_to_text(file_path)
    # SCSSをCSSにコンパイル
    css = sass.compile(string=scss)

    return css


"""モデル"""


def __show_model_file() -> None:
    html = __convert_to_model(_DESCRIPTION_MODEL_FILE)

    # StreamlitでHTMLを表示
    st.markdown(html, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def __convert_to_model(file_path: str) -> str:
    md_text = __file_to_text(file_path)

    # 画像のパスを変換
    convert_image_dict = {
        "説明画像001": f'![T5](data:image/png;base64,{__get_image_as_base64_string(os.path.join(__IMAGES_PATH, "t5.png"))})',
        "説明画像002": f'![T5_Google](data:image/png;base64,{__get_image_as_base64_string(os.path.join(__IMAGES_PATH, "t5_google.png"))})',
        "説明画像003": f'![T5_Google](data:image/png;base64,{__get_image_as_base64_string(os.path.join(__IMAGES_PATH, "t5_google_train.png"))})',
    }
    for original, replacement in convert_image_dict.items():
        md_text = md_text.replace(original, replacement)

    return __convert_to_styled_html(md_text)


"""T5数学"""


def __show_math_file() -> None:
    md_text, html = __convert_to_math(_DESCRIPTION_MATH_FILE)

    # StreamlitでHTMLを表示
    st.markdown(html, unsafe_allow_html=True)
    st.markdown(md_text)


@st.cache_data(show_spinner=False)
def __convert_to_math(file_path: str) -> Tuple[str, str]:
    md_text = __file_to_text(file_path)

    start_md_text = "\n### 数式一覧\n"
    # MarkdownをHTMLに変換
    html = __convert_to_styled_html(start_md_text)

    return md_text, html


"""共通メソッド"""


def __convert_to_styled_html(md_text: str) -> str:
    # MarkdownをHTMLに変換
    html = markdown.markdown(
        md_text,
        extensions=[
            "fenced_code",
            "tables",
            "toc",
            "attr_list",
            "nl2br",
        ],
    )

    # CSSを適用したHTMLを生成
    styled_html = f"""
    <div class="description">
    {html}
    </div>
    """

    return styled_html


def __file_to_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


def __get_image_as_base64_string(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
