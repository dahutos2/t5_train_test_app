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

# アプリの説明を保存するファイルのパス
__MARKDOWN_FILE = os.path.join(__executable_dir, "input", "description.md")

# アプリの説明のSCSSを保存するファイルのパス
__MARKDOWN_FILE_SCSS = os.path.join(__executable_dir, "input", "description.scss")


def __get_image_as_base64_string(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


# 画像ファイルのパス
__IMAGES_PATH = os.path.join(__executable_dir, "input", "images")


def show():
    """.mdファイルをcssを適用して表示する"""

    styled_html, css = __get_file_data(__MARKDOWN_FILE, __MARKDOWN_FILE_SCSS)

    # CSSを適用する
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # StreamlitでHTMLを表示
    st.markdown(styled_html, unsafe_allow_html=True)


@st.cache_data(
    show_spinner="アプリ説明ファイルを読み込み中...",
    max_entries=1,
)
def __get_file_data(md_path, scss_path):
    # Markdownファイルを読み込む
    md_text = __file_to_text(md_path)

    # 画像のパスを変換
    convert_image_dict = {
        "説明画像001": f'![T5](data:image/png;base64,{__get_image_as_base64_string(os.path.join(__IMAGES_PATH, "t5.png"))})',
        "説明画像002": f'![T5_Google](data:image/png;base64,{__get_image_as_base64_string(os.path.join(__IMAGES_PATH, "t5_google.png"))})',
        "説明画像003": f'![T5_Google](data:image/png;base64,{__get_image_as_base64_string(os.path.join(__IMAGES_PATH, "t5_google_train.png"))})',
    }
    for original, replacement in convert_image_dict.items():
        md_text = md_text.replace(original, replacement)

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

    # CSSファイルを読み込む
    scss = __file_to_text(scss_path)
    # SCSSをCSSにコンパイル
    css = sass.compile(string=scss)

    # CSSを適用したHTMLを生成
    styled_html = f"""
    <div class="description">
    {html}
    </div>
    """

    return styled_html, css


def __file_to_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text
