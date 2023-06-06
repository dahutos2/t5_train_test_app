import logging
import uuid
import streamlit as st
from typing import Optional


class __StreamlitHandler(logging.Handler):
    """
    Streamlit でログメッセージを表示するためのログハンドラーです。

    ログをバッファに保存し、リストアやクリアの機能を提供します。
    """

    def __init__(self):
        super().__init__()
        self.buffer = []
        self.nav_bar = st.sidebar.empty()
        self.log_area = st.sidebar.empty()
        self.clear_button = st.sidebar.empty()

    def restore(self):
        self.log_area.write("\n".join(self.buffer))

    def emit(self, record):
        self.buffer.append(self.format(record))
        self.log_area.write("\n".join(self.buffer))

    def clear(self):
        self.buffer = []
        self.log_area.write("")


# ロガーの設定
handler = __StreamlitHandler()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s : \n    %(message)s\n",
    level=logging.INFO,
    handlers=[handler],
)
logger = logging.getLogger(__name__)

__is_enabled = True


def change_enabled(arg: Optional[bool] = None) -> None:
    """このメソッドは、引数がTrue の場合、ログをクリアするボタンが有効化され、False の場合は無効化します。

    引数:
        arg (Optional[bool], optional): 指定しない場合、__is_enabled の現在の値が使用されます。
    """
    # 引数が指定されていない場合、グローバル変数を使用
    global __is_enabled
    if arg is None:
        arg = __is_enabled
    else:
        __is_enabled = arg

    if arg:
        # 活性化
        handler.clear_button.button(
            "ログをクリア", on_click=handler.clear, key=str(uuid.uuid4())
        )
    else:
        # 非活性化
        handler.clear_button.button("ログをクリア", disabled=True, key=str(uuid.uuid4()))
