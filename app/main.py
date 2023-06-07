import streamlit as st
from page import train
from page import test
from page import trial
from page import description
from share import log
from share import settings


def main():
    # アプリのタイトル
    st.title("T5モデルの学習、検証アプリ")

    # ナビゲーションバーの作成
    option = log.handler.nav_bar.selectbox("どの機能を使用しますか？", ("説明", "学習", " 検証", "試験"))

    # 初期化
    log.handler.restore()
    log.change_enabled()
    settings.set_setting()

    try:
        # 説明
        if option == "説明":
            description.show()
        # 学習
        elif option == "学習":
            train.show()
        # 検証
        elif option == "検証":
            test.show()
        # 試験
        elif option == "試験":
            trial.show()
    except Exception as e:
        # ボタンを活性化
        log.change_enabled(True)
        st.error(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
