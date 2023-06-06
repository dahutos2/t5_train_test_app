import streamlit as st
from page import train
from page import test
from page import trial
from page import description
from share import log
from share import settings


def main():
    # アプリのタイトル
    st.title("T5モデルの訓練、評価アプリ")

    # ナビゲーションバーの作成
    option = log.handler.nav_bar.selectbox("どの機能を使用しますか？", ("説明", "訓練", "評価", "お試し"))

    # 初期化
    log.handler.restore()
    log.change_enabled()
    settings.set_setting()

    try:
        # 説明
        if option == "説明":
            description.show()
        # 訓練
        elif option == "訓練":
            train.show()
        # 評価
        elif option == "評価":
            test.show()
        # お試し
        elif option == "お試し":
            trial.show()
    except Exception as e:
        # ボタンを活性化
        log.change_enabled(True)
        st.error(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
