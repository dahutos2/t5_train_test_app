# 目的
- 学習検証実行可能なアプリ
## 実行
### コマンド
- 実行
  - `streamlit run app.py`
## 仮想環境構築(Anaconda)
### 名前
- train_app_env
### コマンド
- 作成
  - `conda create -n train_app_env  python=3.8`
- 活性化
  - `conda activate train_app_env`
- 非活性化
  - `conda deactivate`
### インストールパッケージ
#### ファイル名
- ./requirements.txt
#### インストールコマンド
- `pip install -r requirements.txt`
### VSCode設定
- Pythonインタプリタの適応
  - Python: Select Interpreter
  - `~/opt/anaconda3/envs/train_app_env/bin/python`
    - 仮想環境のフォルダパスを選択する
- 実行設定ファイルを設定する
  - `./launch.json`
### 検証実行用のモデルのダウンロードと配置
- ダウンロード対象物
  - http://public.shiroyagi.s3.amazonaws.com/latest-ja-word2vec-gensim-model.zip
- 配置位置
  - /app/input/model_word2vec
  ※ダウンロードしたフォルダごと配置すること