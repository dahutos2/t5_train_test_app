import json
import os
import sys
from typing import Any, Dict
from . import settings

# 実行ファイルの絶対パスを取得
__executable_path = os.path.abspath(sys.argv[0])

# 実行ファイルのディレクトリを取得
__executable_dir = os.path.dirname(__executable_path)

# アプリのパラメータの初期値を保存するファイルのパス
__PARAMS_FILE = os.path.join(__executable_dir, "params.json")

# アプリのパラメータの初期値
__default_params = {
    "model_path": settings.model_selections[1],
    "prefix": "translate English to Japanese: ",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 16,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "learning_rate": 5e-5,
    "gradient_accumulation_steps": 1,
}


def get_train_param() -> Dict[str, Any]:
    """このメソッドは、訓練パラメータを取得します。

    訓練パラメータは、定義済みのファイル(PARAMS_FILE)から読み込まれます。

    PARAMS_FILEが存在しない場合は、デフォルトのパラメータが返されます。

    戻り値:
        Dict[str, Any]: 訓練パラメータを保持する辞書
    """
    # PARAMS_FILEが存在すれば読み込む
    if os.path.exists(__PARAMS_FILE):
        with open(__PARAMS_FILE, "r") as f:
            saved_params = json.load(f)
            return saved_params
    else:
        return __default_params


def set_train_param(params: Dict[str, Any]) -> None:
    """このメソッドは、訓練パラメータを設定します。

    設定したパラメータは、定義済みのファイル(PARAMS_FILE)に書き込まれます。

    引数:
        params (Dict[str, Any]): 設定する訓練パラメータを保持する辞書
    """
    json_str = json.dumps(params)
    with open(__PARAMS_FILE, "w") as f:
        f.write(json_str)


def get_model_param() -> str:
    """このメソッドは、モデルのパスを取得します。

    モデルのパスは、定義済みのファイル(PARAMS_FILE)から読み込まれます。

    PARAMS_FILEが存在しない場合は、デフォルトのモデルパスが返されます。

    戻り値:
        str: モデルのパス
    """
    # PARAMS_FILEが存在すれば読み込む
    if os.path.exists(__PARAMS_FILE):
        with open(__PARAMS_FILE, "r") as f:
            saved_params = json.load(f)
            return saved_params["model_path"]
    else:
        return __default_params["model_path"]


def set_model_param(model_path: str) -> None:
    """このメソッドは、モデルのパスを設定します。

    設定したモデルのパスは、定義済みのファイル(PARAMS_FILE)に書き込まれます。

    引数:
        model_path (str): 設定するモデルのパス
    """
    params = get_train_param()
    params["model_path"] = model_path
    json_str = json.dumps(params)
    with open(__PARAMS_FILE, "w") as f:
        f.write(json_str)


def get_prefix_param() -> str:
    """このメソッドは、モデルに使用する、接頭辞を取得します。

    接頭辞は、定義済みのファイル(PARAMS_FILE)から読み込まれます。

    PARAMS_FILEが存在しない場合は、デフォルトの接頭辞が返されます。

    戻り値:
        str: モデルに使用する、接頭辞
    """
    # PARAMS_FILEが存在すれば読み込む
    if os.path.exists(__PARAMS_FILE):
        with open(__PARAMS_FILE, "r") as f:
            saved_params = json.load(f)
            return saved_params["prefix"]
    else:
        return __default_params["prefix"]


def set_prefix_param(prefix: str) -> None:
    """このメソッドは、モデルに使用する、接頭辞を設定します。

    設定した接頭辞は、定義済みのファイル(PARAMS_FILE)に書き込まれます。

    引数:
        prefix (str): モデルに使用する、接頭辞
    """
    params = get_train_param()
    params["prefix"] = prefix
    json_str = json.dumps(params)
    with open(__PARAMS_FILE, "w") as f:
        f.write(json_str)
