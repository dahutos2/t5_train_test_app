## アプリ説明 
### 訓練
- csvファイルをアップロードし、指定したハイパーパラメータの値で訓練を行うことができます。
- 訓練を行わず、保存のみを行うこともできます。
#### データの形式(csv)
| Input        | Output           |
| ------------ | ---------------- |
| 入力元文字列 | 出力したい文字列 |
| …            | …                |

### 評価
- csvファイルをアップロードし、選択した訓練済みモデルを、評価することが出来ます。
- 検証とは、モデルが生成した文字列とcsvファイルの文字列の類似度で確認します。
#### データの形式(csv)
| Input        | Output               |
| ------------ | -------------------- |
| 入力元文字列 | 出力されるべき文字列 |
| …            | …                    |

### お試し(T5)
- 選択したモデルの動作を試すことができます。
- 単体と複数の二つを選択可能です。
#### 単体
- 指定した入力文字列からモデルの機能に沿ってを出力します。
#### 複数
- 複数の入力文字列含むcsvファイルをアップロードし、出力結果をテーブル形式で表示します。
##### データの形式(csv)
| Input        |
| ------------ |
| 入力元文字列 |
| …            |

### お試し(Word2Vec)
- 単語に対するコサイン類似度(-1~1)を確認することができます。
- 類似度の比較、類似した単語の生成、単語の足し引きが行えます。
### ログ
- 訓練状況や、評価の状況をログで確認することが出来ます。
### 訓練済みモデル一覧
#### Huggingface
<details>
<summary>クリックして展開します</summary>

##### T5ベースのモデルカード
説明画像001
##### モデルの詳細
###### モデルの説明
T5では、クラスラベルか入力のスパンしか出力できないBERTスタイルのモデルとは対照的に、すべてのNLPタスクを、入力と出力が常にテキスト文字列である統一されたテキスト-テキスト形式に再構築することを提案します。
このtext-to-textフレームワークにより、どのようなNLPタスクに対しても同じモデル、損失関数、ハイパーパラメータを使用することができます。
###### 言語 (NLP)
英語、フランス語、ルーマニア語、ドイツ語
###### 関連リンク
[研究論文](https://jmlr.org/papers/volume21/20-074/20-074.pdf)
[Google の T5 ブログ投稿](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)
[GitHub リポジトリ](https://github.com/google-research/text-to-text-transfer-transformer)
[Huggingface T5 ドキュメント](https://huggingface.co/docs/transformers/model_doc/t5)
##### 用途
text-to-Text フレームワークを使用すると、機械翻訳、文書要約、質問応答、分類タスク (感情分析など) を含む、あらゆる NLP タスクで同じモデル、損失関数、ハイパーパラメーターを使用できます。
数値そのものではなく、数値の文字列表現を予測するように T5 をトレーニングすることで、T5 を回帰タスクに適用することもできます。
##### トレーニングの詳細
###### トレーニングデータ
モデルは、T5 と同じ[研究論文](https://jmlr.org/papers/volume21/20-074/20-074.pdf)のコンテキストで開発およびリリースされた [Colossal Clean Crawled Corpus (C4)](https://www.tensorflow.org/datasets/catalog/c4?hl=ja) で事前トレーニングされています。
モデルは、教師なしタスク (1.) と教師ありタスク (2.) のマルチタスク混合物で事前トレーニングされました。

1. 教師なしノイズ除去目的に使用されるデータセット
    - [C4](https://huggingface.co/datasets/c4)
    - [Wiki-DPR](https://huggingface.co/datasets/wiki_dpr)
2. 教師ありテキスト対テキスト言語モデリングの目的に使用されるデータセット
    - 量刑の適否判定
        - [CoLAワルシュタットほか、2018](https://arxiv.org/abs/1805.12471)
    - 感情分析
        - [ST-2 Socher 他、2013](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
    - 言い換え/文の類似性
        - [MRPCドーランとブロケット、2005 年](https://aclanthology.org/I05-5002/)
        - [STS-B認定者、2017](https://arxiv.org/abs/1708.00055)
        - [QQPアイヤー他、2017](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)
    - 自然言語推論
        - [MNLIウィリアムズ他、2017](https://arxiv.org/abs/1704.05426)
        - [QNLI Rajpurkar 他、2016](https://arxiv.org/abs/1606.05250)
        - [RTEダガン他、2005](https://link.springer.com/chapter/10.1007/11736790_9)
        - [CBデ・マルネフ他、2019](https://semanticsarchive.net/Archive/Tg3ZGI2M/Marneffe.pdf)
    - 文の完成
        - [COPA Roemmele 他、2011](https://www.researchgate.net/publication/221251392_Choice_of_Plausible_Alternatives_An_Evaluation_of_Commonsense_Causal_Reasoning)
    - 語感の曖昧さ回避
        - [WICピレヴァルとカマチョ・コジャドス、2018](https://arxiv.org/abs/1808.09121)
    - 質疑応答
        - [MultiRC Khashabi 他、2018](https://aclanthology.org/N18-1023/)
        - [ReCoRD Zhang 他、2018](https://arxiv.org/abs/1810.12885)
        - [BoolQ Clark 他、2019](https://arxiv.org/abs/1905.10044)
###### トレーニング手順
このペーパーでは、あらゆる言語の問題をテキストからテキストへの形式に変換する統一フレームワークを導入することにより、NLP の転移学習テクニックの展望を探ります。
私たちの体系的な研究では、数十の言語理解タスクに関する事前トレーニングの目的、アーキテクチャ、ラベルのないデータセット、転送アプローチ、その他の要素を比較しています。
##### t5-small
6000万個のパラメータを持つチェックポイントです。
##### t5-base
2億2000万個のパラメータを持つチェックポイントです。
##### t5-large
7億7000 万個のパラメータを持つチェックポイントです。
</details>
<p></p>

#### Google
<details>
<summary>クリックして展開します</summary>

##### T5ベースのモデルカード
説明画像002
##### モデルの詳細
###### モデルの説明
すでに T5 を知っている場合は、FLAN-T5 の方がすべてにおいて優れています。
同じ数のパラメーターについて、これらのモデルは、より多くの言語をカバーする 1,000 以上の追加タスクで微調整されています。
Flan-PaLM 540B は、5 ショット MMLU で 75.2% など、いくつかのベンチマークで最先端のパフォーマンスを達成します。
また、PaLM 62B などのはるかに大型のモデルと比較しても強力な数ショット性能を実現する Flan-T5 チェックポイント 1 も公開しています。
全体として、命令の微調整は、事前トレーニングされた言語モデルのパフォーマンスと使いやすさを向上させるための一般的な方法です。
###### 言語 (NLP)
英語、スペイン語、日本語、ペルシア語、ヒンディー語、フランス語、中国語、ベンガル語、グジャラート語、ドイツ語、テルグ語、イタリア語、アラビア語、ポーランド語、タミル語、マラーティー語、マラヤーラム語、オリヤー語、パンジャブ語、ポルトガル語、ウルドゥー語、ガリシア語、ヘブライ語、韓国語、カタルーニャ語、タイ語、オランダ語、インドネシア語、ベトナム語、ブルガリア語、フィリピン語、中央クメール語、ラオス語、トルコ語、ロシア語、クロアチア語、スウェーデン語、ヨルバ語、クルド語、ビルマ語、マレー語、チェコ語、フィンランド語、ソマリ語、タガログ語、スワヒリ語、シンハラ語、カンナダ語、チワン語、イボ語、コーサ語、ルーマニア語、ハイチ語、エストニア語、スロバキア語、リトアニア語、ギリシャ語、ネパール語、アッサム語、ノルウェー語
###### 関連リンク
[研究論文](https://arxiv.org/pdf/2210.11416.pdf)
[GitHub リポジトリ](https://github.com/google-research/t5x)
[Huggingface T5 ドキュメント](https://huggingface.co/docs/transformers/model_doc/t5)
##### 使用法
モデルを使用する方法については、以下のサンプルスクリプトを参照してください。
###### Pytorch モデルの使用
- CPU 上でモデルを実行する
<details>
<summary>クリックして展開します</summary>

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```
</details>
<p></p>

- GPU でのモデルの実行
<details>
<summary>クリックして展開します</summary>

```python
# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```
</details>
<p></p>

###### さまざまな精度を使用して GPU でモデルを実行する
- FP16
<details>
<summary>クリックして展開します</summary>

```python
# pip install accelerate
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto", torch_dtype=torch.float16)

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))

```
</details>
<p></p>

- INT8
<details>
<summary>クリックして展開します</summary>

```python
# pip install bitsandbytes accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto", load_in_8bit=True)

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```
</details>

##### 用途
主な用途は、次のような言語モデルの研究です。
ゼロショット NLP タスクと、推論や質問応答などのコンテキスト内での少数ショット学習 NLP タスクの研究。
公平性と安全性の研究を進め、現在の大規模な言語モデルの限界を理解する。
##### トレーニングの詳細
###### トレーニングデータ
モデルは、以下の表で説明されているタスクを含むタスクの混合でトレーニングされました 。
説明画像003
###### トレーニング手順
これらのモデルは、事前トレーニングされた T5 (Raffel et al., 2020) に基づいており、ゼロショットおよび少数ショットのパフォーマンスを向上させるための命令で微調整されています。
T5 モデル サイズごとに 1 つの微調整された Flan モデルがあります。
##### input
###### Translation(翻訳)
Translate to German:  My name is Arthur
###### Question Answering(質問回答)
Please answer to the following question. Who is going to be the next Ballon d'or?(
###### Logical reasoning(論理的な推論)
Q: Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering.
###### Scientific knowledge(科学的知識)
Please answer the following question. What is the boiling point of Nitrogen?
###### Yes/no question(はい/いいえ の質問)
Answer the following yes/no question. Can you write a whole Haiku in a single tweet?
###### Reasoning task(推論タスク)
Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?
###### Boolean Expressions(ブール式)
Q: ( False or not False or False ) is? A: Let's think step by step
###### Math reasoning(数学的な推論)
The square root of x is the cube root of y. What is y to the power of 2, if x = 4?
###### Premise and hypothesis(前提・仮説)
Premise:  At my age you will probably have learnt one lesson. Hypothesis:  It's not certain how many lessons you'll learn by your thirties. Does the premise entail the hypothesis? 
##### google/flan-t5-small
##### google/flan-t5-base
##### google/flan-t5-large
##### google/flan-t5-xl
##### google/flan-t5-xxl
</details>
<p></p>

#### Retrieva-jp
<details>
<summary>クリックして展開します</summary>

##### モデルの詳細
T5 は、Transformer ベースの Encoder-Decoder モデルです。
オリジナルの T5 に対して次の点が改善されています。

- ReLU ではなく、フィードフォワード隠れ層での [GEGLU のアクティブ化](https://arxiv.org/abs/2002.05202)を参照してください。
- ドロップアウトは事前トレーニングでオフになりました (品質優先)。
微調整中にドロップアウトを再度有効にする必要があります。
- 埋め込み層と分類層の間でパラメータを共有しない。
- 「xl」と「xxl」は「3B」と「11B」を置き換えます。
モデルの形状は少し異なります。
d_model が大きく、num_heads と d_ff が小さくなります。

このモデルは T5 v1.1 に基づいています。
日本語コーパスで事前トレーニングされました。
日本語コーパスには日本語版WikipediaとmC4/jaを使用しました。
##### モデルの説明
言語 (NLP):日本語
##### トレーニングの詳細
このモデルのトレーニングには[T5X](https://github.com/google-research/t5x) を使用しており、Huggingface トランスフォーマー形式に変換されています。
##### トレーニングデータ
- 多言語C4(mC4/ja)の日本語部分。
- 日本語ウィキペディア(20220920)。
##### 前処理
- ひらがなを 1 文字も使用していない文書を削除します。
これにより、英語のみのドキュメントと中国語のドキュメントが削除されます。
- URL のトップレベル ドメインを使用してアフィリエイト サイトを削除するホワイトリスト スタイルのフィルタリング。
##### トレーニングのハイパーパラメータ
- dropout rate: 0.0
- バッチサイズ: 256
- fp32
- 入力長さ: 512
- 出力長さ: 114
- それ以外の場合は、以下を含む[T5X]( https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/)のデフォルト値に従います。
    - オプティマイザー: Adafactor
    - 基本学習率: 1.0
    - ウォームアップステップ: 10000
##### retrieva-jp/t5-small-short
- 589824 ステップをトレーニングしました。
- サイズ: 小規模 (約 7,700 万パラメータ)
##### retrieva-jp/t5-small-medium
- 1048576 ステップをトレーニングしました。
- サイズ: 小規模 (約 7,700 万パラメータ)
##### retrieva-jp/t5-small-long
- 2097152 ステップをトレーニングしました。
- サイズ: 小規模 (約 7,700 万パラメータ)
##### retrieva-jp/t5-base-short
- 524288 ステップをトレーニングしました。
- サイズ: 基本(~2 億 2,000 万パラメータ)
##### retrieva-jp/t5-base-medium
- 1048576 ステップをトレーニングしました。
- サイズ: 基本(~2 億 2,000 万パラメータ)
##### retrieva-jp/t5-base-long
- 2097152 ステップをトレーニングしました。
- サイズ: 基本(~2 億 2,000 万パラメータ)
##### retrieva-jp/t5-large-short
- 524288 ステップをトレーニングしました。
- サイズ: 大(~7 億 7,000 万のパラメータ)
##### retrieva-jp/t5-large-medium
- 1008824 ステップをトレーニングしました。
- サイズ: 大(~7 億 7,000 万のパラメータ)
##### retrieva-jp/t5-large-long
- 2097152 ステップをトレーニングしました。
- サイズ: 大(~7 億 7,000 万のパラメータ)
##### retrieva-jp/t5-xl
- 524288 ステップをトレーニングしました。
- サイズ: XL(~30 億パラメータ)
</details>
<p></p>

#### Sonoisa
<details>
<summary>クリックして展開します</summary>

##### sonoisa/t5-base-japanese-mC4-Wikipedia
日本語T5事前学習済みモデル
次の日本語コーパス（約890GB）を用いて事前学習を行ったT5 (Text-to-Text Transfer Transformer) モデルです。

- [Wikipedia](https://ja.wikipedia.org/wiki/%E3%83%A1%E3%82%A4%E3%83%B3%E3%83%9A%E3%83%BC%E3%82%B8)の日本語ダンプデータ (2020年7月6日時点のもの)
- [mC4](https://github.com/allenai/allennlp/discussions/5056)の日本語コーパス（正確にはc4/multilingualのjaスプリット）

このモデルは事前学習のみを行なったものであり、特定のタスクに利用するにはファインチューニングする必要があります。
本モデルにも、大規模コーパスを用いた言語モデルにつきまとう、学習データの内容の偏りに由来する偏った（倫理的ではなかったり、有害だったり、バイアスがあったりする）出力結果になる問題が潜在的にあります。 
この問題が発生しうることを想定した上で、被害が発生しない用途にのみ利用するよう気をつけてください。

SentencePieceトークナイザーの学習には上記Wikipediaの全データを用いました。

- [転移学習のサンプルコード](https://github.com/sonoisa/t5-japanese)
##### sonoisa/t5-base-english-japanese
英語+日本語T5事前学習済みモデル
次の日本語コーパス（約500GB）を用いて事前学習を行ったT5 (Text-to-Text Transfer Transformer) モデルです。

- [Wikipedia](https://en.wikipedia.org/wiki/Main_Page)の英語ダンプデータ (2022年6月27日時点のもの)
- [Wikipedia](https://ja.wikipedia.org/wiki/%E3%83%A1%E3%82%A4%E3%83%B3%E3%83%9A%E3%83%BC%E3%82%B8)の日本語ダンプデータ (2022年6月27日時点のもの)
- [OSCAR](https://oscar-project.org/)の日本語コーパス
- [CC-100](https://data.statmt.org/cc-100/)の英語コーパス
- [CC-100](https://data.statmt.org/cc-100/)の日本語コーパス

このモデルは事前学習のみを行なったものであり、特定のタスクに利用するにはファインチューニングする必要があります。
本モデルにも、大規模コーパスを用いた言語モデルにつきまとう、学習データの内容の偏りに由来する偏った（倫理的ではなかったり、有害だったり、バイアスがあったりする）出力結果になる問題が潜在的にあります。 この問題が発生しうることを想定した上で、被害が発生しない用途にのみ利用するよう気をつけてください。

SentencePieceトークナイザーの学習には、上記WikipediaとCC-100を約10:1の比率で混ぜ、英語と日本語の文字数がほぼ同数になるように調整（文はランダムに抽出）したデータから2650万文選んだデータを用いました。byte-fallbackあり設定で学習しており、実質未知語が発生しません。

- [転移学習のサンプルコード](https://github.com/sonoisa/t5-japanese)
##### sonoisa/sentence-t5-base-ja-mean-tokens
日本語用Sentence-T5モデルです。

事前学習済みモデルとしてsonoisa/t5-base-japaneseを利用しました。
推論の実行にはsentencepieceが必要です（pip install sentencepiece）。

- 使い方
<details>
<summary>クリックして展開します</summary>

```python
from transformers import T5Tokenizer, T5Model
import torch


class SentenceT5:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, is_fast=False)
        self.model = T5Model.from_pretrained(model_name_or_path).encoder
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)


MODEL_NAME = "sonoisa/sentence-t5-base-ja-mean-tokens"
model = SentenceT5(MODEL_NAME)

sentences = ["暴走したAI", "暴走した人工知能"]
sentence_embeddings = model.encode(sentences, batch_size=8)

print("Sentence embeddings:", sentence_embeddings)
```
</details>
<p></p>

##### sonoisa/t5-base-japanese-adapt
日本語T5 Prefix Language Model

このモデルは[日本語T5事前学習済みモデル](https://huggingface.co/sonoisa/t5-base-japanese-v1.1)を初期値にして、
[Adapted Language Modelタスク](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#lm-adapted-t511lm100k)（与えられたトークン列の続きのトークン列を予測するタスク）用に100Kステップ追加学習したものです。

追加学習には次の日本語コーパス（約100GB）を用いました。

- [Wikipedia](https://ja.wikipedia.org/wiki/%E3%83%A1%E3%82%A4%E3%83%B3%E3%83%9A%E3%83%BC%E3%82%B8)の日本語ダンプデータ (2022年6月27日時点のもの)
- [OSCAR](https://oscar-project.org/)の日本語コーパス
- [CC-100](https://data.statmt.org/cc-100/)の日本語コーパス

- 使い方
<details>
<summary>クリックして展開します</summary>

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import textwrap

tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-prefixlm-base-japanese", is_fast=False)
trained_model = T5ForConditionalGeneration.from_pretrained("sonoisa/t5-prefixlm-base-japanese")

# GPUの利用有無
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    trained_model.cuda()

# 推論モード設定
trained_model.eval()

# 前処理とトークナイズを行う
inputs = [normalize_text("深層学習（ディープラーニング）とは、")]
batch = tokenizer.batch_encode_plus(
    inputs, max_length=1024, truncation=True, 
    padding="longest", return_tensors="pt")

input_ids = batch['input_ids']
input_mask = batch['attention_mask']
if USE_GPU:
    input_ids = input_ids.cuda()
    input_mask = input_mask.cuda()

# 生成処理を行う
outputs = trained_model.generate(
    input_ids=input_ids, attention_mask=input_mask, 
    max_length=256,
    temperature=1.0,  # 生成にランダム性を入れる温度パラメータ
    num_beams=10,  # ビームサーチの探索幅
    diversity_penalty=1.0,  # 生成結果の多様性を生み出すためのペナルティパラメータ
    num_beam_groups=10,  # ビームサーチのグループ
    num_return_sequences=10,  # 生成する文の数
    repetition_penalty=2.0,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
)

# 生成されたトークン列を文字列に変換する
generated_bodies = [tokenizer.decode(ids, skip_special_tokens=True, 
                                     clean_up_tokenization_spaces=False) 
                    for ids in outputs]

# 生成された文章を表示する
for i, body in enumerate(generated_bodies):
    print("\n".join(textwrap.wrap(f"{i+1:2}. {body}")))
```
</details>
<p></p>

- 実行結果:
<details>
<summary>クリックして展開します</summary>

```
 1. 様々なデータから、そのデータを抽出し、解析する技術です。深層学習とは、ディープラーニングの手法の一つです。ディープラーニングは、コン
ピュータが行う処理を機械学習で実現する技術です。ディープラーニングは、コンピュータが行う処理を機械学習で実現する技術です。ディープラーニング
は、コンピュータが行う処理を機械学習で実現する技術です。ディープラーニングは、コンピュータが行う処理を機械学習で実現する技術です。

 2. 様々なデータから、そのデータを抽出し、解析する技術です。深層学習とは、ディープラーニングの手法の一つです。ディープラーニングは、コン
ピュータが行う処理を機械学習で実現する技術です。

 3. ディープラーニングの手法のひとつで、人間の脳をモデル化し、そのデータを解析する手法である。深層学習は、コンピュータが処理するデータの
量や質を予測し、それを機械学習に応用するものである。この手法は、人工知能(ai)の分野において広く用いられている。

 4. 深層学習(deep learning)の手法の一つ。ディープラーニングとは、人間の脳に蓄積されたデータを解析し、そのデータから得られ
た情報を分析して、それを機械学習や人工知能などの機械学習に応用する手法である。ディープラーニングは、コンピュータが処理するデータの量と質を測
定し、その結果を機械学習で表現する手法である。ディープラーニングは、人間が行う作業を機械学習で表現する手法である。ディープラーニングは、人間
にとって最も身近な技術であり、多くの人が利用している。ディープラーニングは、コンピューターの処理能力を向上させるために開発された。

 5. 人間の脳の深層学習を応用した人工知能(ai)である。ディープラーニングは、コンピュータが処理するデータを解析し、そのデータから得られ
た結果を分析して、それを機械学習やディープラーニングに変換して学習するものである。ディープラーニングは、人間が行う作業を機械学習で再現するこ
とを可能にする。

 6. 人間の脳の深層学習を応用した人工知能(ai)である。ディープラーニングは、コンピュータが処理するデータを解析し、そのデータから得られ
た結果を分析して、それを機械学習やディープラーニングに変換して学習するものである。ディープラーニングは、人間が行う作業を機械学習で再現するこ
とを可能にする。ディープラーニングは、人間と機械との対話を通じて、人間の脳の仕組みを理解することができる。

 7. 深層学習によって、人間の脳の神経細胞や神経細胞を活性化し、その働きを解明する手法です。ディープラーニングは、コンピュータが処理するデ
ータの量と質を測定することで、人間が行う作業の効率化を図る手法です。また、機械学習では、人間に与えられたデータを解析して、それを機械学習で処
理する技術です。このため、機械学習では、機械学習のアルゴリズムを組み合わせることで、より精度の高い処理を実現します。さらに、機械学習では、機
械学習のアルゴリズムを組み合わせることで、より精度の高い処理を実現します。

 8. 学習したデータを解析し、そのデータの分析を行う手法。ディープラーニングは、コンピュータが人間の脳に与える影響を予測する技術である。深
層学習は、人間が脳に与える影響を予測する技術である。ディープラーニングは、人間と機械との相互作用によって行われる。例えば、ロボットが物体を分
解して、それを検出したり、物体を分解して、その物体を別の物体として認識したりする。また、物体を分解して、その物体を別の物体として認識する。こ
のプロセスは、コンピューターが行う処理よりもはるかに高速である。ディープラーニングは、コンピュータが行う処理よりも速く、より効率的な処理が可
能である。

 9. 深層学習によって、人間の脳の神経細胞や神経細胞を活性化し、その働きを解明する手法です。ディープラーニングは、コンピュータが処理するデ
ータの量と質を測定することで、人間が行う作業の効率化を図る手法です。

10. 学習したデータを解析し、そのデータの分析を行う手法。ディープラーニングは、コンピュータが人間の脳に与える影響を予測する技術である。深
層学習は、人間が脳に与える影響を予測する技術である。ディープラーニングは、人間と機械との相互作用によって行われる。例えば、ロボットが物体を分
解して、それを検出したり、物体を分解して、その物体を別の物体として認識したりする。また、物体を分解して、その物体を別の物体として認識する。こ
のプロセスは、コンピューターが行う処理よりもはるかに高速である。

```
</details>
<p></p>

##### sonoisa/t5-base-japanese-question-generation
[回答と回答が出てくるパラグラフを与えると質問文を生成するモデル](https://github.com/sonoisa/deep-question-generation)
###### 本モデルの作成ステップ概要
1. [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/)を日本語に機械翻訳し、不正なデータをクレンジング（有効なデータは約半分）。
回答が含まれるコンテキスト、質問文、解答の3つ組ができる。
1. 日本語T5モデルを次の設定でファインチューニング
- 入力: "answer: {解答} content: {回答が含まれるコンテキスト}"
- 出力: "{質問文}"
- 各種ハイパーパラメータ
  - 最大入力トークン数: 512
  - 最大出力トークン数: 64
  - 最適化アルゴリズム: AdaFactor
  - 学習率: 0.001（固定）
  - バッチサイズ: 128
  - ステップ数: 2500（500ステップごとにチェックポイントを出力、定量・定性評価を行い2500ステップ目を採用）
##### sonoisa/t5-base-japanese-title-generation
[記事本文からタイトルを生成するモデル](https://qiita.com/sonoisa/items/a9af64ff641f0bbfed44)
##### sonoisa/t5-qiita-title-generation
[記事本文からタイトルを生成するモデル](https://qiita.com/sonoisa/items/30876467ad5a8a81821f)
##### sonoisa/t5-base-japanese-article-generation
[タイトルから記事本文を生成するモデル](https://qiita.com/sonoisa/items/a9af64ff641f0bbfed44)
</details>
<p></p>

## T5について
### 概要
T5（Text-to-Text Transfer Transformer）は、Googleの研究チームによって開発された自然言語処理（NLP）のためのトランスフォーマーベースのモデルです。
T5は、すべてのNLPタスクをテキストからテキストへの変換問題として扱うというユニークなアプローチを採用しています。
### 内部構造
T5は、エンコーダとデコーダの両方を持つトランスフォーマーモデルです。
エンコーダは入力テキストを連続したベクトル表現に変換し、デコーダはそのベクトル表現を元に出力テキストを生成します。
トランスフォーマーモデルの特徴的な部分は、自己注意（Self-Attention）メカニズムです。
自己注意（Self-Attention）メカニズムは、入力テキスト内の各単語が他の単語とどのように関連しているかをモデルが理解するための手法です。
自己注意メカニズムの基本的なアイデアは、各単語が他の全ての単語に「注意」を払い、それぞれの単語からの情報を集約することです。
具体的には、各単語に対して「クエリ」、「キー」、「バリュー」の3つのベクトルを計算します。
これらのベクトルは、モデルのパラメータと入力単語のエンベディング（ベクトル表現）の積によって得られます。
次に、各単語のクエリベクトルと他の全ての単語のキーベクトルとの間の類似度（通常は内積）を計算します。
これにより、各単語が他の単語にどれだけ注意を払うべきか（つまり、他の単語からどれだけ情報を取り入れるべきか）を決定します。
これらの類似度スコアはソフトマックス関数を通じて正規化され、合計が1になるようにします。
最後に、これらの正規化された類似度スコアを各単語のバリューベクトルに掛けて、加重平均を計算します。
これにより、各単語に対する新しいベクトル表現が得られます。
この新しいベクトル表現は、各単語が他の全ての単語から得た情報を反映しています。
この自己注意メカニズムにより、モデルは文脈に応じた単語の表現を学習することができます。
例えば、"I"という単語があった場合、それが"eat"という単語の近くにあるか、"sleep"という単語の近くにあるかによって、その意味や重要性が変わるかもしれません。自己注意メカニズムは、このような文脈依存性を捉えることができます。
### 学習の影響
T5モデルの学習では、モデルのパラメータ（重みとバイアス）が更新されます。
これらのパラメータは、モデルが入力データから出力を生成する方法を制御します。
具体的には、各パラメータはモデル内の特定の接続の強さを表しており、これらの接続の強さが変化すると、モデルの出力も変化します。
パラメータの更新は、勾配降下法という最適化アルゴリズムを通じて行われます。このアルゴリズムは、モデルの出力と目標値との間の誤差（損失）を計算し、この損失を最小化するようにパラメータを調整します。
具体的には、損失関数の勾配（つまり、パラメータを少しだけ変化させたときの損失の変化量）を計算します。
そして、この勾配に基づいてパラメータを更新します。
勾配が正の場合、パラメータを減らすことで損失を減らすことができます。
逆に、勾配が負の場合、パラメータを増やすことで損失を減らすことができます。
このプロセスは、全てのパラメータに対して同時に行われ、多数のエポック（全データセットを通じての学習サイクル）にわたって繰り返されます。
これにより、モデルは徐々にデータからパターンを学習し、損失を最小化します。
なお、パラメータの更新量は学習率によって制御されます。
学習率は大きければ大きなステップでパラメータを更新し、小さければ小さなステップで更新します。
適切な学習率を選択することは、モデルの学習速度と性能に大きな影響を与えます。
### 単語の生成
生成処理は、学習されたモデルを使って新しいテキストを生成するプロセスです。
T5では、エンコーダが入力テキストをベクトル表現に変換し、このベクトル表現をデコーダが出力テキストに変換します。デコーダは、一度に1つの単語を生成し、生成された単語を次のステップの入力として使用します。
このプロセスは、終了トークンが生成されるか、または指定された最大長に達するまで続けられます。
生成にはいくつかの戦略があります。
最も単純なのは、各ステップで最も確率的に高い単語を選択する「貪欲なデコーディング」です。
しかし、これは必ずしも最適な結果をもたらさないため、より洗練された戦略が使用されることがあります。
例えば、「ビームサーチ」は、各ステップで複数の可能性を追跡し、全体として最も確率的に高いシーケンスを見つけ出します。
また、「トップ-k サンプリング」や「トップ-p サンプリング」は、ランダム性を導入してより多様な結果を生成します。
## 専門用語説明
### 共通
#### AI（Artificial Intelligence）
人工知能。
人間の知能をコンピュータ上で再現したもの。
これには、学習、推論、問題解決、知識表現、認識などの能力が含まれます。
#### Machine Learning (ML)
機械学習。
AIの一分野で、コンピュータがデータから学習し、その学習結果に基づいて予測や判断を行う技術です。
#### Deep Learning (DL)
深層学習。
大量のデータから学習する際に、多層のニューラルネットワーク（人間の脳の神経細胞がつながっている構造を模倣したモデル）を用いることで高度な予測や判断を行う技術です。
#### Neural Network (NN)
ニューラルネットワーク。
人間の脳の神経細胞のつながり方を模倣した計算モデルのことです。
#### Supervised Learning
教師あり学習。
学習データ（入力）とそれに対する正解（出力）を用いて、入力から出力を予測するモデルを学習する方法です。
#### Unsupervised Learning
教師なし学習。
学習データのみを用いて、データの構造やパターンを発見するモデルを学習する方法です。
#### Reinforcement Learning (RL)
強化学習。
エージェントが環境と相互作用しながら報酬を最大化するような行動を学習する方法です。
#### Natural Language Processing (NLP)
自然言語処理。人間が日常的に使う言語（自然言語）をコンピュータに理解させる技術のことです。
#### Computer Vision
コンピュータビジョン。画像や動画から有用な情報を抽出する技術のことを指します。
#### Generative Adversarial Networks (GANs)
敵対的生成ネットワーク。二つのニューラルネットワークが競争することで新たなデータを生成する技術です。
#### n-gram
テキストや音声などの連続するデータの列に対する統計的モデルの一種です。
n-gramでは、n個の項目の連続（つまり、"gram"）を一つの単位として扱います。
このnは任意の正の整数で、例えば2ならビグラム（bigram）、3ならトリグラム（trigram）と呼ばれます。

以下にn-gramの例を示します。"I love dogs"という文があるとき：

- ユニグラム（1-gram）は ["I", "love", "dogs"]
- ビグラム（2-gram）は ["I love", "love dogs"]
- トリグラム（3-gram）は ["I love dogs"]
  
n-gramは特に自然言語処理（NLP）の分野で広く使われています。
たとえば、機械翻訳や音声認識などのタスクで、単語やフレーズがどれくらいの頻度で一緒に出現するかを調べるために用いられます。
また、テキストの類似性を測るためにも利用されます。
n-gramは、単語の出現確率を計算するためのシンプルなモデルであり、文脈を考慮した情報を捉える能力があります。

ただし、n-gramにはいくつかの限界もあります。
特に、nが大きくなるとデータのスパース性（データ中に存在する可能性のあるn-gramのうち、実際に観測されるn-gramが非常に少ないという問題）が問題となることがあります。
また、n-gramは直前のn-1個の単語の情報しか考慮しないため、長い範囲の依存関係を捉えることができません。
### ニューラルネットワーク
#### フィードフォワードニューラルネットワーク（Feedforward Neural Networks, FNN）
最も基本的な形式のニューラルネットワークで、情報が入力層から隠れ層を経由して出力層へ一方向に流れます。
フィードフォワードニューラルネットワークでは、各層は前の層からの出力を入力として受け取ります。
#### 畳み込みニューラルネットワーク（Convolutional Neural Networks, CNN）
画像認識や画像処理タスクに広く使用されるニューラルネットワーク。
CNNは畳み込み層を持つことで特徴を抽出し、最終的に全結合層でクラス分類などを行います。
#### リカレントニューラルネットワーク（Recurrent Neural Networks, RNN）
時系列データや自然言語など、順序性が重要なデータを扱うためのニューラルネットワーク。
過去の情報を保持し、それを次のステップの予測に利用します。
#### 長短期記憶（Long Short-Term Memory, LSTM）
RNNの一種で、長期的な依存関係を学習する能力が向上しています。
LSTMは「ゲート」と呼ばれる構造を用いて、情報の流れを調節します。
#### 変分オートエンコーダ（Variational Autoencoders, VAE）
教師なし学習を行うニューラルネットワークで、入力データを特定の形の潜在空間にマッピングします。VAEは、生成モデルとしても使用されます。
#### 生成敵対ネットワーク（Generative Adversarial Networks, GAN）
GANは二つのネットワーク、生成ネットワークと識別ネットワークから成ります。
成ネットワークは、データの分布を学習し、新たなデータを生成します。
一方、識別ネットワークは、入力されたデータが本物（実際のデータ）か偽物（生成ネットワークが生成したデータ）かを識別します。
これら二つのネットワークが互いに競争しながら学習を進めます。
#### トランスフォーマーネットワーク（Transformer Networks）
トランスフォーマーネットワークは自然言語処理（NLP）タスクに広く使われています。
特に、Attentionメカニズムを用いて、入力の異なる部分に対するモデルの焦点を動的に変えることができます。
### ハイパーパラメータ
#### 学習率（Learning Rate）
学習率は、ニューラルネットワークが学習する速度を制御する重要なハイパーパラメータです。
学習率が高すぎると、モデルは最適な解を見つけることが難しくなり、学習率が低すぎると、学習に時間がかかりすぎて効率的でない可能性があります。
適切な学習率を見つけるには実験が必要であり、一般的には学習の進行とともに学習率を下げる方法（学習率のスケジューリング）がよく用いられます。
#### バッチサイズ（Batch Size）
バッチサイズは、一度にネットワークに供給されるサンプルの数を決定します。
バッチサイズが大きいほど、一度に処理するデータが多くなり、学習の速度が向上しますが、メモリ使用量が増えます。
また、バッチサイズが大きすぎると、モデルの学習が不安定になる可能性があります。
一方、バッチサイズが小さい場合、学習はより安定しますが、学習に必要な時間が長くなります。
#### エポック数（Number of Epochs）
エポック数は、訓練データ全体がネットワークを通過する回数を指します。
エポック数が多いほど、ネットワークは訓練データをより多く見ることができますが、過学習のリスクが高まります。
エポック数が少なすぎると、モデルは訓練データから十分な情報を学習できない可能性があります。一般的には、エポック数はモデルの性能が検証データセットで改善しなくなるまで増やします。
検証データのパフォーマンスが改善しなくなったときに学習を止めるというテクニックを早期停止といい、過学習を防ぐのに役立ちます。
#### 正則化パラメータ（Regularization Parameter）
正則化は過学習を防ぐための手法で、正則化パラメータはその正則化の強さを調節します。
正則化パラメータが大きいほど、モデルの重みを小さく保つことでモデルの複雑さを抑える効果が強まります。
しかし、パラメータが大きすぎるとモデルが学習データに適合しきれず、学習不足（underfitting）を起こす可能性があります。そのため、正則化パラメータの適切な値を見つけることが重要です。
#### 隠れ層の数（Number of Hidden Layers）および 隠れ層のユニット数（Number of Units in Hidden Layers）
ニューラルネットワークの隠れ層の数とその各層のユニット数（ニューロン数）は、モデルの複雑さと学習能力を決定します。
隠れ層やユニット数が多いほど、モデルは複雑な関数を学習する能力が高まります。
しかし、適切な数を超えて隠れ層やユニット数を増やすと、モデルは過学習を起こしやすくなり、学習データに対する精度は高くなりますが、新たなデータに対する精度は低下します。
#### 最適化アルゴリズム（Optimizer）
最適化アルゴリズムは、モデルのパラメータを更新し、訓練データに対する誤差を最小化するための手法です。
一般的な最適化アルゴリズムには、SGD（確率的勾配降下法）、Adam、RMSpropなどがあります。
各アルゴリズムは特定の問題に対して最適な場合もあるため、どのアルゴリズムを使用するかは問題によります。
また、最適化アルゴリズムは学習率と密接に関連しています。
学習率は大きすぎると学習が不安定になり、小さすぎると学習が遅くなる可能性があります。
#### ドロップアウト率（Dropout Rate）
ドロップアウトは、過学習を防ぐための手法の一つで、ランダムにノードを「ドロップアウト」（無効化）することで、モデルの汎化能力を向上させることができます。
ドロップアウト率は、無効化するノードの割合を指定します。
ドロップアウト率が高いと学習が遅くなる可能性がありますが、低すぎると効果が薄れる可能性があります。
#### 活性化関数（Activation Function
活性化関数は、ニューロンの出力を決定する関数で、非線形性をモデルに導入する役割があります。
一般的な活性化関数には、ReLU（Rectified Linear Unit）、sigmoid、tanh、Leaky ReLUなどがあります。
適切な活性化関数の選択は、タスクとネットワークの構造によります。
ReLUは一般的に最初の選択肢とされていますが、一部のニューロンが「死ぬ」（出力がほぼゼロになる）ことがあります。
#### 重み初期化方法（Weight Initialization Method）
深層学習モデルのパラメータ（重みとバイアス）の初期値を設定する方法です。
ネットワークの構造（特に活性化関数）と相性の良い初期化を選ぶことが重要です。例えば、ReLU活性化関数を使用している場合、He初期化が適しています。また、タスクの性質とデータの分布も考慮に入れるべきです。
### ハイパーパラメータ詳細
#### 最適化アルゴリズム
下記のアルゴリズムはすべて異なる特性と利点を持ち、その適用は特定のタスクとデータセットに依存します。
また、最適化アルゴリズムの選択は、他のハイパーパラメータ（例えば、学習率、ミニバッチサイズ、エポック数など）と密接に関連しています。
それぞれのアルゴリズムには特定の使用時の注意があります。
たとえば、AdamやAdafactorなどのより高度なアルゴリズムは、パラメータの初期化や学習率の設定に特に注意を要します。
また、モデルのサイズと訓練データの量によって、最適なアルゴリズムが変わることもあります。
このため、最適なアルゴリズムを選択するためには、異なるアルゴリズムとハイパーパラメータ設定を試すことが重要です
##### SGD (Stochastic Gradient Descent)
SGDは最も基本的な最適化アルゴリズムで、各ステップで訓練データの小さなサブセット（ミニバッチ）を使用して勾配を計算します。
SGDの主な問題は、全てのパラメータに対して同じ学習率を使用するため、一部のパラメータが他のパラメータよりも速く収束する可能性があることです。
また、SGDはしばしば局所最小値に捕らわれやすいです。
##### Momentum
MomentumはSGDの改良版で、過去の勾配を累積して、パラメータ更新に"運動量"を与えます。
これにより、学習は全体的によりスムーズになり、局所最小値を超えて全体の最小値に到達する可能性が高まります。
##### Adagrad
Adagradは、各パラメータに対して異なる学習率を設定するアルゴリズムです。
これにより、一部のパラメータが他のパラメータよりも速く収束する問題が解決します。しかし、Adagradの問題は、学習率が徐々に減少し、最終的には0に近づくため、訓練が停止する可能性があることです。
##### RMSProp
RMSPropはAdagradの問題を解決するためのアルゴリズムです。
RMSPropは、過去の全ての勾配ではなく、最近の一部の勾配のみを考慮します。
これにより、学習率が0に近づく問題を解消します。
##### Adam (Adaptive Moment Estimation)
AdamはMomentumとRMSPropのアイデアを組み合わせたアルゴリズムで、過去の勾配の一部を考慮しながら各パラメータに対して適応的な学習率を設定します。
Adamは多くのNLPタスクで最高の結果を提供し、そのためT5モデルの訓練において一般的に推奨されます。
##### Adafactor
Adafactorは、トランスフォーマーモデルの訓練のために設計された最適化アルゴリズムです。
Adafactorは、パラメータの2次モーメント（つまり、パラメータの平方の期待値）を近似的に保存し、それを使用して学習率を適応的に調整します。
Adafactorは、メモリ効率が非常に高いことが特徴であり、大規模なトランスフォーマーモデルの訓練に特に適しています。
#### 活性化関数
活性化関数の選択は、モデルの性能に大きな影響を及ぼす可能性があります。
各活性化関数は特定の特性と利点を持ち、その適用は特定のタスクとモデルアーキテクチャに依存します。
また、活性化関数の選択は、他のハイパーパラメータ（例えば、学習率、ミニバッチサイズ、エポック数など）と密接に関連しています。
それぞれの活性化関数には特定の使用時の注意があります。
たとえば、ReLUやLeaky ReLUは、一部のニューロンが「死ぬ」可能性があるため、その影響を最小限に抑えるための対策が必要です。
また、SigmoidやTanhは勾配消失問題を引き起こす可能性があります。
GELUは一般的に良好な結果をもたらしますが、計算コストが高い可能性があります。
このため、最適な活性化関数を選択するためには、異なる活性化関数とハイパーパラメータ設定を試すことが重要です。
##### ReLU (Rectified Linear Unit)
ReLUは最も広く使われる活性化関数の一つで、非線形性を導入することでモデルの表現力を高めます。
ReLUは入力が0より大きい場合はその入力をそのまま出力し、0以下の場合は0を出力します。
ReLUは計算が非常に効率的であり、勾配消失問題を緩和することができますが、訓練中に一部のニューロンが「死んで」勾配が0になる問題（死んだReLU問題）があります。
##### Leaky ReLU
Leaky ReLUはReLUの一種で、0以下の入力に対しても微小な勾配を持つことで、死んだReLU問題を緩和します。
##### Sigmoid: 
Sigmoid関数は0から1の間の値を出力するため、確率として解釈するのに便利です。
しかし、Sigmoid関数は出力の範囲が限定されており、極端な入力値では勾配がほぼ0になり、勾配消失問題を引き起こす可能性があります。
##### Tanh (Hyperbolic Tangent)
Tanh関数はSigmoid関数のようにS字形の曲線を描きますが、出力範囲が-1から1であるため、より広範な出力を許容します。
しかし、これもまた極端な入力値では勾配消失問題を引き起こす可能性があります。
##### GELU (Gaussian Error Linear Unit)
GELUはTransformerモデル、特にBERTやT5などのモデルでよく使用される活性化関数です。
GELUは入力の正負に応じて異なる勾配を持つため、モデルが複雑なパターンを学習するのを助けます。
#### 重み初期化方法
これらの初期化方法はそれぞれ異なる特性と利点を持ち、その適用は特定のタスクとモデルアーキテクチャに依存します。
また、初期化方法の選択は、他のハイパーパラメータ（例えば、学習率、ミニバッチサイズ、エポック数など）と密接に関連しています。
適切な重み初期化方法を選択することで、モデルの学習速度と最終的な性能を大幅に改善することができます。
それぞれの初期化方法には特定の使用時の注意があります。
たとえば、Zero Initializationは通常避けるべきであり、Random Initializationは適切なスケールが必要です。
Xavier/Glorot InitializationやHe Initializationは特定の活性化関数と一緒に使用することが推奨されます。
Orthogonal InitializationはRNNに特に有用ですが、全てのネットワーク構造で使用可能なわけではありません。
これらの初期化方法を適切に使用することで、訓練の収束を速め、モデルの性能を向上させることができます。
各初期化方法がどのように動作し、どのタイプのネットワークやタスクに最適かを理解することは、成功する深層学習モデルを設計するための重要なスキルです。
##### Zero Initialization
これは最も単純な初期化方法で、すべての重みをゼロに設定します。
しかし、これは通常は避けるべきであり、すべてのニューロンが同じ出力を生成し、学習がうまく進行しない可能性があります。
##### Random Initialization
重みを小さなランダムな値で初期化します。これにより、各ニューロンが独立して学習を開始できます。しかし、重みが非常に大きいか小さい場合、学習中に勾配消失または爆発する可能性があります。
##### Xavier/Glorot Initialization
XavierまたはGlorotの初期化方法は、重みをランダムに初期化するための一般的な方法で、各ニューロンの入力と出力の数に基づいて適切なスケールを計算します。
これは、特にSigmoidやTanhなどの活性化関数と一緒に使用すると効果的です。
##### He Initialization
He初期化はReLUとその派生形（Leaky ReLU、PReLUなど）の活性化関数と一緒に使用するための初期化方法です。
これはXavierの初期化方法と似ていますが、出力の数ではなく入力の数に基づいてスケールを計算します。
##### Orthogonal Initialization
Orthogonal初期化は、重み行列が正規直交行列であるように重みを初期化します。
これは、特に再帰型ニューラルネットワーク（RNN）で有用です。
#### 正則化パラメータ
下記の正則化パラメータとテクニックはそれぞれ異なる特性と利点を持ち、その適用は特定のタスクとモデルアーキテクチャに依存します。
また、正則化パラメータの選択と調整は、他のハイパーパラメータ（例えば、学習率、ミニバッチサイズ、エポック数など）と密接に関連しています。
適切な正則化パラメータを選択し、調整することで、モデルの学習速度と最終的な性能を大幅に改善することができます。
##### L1正則化
L1正則化は、重みの絶対値の合計（L1ノルム）に比例するコストを損失関数に追加します。
これにより、モデルの重みがスパース（つまり、多くの重みがゼロ）になる傾向があります。L1正則化パラメータ（通常λと表記）は、この正則化項の強度を制御します。
##### L2正則化
L2正則化（または重み減衰）は、重みの二乗の合計（L2ノルム）に比例するコストを損失関数に追加します。
これにより、モデルの重みが小さくなる傾向があります。
L2正則化パラメータ（通常λと表記）は、この正則化項の強度を制御します。
##### ドロップアウト
ドロップアウトは、訓練中にランダムにニューロンを「ドロップアウト」（つまり、一時的に無効化）することでモデルを正則化するテクニックです。
ドロップアウト率（通常pと表記）は、各訓練ステップでドロップアウトするニューロンの割合を制御します。
##### 早期停止
早期停止は、検証データのパフォーマンスが改善しなくなった時点で訓練を停止するテクニックです。これにより、モデルの過学習を防ぐことができます。
#### 損失関数
損失関数の選択は、タスク、モデル、データセットに依存します。
例えば、クラスの不均衡が存在する場合、クロスエントロピー損失はクラスの不均衡を増大させる可能性があるため、適切な手法（例えば、重み付け）を用いて調整する必要があります。
ラベルスムージングは、過学習を防ぐ効果があるとされていますが、スムージングの程度（つまり、真のラベルの確信度をどれだけ下げるか）はハイパーパラメータとして調整する必要があります。
損失関数はモデルの学習をどのようにガイドするかを決定します。
したがって、タスクの目的と一致する損失関数を選択することが重要です。
##### クロスエントロピー損失（Cross Entropy Loss）
クロスエントロピー損失は、分類問題やシーケンス生成問題において一般的に使用される損失関数です。
T5などのトランスフォーマーモデルでは、各タイムステップ（単語）での出力分布と、真の単語のワンホットエンコーディングとの間のクロスエントロピーを計算します。
注意点として、この損失関数は出力が確率分布（つまり、出力の合計が1になる）であることを前提としています。
##### ラベルスムージング（Label Smoothing）
ラベルスムージングは、クロスエントロピー損失に対する改良版です。
これは、モデルが一部のクラス（または単語）に対して過信するのを防ぐために、真のラベルの確信度を少し下げるというアイデアに基づいています。
具体的には、各ラベルの確率を少し"スムージング"し、真のラベルだけでなく他のラベルにも一部の確率を割り当てます。
#### 学習率スケジューリング
学習率スケジューリングは、モデルの学習過程で学習率をどのように変更するかを決定する方法で下記の注意すべき点があります。

- ハイパーパラメータの選択
    - 各スケジューリング戦略は異なるハイパーパラメータを必要とします（例えば、ステップデカイの場合はステップ数と減衰率、ウォームアップの場合はウォームアップのステップ数など）。
    これらのハイパーパラメータは訓練の性能に大きな影響を及ぼすため、適切に選択することが重要です。
- 学習率の初期値
    - 学習率スケジューリングは初期の学習率から始まります。この初期値は適切に選ばれるべきで、あまりに小さすぎると学習が遅くなり、大きすぎると訓練が不安定になる可能性があります。
    一般的には、初期値を0.01や0.001などの比較的大きな値から始め、そこからスケジューリング戦略に従って学習率を減少させることが推奨されます​。
- データとモデルの適応性
    - 最適な学習率スケジューリング戦略は、使用するデータセットやモデルにより異なる場合があります。
    そのため、特定のタスクやモデルに最適なスケジューリング戦略を見つけるためには、異なる戦略を試すことが重要です​​。
- 最適化アルゴリズムとの相互作用
    - 一部の最適化アルゴリズム（例えばAdam）は、内部的に学習率を調整する機能を持つため、これらの最適化アルゴリズムと学習率スケジューリング戦略との間には相互作用が存在します。
    このため、最適化アルゴリズムとスケジューリング戦略を同時に選択する際には注意が必要です​​。
##### ステップ減衰
あらかじめ定義されたステップ数（エポック数）ごとに学習率を一定の割合で減少させます。
この方法では、学習率の減少が急激に行われ、局所的な最適解から抜け出すための「衝撃」を与えることが期待されます。
##### 指数的減衰
訓練の進行と共に学習率を指数関数的に減少させます。
これは、訓練が進むにつれて学習率が必要となる更新量が減少するという直感に基づいています。
##### 逆時間減衰
学習率を訓練ステップの逆数の関数として減少させます。
これは、学習が進むにつれて更新の頻度を減らすというアイデアに基づいています。
##### 余弦退行（Cosine Annealing）
学習率を訓練の進行と共にコサイン関数に基づいて減少させます。
この手法では、学習率が初期と終末で低く、中間で高くなるため、訓練の初期と終末で探索を強化し、中間では局所的な最適解に収束することを促します。
##### ウォームアップ付きスケジューリング
訓練の初期段階では学習率を増加させ（ウォームアップ）、一定のステップ数後には他のスケジューリング戦略（例えば、ステップデカイやコサインアニーリング）に従って学習率を減少させます。
ウォームアップは、訓練の初期段階での不安定な振る舞いを防ぐのに役立ちます
### 評価指標
#### Accuracy (精度)
精度は、全体のデータセットに対してモデルが正しく予測したインスタンスの割合を示します。
具体的には、精度は「正しく予測されたインスタンス数」を「全体のインスタンス数」で割ったものです。
しかし、クラスの不均衡が存在する場合（つまり、一部のクラスのインスタンスが他のクラスのインスタンスよりもはるかに多い場合）、精度は誤解を招くことがあります。
なぜなら、多数派のクラスを正しく予測するだけで高い精度が得られるからです。
#### Precision (適合率)
適合率は、モデルが陽性と予測したインスタンスのうち、実際に陽性だったインスタンスの割合を示します。
適合率は「真陽性」（正しく陽性と予測されたもの）を「真陽性＋偽陽性」（陽性と予測されたものの全体）で割ったものです。
適合率は偽陽性（陰性を誤って陽性と予測したもの）の数が少ないことを重視します。
#### Recall (再現率)
再現率は、実際の陽性インスタンスのうち、モデルが陽性と予測したインスタンスの割合を示します。
再現率は「真陽性」を「真陽性＋偽陰性」（陽性と予測すべきものの全体）で割ったものです。
再現率は偽陰性（陽性を誤って陰性と予測したもの）の数が少ないことを重視します。
#### F1 Score
F1スコアは、適合率と再現率の調和平均です。
適合率と再現率はトレードオフの関係にあるため（一方を高めると他方が低下する傾向があるため）、これら二つのバランスを示す指標としてF1スコアがよく使用されます。
#### BLEU (Bilingual Evaluation Understudy)
BLEUスコアは機械翻訳の性能を評価するための指標で、生成された翻訳と参照翻訳（人間が作成した正確な翻訳）との一致度を測ります。
特に、n-gram（連続するn個の単語）の一致度を見ます。
しかし、BLEUスコアは語順の問題や、文脈や意味を正確に捉えることができないという問題があります。
#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
ROUGEスコアは、自動要約や機械翻訳の評価に使用される指標で、参照要約（または翻訳）と生成された要約（または翻訳）との間の一致度を計算します。
複数のバージョンがあり、例えばROUGE-Nはn-gramの一致度を、ROUGE-Lは最長共通部分列（LCS）の一致度を計算します。
#### METEOR (Metric for Evaluation of Translation with Explicit ORdering)
METEORは、機械翻訳の評価指標で、BLEUよりも高度な評価を提供します。
同義語や語形変化を認識し、語順も考慮します。また、精度と再現率の調和平均を計算します。
#### Perplexity
Perplexityは、言語モデルの性能を評価するための指標で、言語モデルが次に来る単語をどれだけうまく予測できるかを評価します。
低いパープレキシティは良いモデルを示します。しかし、パープレキシティは絶対的な指標ではなく、特定のタスクに対するモデルの性能を直接反映するわけではありません。
#### Word Error Rate (WER)
WERは、音声認識や機械翻訳の評価に使用され、生成されたテキストと参照テキストとの間の単語レベルでの差異を計算します。
WERは挿入、削除、置換などの操作を考慮に入れます。
#### Character Error Rate (CER)
CERは、生成されたテキストと参照テキストとの間の文字レベルでの差異を計算します。
主に手書き認識や音声認識の評価に使われます。WERと同様に、挿入、削除、置換などの操作を考慮に入れます。
### 考え中
- 尤度
- 演繹
- 加重平均
- ソフトマックス関数
- 重み
- バイアス
- 接続の強さ
- 勾配降下法