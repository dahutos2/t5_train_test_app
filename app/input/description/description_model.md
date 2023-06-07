## 学習済みモデル一覧
### Huggingface
<details>
<summary>クリックして展開します</summary>

#### T5ベースのモデルカード
説明画像001
#### モデルの詳細
##### モデルの説明
T5では、クラスラベルか入力のスパンしか出力できないBERTスタイルのモデルとは対照的に、すべてのNLPタスクを、入力と出力が常にテキスト文字列である統一されたテキスト-テキスト形式に再構築することを提案します。
このtext-to-textフレームワークにより、どのようなNLPタスクに対しても同じモデル、損失関数、ハイパーパラメータを使用することができます。
##### 言語 (NLP)
英語、フランス語、ルーマニア語、ドイツ語
##### 関連リンク
- [研究論文](https://jmlr.org/papers/volume21/20-074/20-074.pdf)
- [Google の T5 ブログ投稿](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)
- [GitHub リポジトリ](https://github.com/google-research/text-to-text-transfer-transformer)
- [Huggingface T5 ドキュメント](https://huggingface.co/docs/transformers/model_doc/t5)
#### 用途
text-to-Text フレームワークを使用すると、機械翻訳、文書要約、質問応答、分類タスク (感情分析など) を含む、あらゆる NLP タスクで同じモデル、損失関数、ハイパーパラメーターを使用できます。
数値そのものではなく、数値の文字列表現を予測するように T5 をトレーニングすることで、T5 を回帰タスクに適用することもできます。
#### トレーニングの詳細
##### トレーニングデータ
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
##### トレーニング手順
このペーパーでは、あらゆる言語の問題をテキストからテキストへの形式に変換する統一フレームワークを導入することにより、NLP の転移学習テクニックの展望を探ります。
私たちの体系的な研究では、数十の言語理解タスクに関する事前トレーニングの目的、アーキテクチャ、ラベルのないデータセット、転送アプローチ、その他の要素を比較しています。
#### t5-small
6000万個のパラメータを持つチェックポイントです。
#### t5-base
2億2000万個のパラメータを持つチェックポイントです。
#### t5-large
7億7000 万個のパラメータを持つチェックポイントです。
</details>
<p></p>

### Google
<details>
<summary>クリックして展開します</summary>

#### T5ベースのモデルカード
説明画像002
#### モデルの詳細
##### モデルの説明
すでに T5 を知っている場合は、FLAN-T5 の方がすべてにおいて優れています。
同じ数のパラメーターについて、これらのモデルは、より多くの言語をカバーする 1,000 以上の追加タスクで微調整されています。
Flan-PaLM 540B は、5 ショット MMLU で 75.2% など、いくつかのベンチマークで最先端のパフォーマンスを達成します。
また、PaLM 62B などのはるかに大型のモデルと比較しても強力な数ショット性能を実現する Flan-T5 チェックポイント 1 も公開しています。
全体として、命令の微調整は、事前トレーニングされた言語モデルのパフォーマンスと使いやすさを向上させるための一般的な方法です。
##### 言語 (NLP)
英語、スペイン語、日本語、ペルシア語、ヒンディー語、フランス語、中国語、ベンガル語、グジャラート語、ドイツ語、テルグ語、イタリア語、アラビア語、ポーランド語、タミル語、マラーティー語、マラヤーラム語、オリヤー語、パンジャブ語、ポルトガル語、ウルドゥー語、ガリシア語、ヘブライ語、韓国語、カタルーニャ語、タイ語、オランダ語、インドネシア語、ベトナム語、ブルガリア語、フィリピン語、中央クメール語、ラオス語、トルコ語、ロシア語、クロアチア語、スウェーデン語、ヨルバ語、クルド語、ビルマ語、マレー語、チェコ語、フィンランド語、ソマリ語、タガログ語、スワヒリ語、シンハラ語、カンナダ語、チワン語、イボ語、コーサ語、ルーマニア語、ハイチ語、エストニア語、スロバキア語、リトアニア語、ギリシャ語、ネパール語、アッサム語、ノルウェー語
##### 関連リンク
- [研究論文](https://arxiv.org/pdf/2210.11416.pdf)
- [GitHub リポジトリ](https://github.com/google-research/t5x)
- [Huggingface T5 ドキュメント](https://huggingface.co/docs/transformers/model_doc/t5)
#### 使用法
モデルを使用する方法については、以下のサンプルスクリプトを参照してください。
##### Pytorch モデルの使用
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

##### さまざまな精度を使用して GPU でモデルを実行する
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

#### 用途
主な用途は、次のような言語モデルの研究です。
ゼロショット NLP タスクと、推論や質問応答などのコンテキスト内での少数ショット学習 NLP タスクの研究。
公平性と安全性の研究を進め、現在の大規模な言語モデルの限界を理解する。
#### トレーニングの詳細
##### トレーニングデータ
モデルは、以下の表で説明されているタスクを含むタスクの混合でトレーニングされました 。
説明画像003
##### トレーニング手順
これらのモデルは、事前トレーニングされた T5 (Raffel et al., 2020) に基づいており、ゼロショットおよび少数ショットのパフォーマンスを向上させるための命令で微調整されています。
T5 モデル サイズごとに 1 つの微調整された Flan モデルがあります。
#### input
##### Translation(翻訳)
Translate to German:  My name is Arthur
##### Question Answering(質問回答)
Please answer to the following question. Who is going to be the next Ballon d'or?(
##### Logical reasoning(論理的な推論)
Q: Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering.
##### Scientific knowledge(科学的知識)
Please answer the following question. What is the boiling point of Nitrogen?
##### Yes/no question(はい/いいえ の質問)
Answer the following yes/no question. Can you write a whole Haiku in a single tweet?
##### Reasoning task(推論タスク)
Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?
##### Boolean Expressions(ブール式)
Q: ( False or not False or False ) is? A: Let's think step by step
##### Math reasoning(数学的な推論)
The square root of x is the cube root of y. What is y to the power of 2, if x = 4?
##### Premise and hypothesis(前提・仮説)
Premise:  At my age you will probably have learnt one lesson. Hypothesis:  It's not certain how many lessons you'll learn by your thirties. Does the premise entail the hypothesis? 
#### google/flan-t5-small
#### google/flan-t5-base
#### google/flan-t5-large
#### google/flan-t5-xl
#### google/flan-t5-xxl
</details>
<p></p>

### Retrieva-jp
<details>
<summary>クリックして展開します</summary>

#### モデルの詳細
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
#### モデルの説明
言語 (NLP):日本語
#### トレーニングの詳細
このモデルのトレーニングには[T5X](https://github.com/google-research/t5x) を使用しており、Huggingface トランスフォーマー形式に変換されています。
#### トレーニングデータ
- 多言語C4(mC4/ja)の日本語部分。
- 日本語ウィキペディア(20220920)。
#### 前処理
- ひらがなを 1 文字も使用していない文書を削除します。
これにより、英語のみのドキュメントと中国語のドキュメントが削除されます。
- URL のトップレベル ドメインを使用してアフィリエイト サイトを削除するホワイトリスト スタイルのフィルタリング。
#### トレーニングのハイパーパラメータ
- dropout rate: 0.0
- バッチサイズ: 256
- fp32
- 入力長さ: 512
- 出力長さ: 114
- それ以外の場合は、以下を含む[T5X]( https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/)のデフォルト値に従います。
    - オプティマイザー: Adafactor
    - 基本学習率: 1.0
    - ウォームアップステップ: 10000
#### 関連記事
- [日本語T5モデルの公開](https://note.com/retrieva/n/n7b4186dc5ada)
#### retrieva-jp/t5-small-short
- 589824 ステップをトレーニングしました。
- サイズ: 小規模 (約 7,700 万パラメータ)
#### retrieva-jp/t5-small-medium
- 1048576 ステップをトレーニングしました。
- サイズ: 小規模 (約 7,700 万パラメータ)
#### retrieva-jp/t5-small-long
- 2097152 ステップをトレーニングしました。
- サイズ: 小規模 (約 7,700 万パラメータ)
#### retrieva-jp/t5-base-short
- 524288 ステップをトレーニングしました。
- サイズ: 基本(~2 億 2,000 万パラメータ)
#### retrieva-jp/t5-base-medium
- 1048576 ステップをトレーニングしました。
- サイズ: 基本(~2 億 2,000 万パラメータ)
#### retrieva-jp/t5-base-long
- 2097152 ステップをトレーニングしました。
- サイズ: 基本(~2 億 2,000 万パラメータ)
#### retrieva-jp/t5-large-short
- 524288 ステップをトレーニングしました。
- サイズ: 大(~7 億 7,000 万のパラメータ)
#### retrieva-jp/t5-large-medium
- 1008824 ステップをトレーニングしました。
- サイズ: 大(~7 億 7,000 万のパラメータ)
#### retrieva-jp/t5-large-long
- 2097152 ステップをトレーニングしました。
- サイズ: 大(~7 億 7,000 万のパラメータ)
#### retrieva-jp/t5-xl
- 524288 ステップをトレーニングしました。
- サイズ: XL(~30 億パラメータ)
</details>
<p></p>

### Sonoisa
<details>
<summary>クリックして展開します</summary>

#### sonoisa/t5-base-japanese-mC4-Wikipedia
日本語T5事前学習済みモデル
次の日本語コーパス（約890GB）を用いて事前学習を行ったT5 (Text-to-Text Transfer Transformer) モデルです。

- [Wikipedia](https://ja.wikipedia.org/wiki/%E3%83%A1%E3%82%A4%E3%83%B3%E3%83%9A%E3%83%BC%E3%82%B8)の日本語ダンプデータ (2020年7月6日時点のもの)
- [mC4](https://github.com/allenai/allennlp/discussions/5056)の日本語コーパス（正確にはc4/multilingualのjaスプリット）

このモデルは事前学習のみを行なったものであり、特定のタスクに利用するにはファインチューニングする必要があります。
本モデルにも、大規模コーパスを用いた言語モデルにつきまとう、学習データの内容の偏りに由来する偏った（倫理的ではなかったり、有害だったり、バイアスがあったりする）出力結果になる問題が潜在的にあります。 
この問題が発生しうることを想定した上で、被害が発生しない用途にのみ利用するよう気をつけてください。

SentencePieceトークナイザーの学習には上記Wikipediaの全データを用いました。

- [転移学習のサンプルコード](https://github.com/sonoisa/t5-japanese)
#### sonoisa/t5-base-english-japanese
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
#### sonoisa/sentence-t5-base-ja-mean-tokens
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

#### sonoisa/t5-base-japanese-adapt
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

#### sonoisa/t5-base-japanese-question-generation
[回答と回答が出てくるパラグラフを与えると質問文を生成するモデル](https://github.com/sonoisa/deep-question-generation)
##### 本モデルの作成ステップ概要
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
#### sonoisa/t5-base-japanese-title-generation
[記事本文からタイトルを生成するモデル](https://qiita.com/sonoisa/items/a9af64ff641f0bbfed44)
#### sonoisa/t5-qiita-title-generation
[記事本文からタイトルを生成するモデル](https://qiita.com/sonoisa/items/30876467ad5a8a81821f)
#### sonoisa/t5-base-japanese-article-generation
[タイトルから記事本文を生成するモデル](https://qiita.com/sonoisa/items/a9af64ff641f0bbfed44)
</details>
<p></p>