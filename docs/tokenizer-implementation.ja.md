# トークナイザー実装 処理フローメモ

## 概要

`nanochat/tokenizer.py` は **GPT-4 スタイルの BPE(Byte Pair Encoding)トークナイザー**の実装です。最大の特徴は「**学習は Rust 実装の `rustbpe`、推論は `tiktoken`**」という二段構えになっている点で、これはファイル冒頭の docstring にそのまま書かれています。

```python
"""
BPE Tokenizer in the style of GPT-4: train with rustbpe, inference with tiktoken.
"""
```

中心となるのは `RustBPETokenizer` クラスで、`tiktoken` の `Encoding` オブジェクトを薄くラップした構造になっています。役割は大きく 3 つです。

- 文字列 → トークン ID(`encode`)、トークン ID → 文字列(`decode`)の相互変換。
- Chat 会話(system / user / assistant のメッセージ列)を、学習に使えるトークン列 + マスクへ変換する `render_conversation`。
- 学習・保存・読み込み(`train_from_iterator` / `save` / `from_directory` / `from_pretrained`)。

語彙は「通常トークン + 9 個の特殊トークン」で構成されます。特殊トークンは `<|bos|>`(ドキュメント区切り)と、会話 / ツール呼び出しのレンダリング用の 8 個(`<|user_start|>`〜`<|output_end|>`)です。

## 主な構成

### モジュールレベル定義

#### `SPECIAL_TOKENS`

9 個の特殊トークンを定義したリストです。先頭の `<|bos|>` は全ドキュメントの先頭に付くドキュメント区切りで、残り 8 個はファインチューニング時に会話をレンダリングする際にのみ使われます(user / assistant / python / output の start・end ペア)。

```python
SPECIAL_TOKENS = [
    # 全ドキュメントの先頭に付く BOS(Beginning of Sequence)トークン。ドキュメントの区切りを表す
    "<|bos|>",
    # 以下はファインチューニング時に会話(Conversation)をトークン ID 列へレンダリングするためだけに使う
    "<|user_start|>", # user メッセージ
    "<|user_end|>",
    "<|assistant_start|>", # assistant メッセージ
    "<|assistant_end|>",
    "<|python_start|>", # assistant が Python REPL ツールを呼び出す
    "<|python_end|>",
    "<|output_start|>", # Python REPL が assistant に返す出力
    "<|output_end|>",
]
```

#### `SPLIT_PATTERN`

トークン化前にテキストを分割する正規表現です。GPT-4 の分割パターンをベースにしていますが、**数字部分だけ `\p{N}{1,2}`(GPT-4 は `{1,3}`)に変更**しています。コメントにあるとおり、小さい語彙サイズで数字にトークンを浪費しないためで、32K 語彙では 2 桁が最適だと検証されています。

```python
# NOTE: この分割パターンは GPT-4 と異なり、\p{N}{1,3} ではなく \p{N}{1,2} を使う。
# 小さい語彙サイズで数字にトークンを「浪費」しすぎないためにこうしている。
# 語彙サイズ 32K では 2 が最適だと確認済み。1 は少し劣り、3 はさらに劣った。
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

### `RustBPETokenizer` クラスの主なメソッド

#### `__init__(self, enc, bos_token)`

`tiktoken` の `Encoding` オブジェクト `enc` を保持し、BOS トークンの ID をキャッシュします。

```python
def __init__(self, enc, bos_token):
    self.enc = enc
    self.bos_token_id = self.encode_special(bos_token) # BOS トークンの ID をキャッシュ
```

#### `train_from_iterator(cls, text_iterator, vocab_size)`

学習のエントリポイントです。処理は 2 段階に分かれます。

1. `rustbpe.Tokenizer()` を使い、`SPLIT_PATTERN` で `vocab_size - len(SPECIAL_TOKENS)` の語彙を学習する(特殊トークンはここでは学習しない。最低 256 が必要)。
2. 学習済みの `mergeable_ranks` と `pattern` を取り出し、特殊トークンを末尾のオフセット ID に割り当てて `tiktoken.Encoding` を構築する。

```python
@classmethod
def train_from_iterator(cls, text_iterator, vocab_size):
    # 1) rustbpe で学習する
    tokenizer = rustbpe.Tokenizer()
    # 特殊トークンは後で __init__ 側で挿入するので、ここでは学習しない
    vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
    assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
    tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
    # 2) 推論用に、対応する tiktoken の Encoding を構築する
    pattern = tokenizer.get_pattern()
    mergeable_ranks_list = tokenizer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    tokens_offset = len(mergeable_ranks)
    # 特殊トークンは通常トークンの末尾に続く ID(オフセット)を割り当てる
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks, # dict[bytes, int](トークンのバイト列 -> マージ優先順位)
        special_tokens=special_tokens, # dict[str, int](特殊トークン名 -> トークン ID)
    )
    return cls(enc, "<|bos|>")
```

#### `from_directory` / `from_pretrained`

`from_directory` は pickle 保存された `Encoding` を復元します。`from_pretrained` は `tiktoken` の既存エンコーディングを名前で読み込むもので、この場合 BOS トークンは `<|endoftext|>` になります。コメントには「`<|endoftext|>` という名前は紛らわしいが、実際にはドキュメントの**先頭**に prepend されて使われる」という歴史的経緯が書かれています。

```python
@classmethod
def from_directory(cls, tokenizer_dir):
    pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(pickle_path, "rb") as f:
        enc = pickle.load(f) # pickle から Encoding を復元
    return cls(enc, "<|bos|>")

@classmethod
def from_pretrained(cls, tiktoken_name):
    enc = tiktoken.get_encoding(tiktoken_name)
    # tiktoken はドキュメント区切りの特殊トークンを "<|endoftext|>" と呼ぶ。
    # 名前は紛らわしいが、このトークンはほぼ常にドキュメントの「先頭」に prepend される。
    # なので nanochat では常に "<|bos|>"(beginning of sequence)と呼ぶが、歴史的には "<|endoftext|>" と呼ばれることが多い。
    return cls(enc, "<|endoftext|>")
```

#### `encode(self, text, prepend, append, num_threads)`

文字列またはその配列をトークン ID に変換します。`str` は `encode_ordinary`、`list[str]` は `encode_ordinary_batch`(マルチスレッド)で処理し、`prepend` / `append` で BOS などを前後に付与します。

```python
def encode(self, text, prepend=None, append=None, num_threads=8):
    # text は文字列でも、文字列のリストでもよい

    if prepend is not None:
        prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
    if append is not None:
        append_id = append if isinstance(append, int) else self.encode_special(append)

    if isinstance(text, str):
        ids = self.enc.encode_ordinary(text) # 単一文字列はそのままエンコード
        if prepend is not None:
            ids.insert(0, prepend_id) # 先頭に prepend(例: BOS)
        if append is not None:
            ids.append(append_id) # 末尾に append
    elif isinstance(text, list):
        ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads) # リストはマルチスレッドで一括エンコード
        if prepend is not None:
            for ids_row in ids:
                ids_row.insert(0, prepend_id)
        if append is not None:
            for ids_row in ids:
                ids_row.append(append_id)
    else:
        raise ValueError(f"Invalid input type: {type(text)}")

    return ids
```

#### その他のメソッド

- `decode(self, ids)`: トークン ID 列を文字列へ戻す(`self.enc.decode` の薄いラッパー)。
- `encode_special(self, text)`: 特殊トークン 1 個を ID へ変換する。`lru_cache(maxsize=32)` 付きで、繰り返し呼び出しを高速化している。
- `get_bos_token_id(self)`: キャッシュ済みの `bos_token_id` を返す。
- `get_vocab_size(self)`: `self.enc.n_vocab`(特殊トークンを含む語彙数)を返す。
- `save(self, tokenizer_dir)`: `enc` を `tokenizer.pkl` として pickle 保存する。

```python
def save(self, tokenizer_dir):
    # Encoding オブジェクトをディスクに保存する
    os.makedirs(tokenizer_dir, exist_ok=True)
    pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(self.enc, f) # enc を pickle で tokenizer.pkl に保存
    print(f"Saved tokenizer encoding to {pickle_path}")
```

### 会話レンダリング `render_conversation`

このトークナイザーで最も重要なロジックが `render_conversation` です。1 つの Chat 会話を受け取り、`ids`(トークン ID 列)と、同じ長さの `mask`(**Assistant が学習対象とするトークンだけ 1**、それ以外は 0)を返します。

戻り値と、トークンとマスクを組で追加していくヘルパー関数は次のとおりです。

```python
def render_conversation(self, conversation, max_tokens=2048):
    """
    1 つの Chat 会話(ここでは "doc" / "document" と呼ぶ)をトークン化する。
    戻り値:
    - ids: この会話をレンダリングしたトークン ID のリスト
    - mask: ids と同じ長さ。Assistant が学習対象とするトークンで 1 になる
    """
    # 返す ids・mask と、それらを組み立てるためのヘルパー関数
    ids, mask = [], []
    def add_tokens(token_ids, mask_val):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids)) # 追加した分だけ同じ mask 値を並べる
```

先頭が system メッセージの場合は、次の user メッセージへ結合する前処理を行います(system 単独では扱わない)。

```python
    # 先頭が system メッセージのことがある...
    # => その内容を 2 番目(user)メッセージに結合してしまう
    if conversation["messages"][0]["role"] == "system":
        # ここで会話に少し「手術」が必要になる...
        conversation = copy.deepcopy(conversation) # 元データを変更しないようディープコピー
        messages = conversation["messages"]
        assert messages[1]["role"] == "user", "System message must be followed by a user message"
        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = messages[1:] # system を取り除いた残りを使う
    else:
        messages = conversation["messages"]
    assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"
```

会話は BOS を先頭に付けたうえで、メッセージを 1 件ずつ処理します。`i % 2` で **user / assistant が交互である前提**を assert しています。

```python
    # ここで会話をトークン化していく
    add_tokens(bos, 0)
    for i, message in enumerate(messages):

        # 想定(user/assistant が交互)を崩す footgun を防ぐためのサニティチェック
        must_be_from = "user" if i % 2 == 0 else "assistant"
        assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

        # content は単純な文字列でも、パーツのリスト(ツール呼び出しなど)でもよい
        content = message["content"]
```

mask 付与規則が肝です。**user メッセージは全て mask=0**。**assistant** は、文字列なら mask=1、パーツのリストなら種類ごとに `text`=1、`python`(`<|python_start|>`〜`<|python_end|>` で囲む)=1、`python_output`(`<|output_start|>`〜`<|output_end|>`)=0 とします。`python_output` が mask=0 なのは、コメントにあるとおり「テスト時に Python 側から返ってくるトークンなので教師しない」ためです。

```python
        if message["role"] == "user":
            assert isinstance(content, str), "User messages are simply expected to be strings"
            value_ids = self.encode(content)
            add_tokens(user_start, 0) # user は全て mask=0(学習しない)
            add_tokens(value_ids, 0)
            add_tokens(user_end, 0)
        elif message["role"] == "assistant":
            add_tokens(assistant_start, 0) # 開始トークンは学習しない
            if isinstance(content, str):
                # 単純な文字列 => そのままトークンを追加(mask=1 で学習)
                value_ids = self.encode(content)
                add_tokens(value_ids, 1)
            elif isinstance(content, list):
                for part in content:
                    value_ids = self.encode(part["text"])
                    if part["type"] == "text":
                        # テキストパーツ => そのままトークン追加(学習する)
                        add_tokens(value_ids, 1)
                    elif part["type"] == "python":
                        # Python ツール呼び出し => <|python_start|>〜<|python_end|> で囲んで追加(学習する)
                        add_tokens(python_start, 1)
                        add_tokens(value_ids, 1)
                        add_tokens(python_end, 1)
                    elif part["type"] == "python_output":
                        # Python の出力 => <|output_start|>〜<|output_end|> で囲む
                        # これらは Python がテスト時に返すトークンなので、いずれも学習対象にしない(mask=0)
                        add_tokens(output_start, 0)
                        add_tokens(value_ids, 0)
                        add_tokens(output_end, 0)
                    else:
                        raise ValueError(f"Unknown part type: {part['type']}")
            else:
                raise ValueError(f"Unknown content type: {type(content)}")
            add_tokens(assistant_end, 1) # 終了トークンは学習する
```

最後に `max_tokens`(既定 2048)で truncate します。長い会話は末尾が切られます。

```python
    # OOM を防ぐため、最大 max_tokens トークンに truncate する
    ids = ids[:max_tokens]
    mask = mask[:max_tokens]
    return ids, mask
```

関連して、RL(強化学習)用の `render_for_completion` もあります。こちらは最後の Assistant メッセージを pop したうえで会話をレンダリングし、末尾に `<|assistant_start|>` を付けて Assistant に completion を促します。SFT と違い mask は返しません。

```python
def render_for_completion(self, conversation):
    """
    強化学習(RL)時に使う。会話を Assistant の completion 直前の状態でレンダリングする。
    Chat SFT と違い、mask は返さない。
    """
    # 「手術」が必要: 最後(Assistant)のメッセージを取り除く
    conversation = copy.deepcopy(conversation)
    messages = conversation["messages"]
    assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
    messages.pop() # 最後(Assistant)のメッセージをインプレースで削除

    # そのうえで会話をトークン化する
    ids, mask = self.render_conversation(conversation)

    # 最後に、Assistant に completion を促すため <|assistant_start|> を付ける
    assistant_start = self.encode_special("<|assistant_start|>")
    ids.append(assistant_start)
    return ids
```

## 全体の処理フロー

トークナイザー学習のフローは次のとおりです。

```text
scripts/tok_train.py を起動
  ↓
parquets_iter_batched(split="train") でドキュメントをバッチ取得
  ↓
text_iterator() で各ドキュメントを doc_cap 文字にクロップ・max_chars で打ち切り
  ↓
RustBPETokenizer.train_from_iterator(text_iter, vocab_size)
  ↓ (1) rustbpe で BPE マージを学習(vocab_size - 特殊トークン数)
  ↓ (2) tiktoken.Encoding を構築(特殊トークンを末尾 ID に付与)
  ↓
tokenizer.save() で tokenizer.pkl を保存
  ↓
往復サニティチェック(encode → decode で元文字列と一致を assert)
  ↓
token_bytes.pt を生成・保存(bits per byte 評価用)
```

推論時は、学習済みトークナイザーを読み込んで文字列とトークンを相互変換するだけの単純なフローです。

```text
文字列
  ↓ tokenizer.encode(text, prepend=<|bos|> など)
トークン ID 列
  ↓ モデル(GPT)
出力トークン ID 列
  ↓ tokenizer.decode(ids)
文字列
```

会話レンダリング(SFT / チャット)のフローは次のとおりです。

```text
conversation(system / user / assistant のメッセージ列)
  ↓ render_conversation(conversation, max_tokens=2048)
ids(トークン ID 列) + mask(Assistant の学習対象だけ 1)
  ↓
mask=1 のトークンにのみ損失をかけて学習
```

## CLI の流れ

学習のエントリスクリプトは `scripts/tok_train.py` です。主なコマンドライン引数は次のとおりです。

- `--max-chars`: 学習に使う最大文字数(既定 2B = `2_000_000_000`)。
- `--doc-cap`: 1 ドキュメントあたりの最大文字数(既定 10,000)。
- `--vocab-size`: 語彙サイズ(既定 32768 = 2^15)。

テキストは `text_iterator()` が供給します。処理は 3 ステップで、(1) バッチを平坦化して 1 本のイテレータにし、(2) 各ドキュメントを `doc_cap` 文字にクロップし、(3) `max_chars` に達したら打ち切ります。

```python
def text_iterator():
    """
    1) バッチを平坦化して 1 本のイテレータにする
    2) 各ドキュメントを args.doc_cap 文字にクロップする
    3) args.max_chars 文字を見たら打ち切る
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap] # doc_cap 文字にクロップ
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars: # max_chars に達したら終了
                return
```

学習・保存・往復サニティチェックの後、`token_bytes.pt` を生成します。これは「トークン ID → そのトークンのバイト数」の対応表で、**語彙サイズに依存しない bits per byte 評価**に使います。特殊トークンは 0、それ以外は `decode_single_token_bytes` の生バイト長を格納します(文字列へ decode するとバイト単体では不正な UTF-8 が壊れるため、生バイトを使う)。

```python
# もう一つ: bits per byte を効率的に評価するために、トークン id -> そのトークンのバイト数
# の対応をキャッシュしておく。通常の平均損失と違い、これによりトークナイザーの語彙サイズに
# 依存しない損失を報告できる。検証セットの bits per byte が我々が重視する主要指標の一つになる。
vocab_size = tokenizer.get_vocab_size()
special_ids = set(tokenizer.encode_special(s) for s in tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    if token_id in special_ids:
        token_bytes.append(0) # 特殊トークンはカウントしない
    else:
        # トークンの生バイトを使う: 先に文字列へ decode すると、単体では正しい UTF-8 でない
        # トークン(例: 0x80 以上の生バイト)が壊れてしまう
        num_bytes = len(tokenizer.decode_single_token_bytes(token_id))
        token_bytes.append(num_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
```

## データパイプラインとの関係

学習済みトークナイザーは各工程で使われます。

- **保存先**: `get_base_dir()/tokenizer/` に `tokenizer.pkl` と `token_bytes.pt` が置かれ、`get_tokenizer()` / `get_token_bytes()` が読み込みます。

```python
def get_tokenizer():
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir) # tokenizer.pkl を読み込む

def get_token_bytes(device="cpu"):
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
```

- **事前学習**: `nanochat/dataloader.py` の `refill_buffer()` が `tokenizer.encode(doc_batch, prepend=bos_token, ...)` を呼び、各ドキュメントを BOS 前置でトークナイズしてバッファに積みます。その後 BOS 整列のベストフィット詰め込みで `(inputs, targets)` バッチを組み立てます。

```python
def refill_buffer():
    nonlocal pq_idx, rg_idx, epoch
    doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
    # 各ドキュメントを BOS 前置でトークナイズ(マルチスレッド)
    token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
    for tokens in token_lists:
        doc_buffer.append(tokens) # バッファに積む
```

- **SFT / チャット**: `render_conversation` の `ids` / `mask` を用い、Assistant の出力トークンだけを教師します。
- **RL**: `render_for_completion` を使って Assistant の completion 直前の状態を作ります。
- **bits per byte 評価**: `token_bytes`(`get_token_bytes()`)を使い、語彙サイズに依存しない損失を評価します(`loss_eval.py` 系)。

## 注意点

- 学習(`rustbpe`)と推論(`tiktoken`)でライブラリが分かれる二段構えである。
- 特殊トークンは学習対象外で、`train_from_iterator` 内で末尾 ID に後付けされる。実効的な学習語彙は `vocab_size - len(SPECIAL_TOKENS)` になる。
- `SPLIT_PATTERN` は数字を最大 2 桁に制限しており、GPT-4(最大 3 桁)と異なる。語彙サイズを変える場合はこの影響を考慮する。
- `render_conversation` は user / assistant が交互である前提の assert を持ち、崩れると例外になる。
- `python_output` パーツと user / 特殊トークンは mask=0 で、学習の勾配はアシスタント生成トークンにのみ効く。
- 出力は `max_tokens`(既定 2048)で truncate されるため、長い会話は末尾が切れる。
- `from_pretrained` 経由では BOS が `<|endoftext|>` になる(名前は紛らわしいが、役割は文頭区切り)。
