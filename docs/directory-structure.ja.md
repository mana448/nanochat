# ディレクトリ構成ガイド

このドキュメントは、nanochat(単一 GPU ノードで LLM の全工程=トークナイズ・事前学習・微調整・評価・推論 を回すための最小限の実験用ハーネス)のディレクトリ構成を説明する資料です。

初めて nanochat に触る人が「どこに何があるか」「どの作業でどこを見るか」を素早く把握できることを目的としています。

## 全体像

nanochat は大きく **3 つの領域**(コアライブラリ / 実行エントリ / タスク定義)と、それらを支える **補助**(オーケストレーション・テスト・開発資料・設定)で構成されています。

- `nanochat/`: コアライブラリ。GPT モデル本体、トークナイザー、データローダー、推論エンジン、オプティマイザ、評価ロジックなど、全工程の実装が集約されています。
- `scripts/`: 各工程を起動する実行エントリ(トークナイザー学習/評価、事前学習/評価、SFT/RL/チャット評価/CLI 対話、推論ベンチ)。
- `tasks/`: 評価・微調整用のデータセット/タスク定義(ARC, GSM8K, HumanEval, MMLU, SmolTalk と、それらを束ねる TaskMixture / TaskSequence)。
- 補助:
  - `runs/`: 工程全体を通すシェルスクリプト。
  - `tests/`: pytest によるテスト。
  - `dev/`: リーダーボード・ログ・分析ノートブック・データ再パッケージ参照実装。
  - ルート直下の設定ファイル群(`pyproject.toml`, `uv.lock`, `.python-version`, `LICENSE`, `README.md`)。

## 各ディレクトリの関係性

データと処理の流れは次のようになっています。

```
                        ┌────────────────────────────────────────────┐
                        │                 nanochat/                   │
                        │  (コアライブラリ: 全工程の実装が集約)         │
                        │                                             │
   parquet データ ──▶  dataset.py ──▶ dataloader.py ──▶ gpt.py       │
   (必要時DL)          (取得/読込)     (トークナイズ供給)  (モデル本体)   │
                        │  tokenizer.py  engine.py  optim.py          │
                        │  core_eval.py  loss_eval.py  ...            │
                        └───────────────▲──────────────┬─────────────┘
                                        │ import       │ 参照
                                        │              ▼
              ┌─────────────────────────┴──┐    ┌──────────────┐
              │        scripts/            │    │    tasks/    │
              │  (各工程の実行エントリ)      │◀───│ (評価/学習    │
              │  tok_train / base_train /  │    │  タスク定義)   │
              │  chat_sft / chat_eval ...  │    └──────────────┘
              └─────────────▲──────────────┘
                            │ 順に呼び出す
              ┌─────────────┴──────────────┐          ┌──────────────┐
              │          runs/             │          │     dev/     │
              │  speedrun.sh などが        │          │ (分析・ログ・  │
              │  パイプライン全体を通す      │          │  判断材料)     │
              └────────────────────────────┘          └──────────────┘
```

要点は以下の通りです。

- 事前学習データは `nanochat/dataset.py` が parquet を必要に応じてダウンロードし、`nanochat/dataloader.py` がトークナイズしながら分散データローダとして供給します。
- `scripts/` の各スクリプトが `nanochat/` のモジュール(`gpt.py`, `tokenizer.py`, `engine.py`, `optim.py`, 各 eval など)を組み合わせて 1 工程を実行します。
- `runs/*.sh`(特に `speedrun.sh`)が `scripts/` を順に呼び出し、トークナイザー学習 → 事前学習 → 微調整 → 評価 → 対話までのパイプライン全体を通します。
- `tasks/` は `scripts/chat_sft.py` / `scripts/chat_eval.py` / `scripts/base_eval.py` などが参照する評価・学習タスクを定義します。
- `dev/` はドキュメント・分析資料で、実装の判断材料(リーダーボードの解釈、ログ、スケーリング分析)を提供します。

## ルート直下

```
.
├── nanochat/         # コアライブラリ(モデル・トークナイザー・データ・推論・最適化・評価)
├── scripts/          # 各工程の実行エントリスクリプト
├── tasks/            # 評価・微調整用タスク/データセット定義
├── runs/             # 工程全体を通すシェルスクリプト
├── tests/            # pytest によるテスト
├── dev/              # リーダーボード・ログ・分析ノートブック・参照実装
├── pyproject.toml    # 依存関係とプロジェクト設定(uv 管理)
├── uv.lock           # uv のロックファイル
├── .python-version   # Python バージョン指定
├── LICENSE           # ライセンス
└── README.md         # プロジェクト概要
```

セットアップは `uv sync --extra gpu`(CUDA 環境)または `uv sync --extra cpu`(CPU/MPS 環境)で行います(詳細は README を参照)。

## `nanochat/`

```
nanochat/
├── gpt.py                # GPT Transformer 本体(nn.Module)
├── tokenizer.py          # GPT-4 スタイルの BPE トークナイザー(学習=rustbpe / 推論=tiktoken)
├── dataloader.py         # トークナイズを行う分散データローダ(BOS整列ベストフィット詰め込み)
├── dataset.py            # 事前学習用 parquet データのダウンロード/読み込み
├── engine.py             # KV キャッシュを用いた効率的な推論エンジン
├── optim.py              # Muon + AdamW オプティマイザ(1GPU / 分散対応)
├── checkpoint_manager.py # モデルチェックポイントの保存/読み込み
├── core_eval.py          # ベースモデルの CORE スコア評価(DCLM 論文)
├── loss_eval.py          # bits per byte(bpb)による評価
├── execution.py          # LLM が Python コードをツールとして実行する仕組み
├── flash_attention.py    # FA3/SDPA を自動切替する統一 Flash Attention インタフェース
├── fp8.py                # 最小限の FP8 学習(tensorwise 動的スケーリング)
└── common.py             # 小さなユーティリティ群
```

主なモジュールの役割は次の通りです。

- `gpt.py`: GPT Transformer 本体。モデルアーキテクチャを定義する `nn.Module`。
- `tokenizer.py`: GPT-4 スタイルの BPE トークナイザー。学習には rustbpe、推論には tiktoken を用います。
- `dataloader.py`: トークナイズを行いながらデータを供給する分散データローダ。BOS 整列のベストフィット詰め込み(bin-packing)でパディングを最小化します。
- `dataset.py`: 事前学習用の parquet データを必要に応じてダウンロードし読み込みます。
- `engine.py`: KV キャッシュを用いた効率的な推論エンジン。
- `optim.py`: Muon + AdamW オプティマイザ。単一 GPU / 分散のどちらにも対応します。

補足:

- `flash_attention.py` は FA3 と PyTorch SDPA を自動で切り替える統一インタフェースです。互換性のない CUDA GPU・MPS・CPU では PyTorch SDPA にフォールバックします。
- `fp8.py` は torchao の `Float8Linear` を約 150 行で置き換える最小限の tensorwise 版 FP8 学習実装です。

## `scripts/`

```
scripts/
├── tok_train.py    # トークナイザー: 学習
├── tok_eval.py     # トークナイザー: 圧縮率の評価
├── base_train.py   # ベースモデル: 事前学習
├── base_eval.py    # ベースモデル: CORE スコア / bpb / サンプル生成
├── chat_sft.py     # チャットモデル: SFT(教師ありファインチューニング)
├── chat_rl.py      # チャットモデル: 強化学習(RL)
├── chat_eval.py    # チャットモデル: タスク評価
├── chat_cli.py     # チャットモデル: CLI で対話
└── infer_bench.py  # 推論: レイテンシ/スループット/VRAM ベンチ
```

各スクリプトは `python -m scripts.<name>` の形式で個別に実行できます。ただし通常は `runs/*.sh` 経由でまとめて実行するのが一般的です。

## `tasks/`

```
tasks/
├── common.py     # 全タスクの基底(Task/HubDataset)と TaskMixture / TaskSequence
├── arc.py        # ARC: 多肢選択の理科問題
├── gsm8k.py      # GSM8K: 小学校レベルの算数問題(8K問)
├── humaneval.py  # 簡単な Python コーディングタスク(名称は HumanEval)
├── mmlu.py       # MMLU: 幅広い分野の多肢選択問題
└── smoltalk.py   # SmolTalk: HuggingFace 由来の会話データセット集合
```

補足:

- `Task` は「会話データセット + メタデータ + 評価基準」をまとめたものです。
- `HubDataset` は pyarrow の Table をラップした軽量なデータセットです。

## `runs/`

```
runs/
├── speedrun.sh     # ~$100 の nanochat d20 を学習する基準スクリプト
├── miniseries.sh   # ミニシリーズ学習
├── scaling_laws.sh # スケーリング則の実験
└── runcpu.sh       # CPU/MPS で動かす小さな例
```

## `tests/`

```
tests/
├── test_tokenizer.py           # BPE のラウンドトリップ、会話レンダリング
├── test_engine.py              # 推論エンジン、KV キャッシュ
├── test_execution.py           # サンドボックス化されたコード実行
├── test_optim.py               # MuonAdamW オプティマイザ(GPU 必要)
├── test_tasks.py               # タスクのスライス/ミックス、HubDataset
└── test_attention_fallback.py  # FA3/SDPA アテンションのフォールバック
```

## `dev/`

```
dev/
├── LEADERBOARD.md               # GPT-2 スピードランのリーダーボード解説
├── LOG.md                       # 開発ログ
├── repackage_data_reference.py  # 事前学習データのシャード生成(参照実装)
├── estimate_gpt3_core.ipynb     # GPT-3 CORE 推定ノートブック
└── scaling_analysis.ipynb       # スケーリング分析ノートブック
```

## 作業別の参照先

- モデル構造を変える場合: `nanochat/gpt.py`
- トークナイザーの学習/挙動を変える場合: `nanochat/tokenizer.py`, `scripts/tok_train.py`, `scripts/tok_eval.py`
- 事前学習データの取得/供給を変える場合: `nanochat/dataset.py`, `nanochat/dataloader.py`
- 事前学習の設定を変える場合: `scripts/base_train.py`, `runs/speedrun.sh`
- 微調整(SFT/RL)を変える場合: `scripts/chat_sft.py`, `scripts/chat_rl.py`
- 評価タスクを追加/変更する場合: `tasks/` 各ファイルと `tasks/common.py`
- 推論やチャットの挙動を変える場合: `nanochat/engine.py`, `scripts/chat_cli.py`, `scripts/infer_bench.py`
- パイプライン全体の実行方法を確認する場合: `runs/speedrun.sh`

## 注意点

- `runs/speedrun.sh` は 8XH100 ノードでの基準実行を反映しています。CPU/MPS 環境では `runs/runcpu.sh` を使ってください。
- `nanochat/flash_attention.py` により、FA3 非対応環境では自動で SDPA にフォールバックします。
- FP8 学習(`nanochat/fp8.py`)は CUDA の `_scaled_mm` を前提とします。
- スクリプトは `python -m scripts.<name>` で個別実行できますが、通常はルートの `runs/*.sh` を使う方が分かりやすいです。
