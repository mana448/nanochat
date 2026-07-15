# nanochat

> これは README.md の日本語訳です。原文(英語)は [README.md](README.md) を参照してください。

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat は、LLM を学習するための最もシンプルな実験用ハーネスです。単一の GPU ノードで動作するように設計されており、コードは最小限でハックしやすく、トークナイゼーション、事前学習(pretraining)、ファインチューニング(finetuning)、評価(evaluation)、推論(inference)といった LLM の主要なステージをすべてカバーしています。例えば、あなた自身の GPT-2 相当の能力を持つ LLM(2019 年当時、学習に約 43,000 ドルかかったもの)を、わずか 48 ドル(8×H100 GPU ノードで約 2 時間)で学習し、シンプルな CLI で対話することができます。スポットインスタンスを使えば、総コストは約 15 ドルにまで下げられます。さらに一般的には、nanochat は 1 つの複雑さのダイヤル、すなわち `--depth`(GPT トランスフォーマーモデルの層数)を設定するだけで、計算最適(compute-optimal)なモデルのミニシリーズ全体を学習できるように、そのまま使える形で構成されています(GPT-2 相当の能力は、たまたま depth 26 あたりになります)。その他のハイパーパラメータ(トランスフォーマーの幅、ヘッド数、学習率の調整、学習ホライズン、weight decay など)はすべて、最適な形で自動的に計算されます。

このリポジトリに関する質問については、Devin/Cognition の [DeepWiki](https://deepwiki.com/karpathy/nanochat) を使って質問するか、[Discussions タブ](https://github.com/karpathy/nanochat/discussions)を利用するか、Discord の [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) チャンネルに来ることをおすすめします。

## Time-to-GPT-2 リーダーボード

現在、開発の主な焦点は、最も多くの計算を要する事前学習ステージのチューニングにあります。modded-nanogpt リポジトリに触発され、進捗とコミュニティのコラボレーションを促進するために、nanochat は「GPT-2 スピードラン」のリーダーボードを維持しています。これは、DCLM CORE スコアで測定される GPT-2 グレードの能力まで nanochat モデルを学習するのに必要な実時間(wall-clock time)です。[runs/speedrun.sh](runs/speedrun.sh) スクリプトは、GPT-2 グレードのモデルを学習して対話するための、常に基準となる方法を反映しています。現在のリーダーボードは次のとおりです。

| # | time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | bump total batch size to 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | change dataset to NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

私たちが最も重視する指標は「time to GPT-2」、すなわち 8×H100 GPU ノード上で GPT-2 (1.6B) の CORE 指標を上回るために必要な実時間です。GPT-2 の CORE スコアは 0.256525 です。2019 年当時、GPT-2 の学習には約 43,000 ドルかかっていたので、7 年間にわたるスタック全体にわたる多くの進歩によって、これほど速く、しかも 100 ドルを大きく下回るコストで実現できるようになったのは驚くべきことです(例えば、現在の GPU 1 枚あたり約 3 ドル/時では、8×H100 ノードは約 24 ドル/時なので、2 時間で約 48 ドルです)。

リーダーボードの解釈方法や貢献方法についての詳しいドキュメントは、[dev/LEADERBOARD.md](dev/LEADERBOARD.md) を参照してください。

## はじめに(Getting started)

### セットアップ(Setup)

nanochat は依存関係の管理に [uv](https://docs.astral.sh/uv/) を使用します。インストールするには次のようにします。

```bash
uv sync --extra gpu    # Use for CUDA (A100/H100/etc.)
uv sync --extra cpu    # (or) Use for CPU-only / MPS
source .venv/bin/activate
```

開発用途(pytest、matplotlib、ipykernel、transformers などを追加)には次のようにします。

```bash
uv sync --extra gpu --group dev
```

### GPT-2 を再現して対話する(Reproduce and talk to GPT-2)

最も楽しめるのは、自分自身の GPT-2 を学習してそれと対話することです。そのためのパイプライン全体は、単一のファイル [runs/speedrun.sh](runs/speedrun.sh) に収められており、8×H100 GPU ノード上で実行するように設計されています。お好きなプロバイダーから新しい 8×H100 GPU マシンを起動し(例えば私は [Lambda](https://lambda.ai/service/gpu-cloud) を使っていて気に入っています)、学習スクリプトを開始します。

```bash
bash runs/speedrun.sh
```

これは実行に約 1.5 時間かかるので、screen セッション内で実行するとよいでしょう。完了したら、CLI 越しにモデルと対話できます。ローカルの uv 仮想環境が有効になっていることを再度確認し(`source .venv/bin/activate` を実行)、チャットします。

```bash
python -m scripts.chat_cli
```

物語や詩を書かせてみましょう。自分が誰なのかを尋ねてハルシネーション(幻覚)を見てみましょう。空はなぜ青いのか、あるいはなぜ緑なのかを尋ねてみましょう。このスピードランは 4e19 FLOPs 相当の能力のモデルなので、幼稚園児と話しているような感じです :)。スピードランモデルとの会話の例を挙げます。

> **User:** Hello whats up?
>
> **Assistant:** Hello! How can I help you today?
>
> **User:** Why is the sky blue?
>
> **Assistant:** The sky is blue due to an optical illusion called the Rayleigh Scattering of Light, which occurs when sunlight passes through tiny particles in the air. These particles are made up of tiny blue and violet particles that cause the light to bend in a particular way.
>
> **User:** That's cool. Write a poem about it.
>
> **Assistant:** In the twilight's glow, the sky is blue,<br>
> A canvas of shimmering blue, where skies unfold,<br>
> A world of blue, where blue and more blue,<br>
> Are woven together in a tapestry so fine,<br>
> Where every hue seems to sing a story.<br>
> ...

その他の補足事項:

- このコードは Ampere の 8×A100 GPU ノードでも問題なく動作しますが、少し遅くなります。
- すべてのコードは、`torchrun` を省略することで単一の GPU でも問題なく動作し、ほぼ同一の結果を生成します(コードは自動的に勾配累積(gradient accumulation)に切り替わります)が、8 倍の時間を待つ必要があります。
- GPU の VRAM が 80GB 未満の場合、一部のハイパーパラメータをチューニングしないと OOM / VRAM 不足になります。スクリプト内の `--device-batch-size` を探し、収まるまで減らしてください。例えば 32(デフォルト)から 16、8、4、2、あるいは 1 まで。それ以下にする場合は、もう少し内容を理解して、より工夫する必要があります。
- コードの大部分はごく普通の PyTorch なので、それをサポートするもの(xpu、mps など)であれば何でも動作するはずですが、これらのコードパスすべてを私自身が検証したわけではないので、荒削りな部分があるかもしれません。

## 研究(Research)

研究者で nanochat の改善に協力したい場合、関心のあるスクリプトが 2 つあります。[runs/scaling_laws.sh](runs/scaling_laws.sh) と [runs/miniseries.sh](runs/miniseries.sh) です。関連ドキュメントとして [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) を参照してください。素早い実験(約 5 分の事前学習)のために私が好むスケールは、12 層モデル(GPT-1 サイズ)を学習することで、例えば次のようにします。

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

これは wandb(run 名は "d12")を使用し、CORE 指標を最終ステップでのみ実行し、途中のチェックポイントのサンプリングや保存を行いません。私はコードの何かを変更し、d12(あるいは d16 など)を再実行して、それが改善につながったかどうかを、反復ループの中で確認するのが好きです。ある実行が効果的かどうかを見るために、私は次の wandb プロットを監視するのが好きです。

1. `val_bpb`(語彙サイズに依存しない bits per byte 単位の検証損失)を、`step`、`total_training_time`、`total_training_flops` の関数として。
2. `core_metric`(DCLM CORE スコア)
3. VRAM 使用率、`train/mfu`(Model FLOPS utilization、モデル FLOPS 利用率)、`train/tok_per_sec`(学習スループット)

例は[こちら](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044)を参照してください。

重要な点として、nanochat はただ 1 つの複雑さのダイヤル、すなわちトランスフォーマーの深さ(depth)を中心に記述・構成されています。この単一の整数が、その他のハイパーパラメータ(トランスフォーマーの幅、ヘッド数、学習率の調整、学習ホライズン、weight decay など)をすべて自動的に決定し、学習されたモデルが計算最適(compute optimal)になるようにします。この考え方は、ユーザーがこれらを考えたり設定したりする必要がなく、単に `--depth` を使ってより小さいモデルやより大きいモデルを要求するだけで、すべてが「ただ動く」というものです。depth をスイープすることで、さまざまなサイズの計算最適なモデルからなる nanochat のミニシリーズが得られます。GPT-2 相当の能力のモデル(現時点で最も関心が高いもの)は、現在のコードではおよそ d24〜d26 の範囲のどこかになります。ただし、リポジトリへの変更候補は、depth のすべての設定で機能するほどに原理的(principled)なものでなければなりません。

## CPU / MPS での実行(Running on CPU / MPS)

スクリプト [runs/runcpu.sh](runs/runcpu.sh) は、CPU や Apple Silicon 上で実行する非常にシンプルな例を示しています。学習される LLM を大幅に縮小し、数十分程度の妥当な時間内に学習が収まるようにしています。この方法では強い結果は得られません。

## 精度 / dtype(Precision / dtype)

nanochat は `torch.amp.autocast` を使用しません。代わりに、精度は 1 つのグローバルな `COMPUTE_DTYPE`(`nanochat/common.py` で定義)を通じて明示的に管理されます。デフォルトでは、これはハードウェアに基づいて自動検出されます。

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | Safe default. On recent macOS, MPS also runs `NANOCHAT_DTYPE=bfloat16` fine (~25% less memory, similar speed) |

`NANOCHAT_DTYPE` 環境変数を使ってデフォルトを上書きできます。

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # force fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # force bf16
```

仕組み: モデルの重みは(オプティマイザの精度のため)fp32 で保存されますが、私たちのカスタム `Linear` 層が、forward パス中にそれらを `COMPUTE_DTYPE` にキャストします。埋め込み(Embeddings)はメモリ節約のため、直接 `COMPUTE_DTYPE` で保存されます。これにより、autocast と同じ混合精度(mixed-precision)の恩恵を受けつつ、どの部分がどの精度で実行されるかを完全に明示的に制御できます。

注意: `float16` での学習では、勾配のアンダーフローを防ぐために `base_train.py` 内で `GradScaler` が自動的に有効になります。SFT もこれをサポートしていますが、RL は現時点ではサポートしていません。fp16 での推論はどこでも問題なく動作します。

## ガイド(Guides)

役立つ情報が含まれているかもしれないガイドをいくつか公開しています(新しいものから古いものの順)。

- [Feb 1 2026: Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) は最初の nanochat モデルのミニシリーズについて記述しています。
- nanochat に新しい能力を追加するには、[Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164) を参照してください。
- [Oct 13 2025: original nanochat post](https://github.com/karpathy/nanochat/discussions/1) は nanochat を紹介したものですが、現在では一部に非推奨(deprecated)の情報が含まれており、モデルも現在の master よりずっと古い(結果も劣る)ものです。

## ファイル構成(File structure)

```
.
├── LICENSE
├── README.md
├── dev
│   ├── nanochat.png
│   └── repackage_data_reference.py # 事前学習データのシャード生成
├── nanochat
│   ├── __init__.py                 # 空
│   ├── checkpoint_manager.py       # モデルのチェックポイントの保存/読み込み
│   ├── common.py                   # 各種の小さなユーティリティ、使い勝手向上のための機能
│   ├── core_eval.py                # ベースモデルの CORE スコアを評価(DCLM 論文)
│   ├── dataloader.py               # トークナイズを行う分散データローダー
│   ├── dataset.py                  # 事前学習データのダウンロード/読み込みユーティリティ
│   ├── engine.py                   # KV キャッシュを用いた効率的なモデル推論
│   ├── execution.py                # LLM がツールとして Python コードを実行できるようにする
│   ├── gpt.py                      # GPT の nn.Module トランスフォーマー
│   ├── loss_eval.py                # (損失の代わりに)bits per byte を評価
│   ├── optim.py                    # AdamW + Muon オプティマイザ、1GPU および分散対応
│   └── tokenizer.py                # GPT-4 スタイルの BPE トークナイザーのラッパー
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # ミニシリーズ学習スクリプト
│   ├── runcpu.sh                   # CPU/MPS 上での実行方法の小さな例
│   ├── scaling_laws.sh             # スケーリング則の実験
│   └── speedrun.sh                 # 約 100 ドルの nanochat d20 を学習
├── scripts
│   ├── base_eval.py                # ベースモデル: CORE スコア、bits per byte、サンプル
│   ├── base_train.py               # ベースモデル: 学習
│   ├── chat_cli.py                 # チャットモデル: CLI 越しに対話
│   ├── chat_eval.py                # チャットモデル: 評価タスク
│   ├── chat_rl.py                  # チャットモデル: 強化学習(RL)
│   ├── chat_sft.py                 # チャットモデル: SFT 学習
│   ├── infer_bench.py              # 推論: レイテンシ/スループット/VRAM ベンチマーク
│   ├── tok_eval.py                 # トークナイザー: 圧縮率を評価
│   └── tok_train.py                # トークナイザー: 学習
├── tasks
│   ├── arc.py                      # 多肢選択式の科学問題
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── gsm8k.py                    # 8K 件の小学校レベルの算数問題
│   ├── humaneval.py                # 名前は誤解を招くが、シンプルな Python コーディングタスク
│   ├── mmlu.py                     # 幅広いトピックの多肢選択式問題
│   └── smoltalk.py                 # HF の SmolTalk を寄せ集めたデータセット
├── tests
│   ├── test_attention_fallback.py  # FA3/SDPA アテンションのフォールバック
│   ├── test_engine.py              # 推論エンジン、KV キャッシュ
│   ├── test_execution.py           # サンドボックス化されたコード実行
│   ├── test_optim.py               # MuonAdamW オプティマイザ(GPU が必要)
│   ├── test_tasks.py               # タスクのスライシング、ミックスチャー、HubDataset
│   └── test_tokenizer.py           # BPE のラウンドトリップ、チャットのレンダリング
└── uv.lock
```

## 貢献(Contributing)

nanochat の目標は、1000 ドル未満の予算でエンドツーエンドに扱えるマイクロモデルの最先端を改善することです。アクセシビリティとは、全体的なコストのことでもありますが、認知的な複雑さのことでもあります。nanochat は、あらゆる設定が可能な LLM「フレームワーク」ではありません。コードベースには、巨大な設定オブジェクトも、モデルファクトリも、if-then-else の怪物も存在しません。これは、単一で、まとまりがあり、最小限で、読みやすく、ハックしやすく、最大限にフォークしやすい「強力なベースライン」となるコードベースであり、最初から最後まで実行して、対話できる ChatGPT モデルを生成するように設計されています。現在、私個人にとって最も興味深いのは、GPT-2 までのレイテンシを高速化すること(すなわち CORE スコアを 0.256525 より上にすること)です。現在これには約 1.5 時間かかりますが(3 時間から短縮)、事前学習ステージを改善することで、これをさらに向上させることができます。

現在の AI ポリシー: 開示(disclosure)。PR を提出する際は、LLM が実質的に寄与した部分や、自分で書いていない部分、あるいは完全には理解していない部分があれば申告してください。

## 謝辞(Acknowledgements)

- この名前(nanochat)は、事前学習のみをカバーしていた私の以前のプロジェクト [nanoGPT](https://github.com/karpathy/nanoGPT) に由来しています。
- nanochat はまた、明確な指標とリーダーボードで nanoGPT リポジトリをゲーム化した [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) にも触発されており、そのアイデアの多くと、事前学習向けの一部の実装を借用しています。
- fineweb と smoltalk を提供してくれた [HuggingFace](https://huggingface.co/) に感謝します。
- このプロジェクトの開発に使用した計算資源を提供してくれた [Lambda](https://lambda.ai/service/gpu-cloud) に感謝します。
- 最高の LLM ウィスパラー 🧙‍♂️ Alec Radford の助言/指導に感謝します。
- nanochat の issue、プルリクエスト、ディスカッションの管理を手伝ってくれた、リポジトリの czar である Sofie [@svlandeg](https://github.com/svlandeg) に感謝します。

## 引用(Cite)

研究において nanochat が役立った場合は、単に次のように引用してください。

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## ライセンス(License)

MIT
