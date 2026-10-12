# NanoJev-GRPO — NanoJev の判断モデルを GRPO で追加学習する

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

> **このリポジトリは、[TianyuCodings](https://github.com/TianyuCodings) さんの [NanoJev](https://github.com/TianyuCodings/NanoJev) を土台にした非公式の拡張です。**
> 判断モデル、公開されている重みとデータセット、ゲーム環境は NanoJev の成果です。このリポジトリは NanoJev のコミット [`76fdfc9`](https://github.com/TianyuCodings/NanoJev/tree/76fdfc9) から履歴を残したまま始め、エピソードの成否からの強化学習（GRPO）を足しています。元の作者に深く感謝します。足した部分の誤りは私の責任です。

![同じ ViZDoom Predict Position のシードを3つの重みでプレイした様子。公開の重みと SFT の続きはすぐ撃って外し、GRPO の重みは待ってから当てる](docs/assets/pp_train.gif)

*同じシードを3つの重みでプレイしたもの（記録した操作の再生）。NanoJev の公開の重みと、その SFT の続きは、すぐに撃って外します。GRPO の重みは待ってから当てます。試験用128エピソードのうち、SFT が外して GRPO が当てたのは36回、その逆は11回で、この動画はその36回の1つです。黄色の印は再生ツールが描き足したものです。*

## このリポジトリが示すこと

判断モデルは、1回の計算で各選択肢の確率を返し、文章を生成しません。そのぶん速い一方で、その選択が何を引き起こすかを考える場所がありません。公開されているゲームのデモは、コードが全部の選択肢をシミュレートし、結果を入力に書くことでこれを補っています。

このリポジトリは、もう一つの道を試します。先読みを重みに持たせる道です。NanoJev（Qwen3-0.6B に判断用のヘッドを付けたモデル）を、ViZDoom Predict Position で **エピソード単位の GRPO** で学習します。手本の操作も、入力に書いたシミュレーション結果も使わず、エピソードが敵を倒して終わったかどうかだけを使います。

| 重み | 手本 | 640エピソードでの成功率（貪欲／サンプリング） | 1判断を8ティックにした条件（128） | ViZDoom Basic（128） |
|---|---|---:|---:|---:|
| NanoJev の公開の重み | — | 19.4% / 18.0% | 14 | 128 |
| 手本の操作で SFT | 使う | 68.6% / 66.1% | 57 | 128 |
| **公開の重みから GRPO** | **使わない** | **79.1% / 77.2%** | 34 | 107 |
| SFT の頭打ちから GRPO を続ける | SFT の分だけ | 81.4% / 80.8% | 69 | 128 |

- 640エピソードは、公開されている試験用128と、学習に使っていない512シードです。
- SFT と GRPO は同じ重みから始め、同じ512個の学習用シードを使います。640エピソードのうち、SFT だけが成功したのは72、GRPO だけが成功したのは139でした（符号検定、p < 0.00001）。
- SFT は約400回の更新で頭打ちになり、さらに800回続けても伸びません。GRPO は手本なしでその高さを越え、SFT の頭打ちから始めた GRPO はさらに伸びました。

![学習中の、学習に使っていない512シードでの成功率。SFT は65〜69%で頭打ち、GRPO 単体は77%、SFT の頭打ちからの GRPO は81%で終わる](docs/assets/curves.png)

### わかっている弱点

- **覚えるのは、学習した条件に合う戦い方です。** GRPO の方策は、待ってから撃ちます。1判断を4ティックから8ティックに変えると、128エピソード中75回は1発も撃ちません（撃てた回の命中率は 64% で、SFT の 45% より高い）。ViZDoom Basic での失敗21回は、すべて時間切れです。
- **伸びたのは成功率で、先読みそのものが SFT より良くなったとは示せていません。** 撃つ確率が「当たる場面」と「外れる場面」をどれだけ分けているか（AUC。敵が見えていて時間が残っている場面）は、学習前 0.64、SFT 0.80、GRPO 0.76 でした。
- **貪欲選択が一時的に崩れました。** GRPO の20ステップ目あたりです。サンプリングでは崩れていません。
- **最後の行は、GRPO の行と条件をそろえた比較ではありません。** 新しい1,600シード、1ステップ32シード、更新2回、学習率は3分の1です。
- **規模は小さいです。** 各行の学習は1回、ゲームは1つ、モデルは0.6B です。

## 理由：モデルは読めるが、先は読めない

同じ課題で、API で提供されている Jev（`typesafe/jev-1.13`）も測りました。

![同じシードを Jev が2通りにプレイした様子。入力そのままでは撃つを選び続けて時間切れになり、コードが結果を候補に書き足すと当てる](docs/assets/pp_jev.gif)

| 誰が選ぶか | 候補に書いてあること | 成功（試験用128エピソード） |
|---|---|---:|
| 一様な乱択 | — | 12 |
| Jev | 元の文面（例：`Hold attack for 4 Doom ticks.`） | 0 |
| Jev | 元の文面＋コードがシミュレートした結果 | 112 |
| Jev | 同上＋書き足しの読み方を説明する指示文 | 126 |
| コードだけ：「倒す」と書いてある選択肢を選ぶ | — | 126 |

- 書き足しありで Jev が成功した126エピソードは、コードだけのルールが成功した126エピソードと完全に同じです。
- Jev は、状態について読めば分かる質問（敵は画面の中心より右か）には 97.7% 正解します。撃つ確率は、当たる場面と外れる場面を分けていません（AUC 0.49）。
- 規則がすべて分かっている Snake では、Jev は読めば分かる質問で 100%、1手先に規則を当てはめる質問で 63.5%、5手先で 70.0% でした。答える前に考える LLM は 98〜100% と 72〜99% です。同じ LLM に一言で答えさせると、5手先は 39.5〜50.5% に落ちます。差を作っているのは、答える前に考えられるかどうかです。

## 判断モデルへの GRPO の当てはめ方

![1つのシードを16回プレイし、各エピソードに成否から組の中での相対評価を付け、そのエピソードのすべての判断をその評価で更新する](docs/assets/grpo.png)

LLM の GRPO と同じもので、置き換えは2つです。1つの行動は、トークンではなく候補からの選択です。1本の軌跡は、1つの回答ではなく1エピソード（20〜30回の判断）です。

```text
A_i     = (r_i - mean(r)) / std(r)                      r_i は 0 か 1。同じシードの16エピソードで計算
rho_i,t = pi_theta(a_i,t | s_i,t) / pi_old(a_i,t | s_i,t)
L       = - 全判断の平均  min(rho_i,t * A_i, clip(rho_i,t, 1 - eps, 1 + eps) * A_i),   eps = 0.2
```

16回の成否がすべて同じ組は、学習の手がかりにならないので捨てます。価値を見積もるモデル、KL の罰、参照モデルはありません。実装は [`scripts/pp_rl.py`](scripts/pp_rl.py) の `group_advantages` と `clipped_loss` で、詳しい説明は [`docs/PP_RL.md`](docs/PP_RL.md) にあります。

これを現実的な計算量にしているのが `scripts/pp_tree.py` です。状態と質問は1判断につき1回だけ計算し、候補がその前半を共有します。1判断あたり約1,950トークンが約530トークンになり、出力は同じです（`pp_rl.py check-forward` で確認できます）。

## 使い方

CUDA の GPU が1枚必要です（上の実験は A100 40GB 1枚）。ViZDoom を動かす CPU コアも使います。

```bash
git clone https://github.com/Hajime-Y/nanojev-grpo && cd nanojev-grpo
bash scripts/pp_setup.sh        # .venv、固定した依存、Hugging Face から NanoJev の重みとデータセット
source .venv/bin/activate
```

公開の重みを評価する（27/128 になるはずです）：

```bash
python scripts/pp_rl.py eval --checkpoint checkpoints/NanoJev-unified \
  --cases configs/sonic_predict_supervision_v1_cases.jsonl --splits test --mode greedy \
  --workers 24 --output runs_pp/base_test_greedy.jsonl
```

公開の重みからの GRPO と、比較用の SFT：

```bash
scripts/pp_train.sh grpo_main grpo --steps 50 --tasks-per-step 16 --group-size 16 --minibatches 8 \
  --clip 0.2 --temperature 1.0 --backbone-lr 1e-5 --head-lr 1e-4 --eval-every 5 --save-every 10 --workers 24

scripts/pp_train.sh sft_main sft --updates 1200 --batch-questions 64 --backbone-lr 1e-5 --head-lr 1e-4 \
  --eval-every 100 --save-every 100 --workers 24
```

SFT の最良の重み（`python scripts/pp_pick_best.py runs_pp/sft_main/eval.jsonl update` で名前が出ます）から、新しいシードで GRPO を続ける：

```bash
INIT=runs_pp/sft_main/ckpt/update-1100 CASES=configs/pp_fresh1600_cases.jsonl \
scripts/pp_train.sh grpo_from_sft grpo --steps 50 --tasks-per-step 32 --group-size 16 --minibatches 2 \
  --clip 0.2 --temperature 1.0 --backbone-lr 3e-6 --head-lr 3e-5 --eval-every 5 --save-every 10 \
  --workers 16 --max-microbatch-tokens 17000 --max-forward-tokens 70000
```

表の評価、エピソードごとの比較、再生動画：

```bash
scripts/pp_final_eval.sh sft  runs_pp/sft_main/ckpt/update-1100
scripts/pp_final_eval.sh grpo runs_pp/grpo_main/ckpt/step-0050
python scripts/pp_compare.py --a runs_pp/final/sft_test_greedy.jsonl  runs_pp/final/sft_heldout_greedy.jsonl \
                             --b runs_pp/final/grpo_test_greedy.jsonl runs_pp/final/grpo_heldout_greedy.jsonl
python scripts/pp_other_games.py --checkpoint runs_pp/grpo_main/ckpt/step-0050 --output runs_pp/other/grpo.json

pip install pillow
python scripts/pp_video.py --seed 9300761 --output pp_train.gif \
  --panel "SFT=runs_pp/final/sft_test_greedy.jsonl" --panel "GRPO=runs_pp/final/grpo_test_greedy.jsonl"
```

モデルなしで CPU だけで動く測定が2つあります。`python scripts/pp_probe.py play --engine random ...` と、`--engine oracle --annotate lookahead ...`（コードだけのルール）です。`--engine jev` は、手元の中継サーバー（`scripts/pp_jev_relay.py`。macOS のキーチェーンから鍵を読み、OpenRouter に転送する）に送ります。Snake の質問は `scripts/pp_snake_probe.py` で作って聞きます。

テスト：

```bash
cd scripts && python -m unittest test_pp_rl test_pp_probe test_pp_snake_probe \
  test_unified_pipeline test_unified_doom_env test_unified_grid_envs test_composed_maze test_question_contract
```

## このリポジトリの中身

| 場所 | 由来 | 役割 |
|---|---|---|
| `scripts/pp_rl.py`、`pp_envpool.py`、`pp_tree.py` | 追加 | GRPO、比較用の SFT、まとめて回す評価、前半を共有する計算 |
| `scripts/pp_probe.py`、`pp_snake_probe.py`、`pp_jev_relay.py` | 追加 | 質問による測定、結果を書き足したプレイ、Jev／コードだけ／乱択の各エンジン、Snake の質問 |
| `scripts/pp_*.sh`、`pp_compare.py`、`pp_pick_best.py`、`pp_probe_auc.py`、`pp_other_games.py`、`pp_video.py` | 追加 | セットアップ、再開つきの学習、表の評価、分析、再生動画 |
| `configs/pp_*_cases.jsonl` | 追加 | 上の数字に使った、評価用・測定用・新しい学習用のシード |
| `scripts/train_toy_decisions.py`、`predict_toy_decisions.py` | NanoJev | 判断モデルと、その読み込み |
| `scripts/unified_*.py`、`snake_game.py`、`scaled_maze.py`、`evaluate_*.py` | NanoJev | 環境、リクエストの形式、プレイの仕組み |
| `configs/sonic_predict_supervision_v1_cases.jsonl`、`configs/appo_basic_supervision_v1_cases.jsonl` | NanoJev | 公開されている学習用・確認用・試験用・条件違いのシード |

NanoJev のブラウザデモ、研究ログ、結果ファイル、データを作るコード、ほかの実験は、ここでは使わないので通常のコミットで削除しました。すべて [NanoJev の `76fdfc9`](https://github.com/TianyuCodings/NanoJev/tree/76fdfc9) にあります。モデルそのもの、その学習の仕組み、ドキュメントもそちらを見てください。この実験で学習した重みは公開していません。

## ライセンスと謝辞

NanoJev と同じ MIT です。[`LICENSE`](LICENSE) には元の著作権表示を残しています。

- TianyuCodings さんの [NanoJev](https://github.com/TianyuCodings/NanoJev)：モデル、公開されている[重み](https://huggingface.co/C-Tianyu/NanoJev)と[データセット](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data)、環境。
- ゲームは [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)。SFT で使った NanoJev のデータセットの手本の操作は、[Sonic Doom](https://github.com/thainv0212/sonic_doom) の方策が出したものです。
- GRPO は [DeepSeekMath](https://arxiv.org/abs/2402.03300) で提案された手法です。
- TypeSafe の [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) は、測定に使った API 提供の判断モデルです。
