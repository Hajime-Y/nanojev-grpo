# NanoJev-GRPO — 用 GRPO 对 NanoJev 决策模型做后训练

[English](README.md) | **简体中文** | [日本語](README.ja.md)

> **本仓库是在 [TianyuCodings](https://github.com/TianyuCodings) 的 [NanoJev](https://github.com/TianyuCodings/NanoJev) 基础上做的非官方扩展。**
> 决策模型、已发布的权重与数据集、游戏环境都是 NanoJev 的成果。本仓库从 NanoJev 的提交 [`76fdfc9`](https://github.com/TianyuCodings/NanoJev/tree/76fdfc9) 出发并保留其全部提交历史，在此之上加入了基于回合成败的强化学习（GRPO）。非常感谢原作者。新增部分如有错误，责任在我。

![同一个 ViZDoom Predict Position 种子由三个权重分别游玩：已发布的权重和继续做 SFT 的权重立刻开火并打偏，GRPO 的权重先等待再命中](docs/assets/pp_train.gif)

*同一个种子、三个权重，由记录下来的操作回放得到。NanoJev 已发布的权重和在其基础上继续 SFT 的权重立刻开火并打偏；GRPO 的权重先等待再命中。在 128 个测试回合中，SFT 打偏而 GRPO 命中的有 36 个，反过来的有 11 个；这段动画是那 36 个之一。黄色标记是回放工具加上的。*

## 本仓库想说明什么

决策模型在一次前向计算中给出每个候选项的概率，不生成文字。这让它很快，但也意味着它没有地方去“想”某个选择会带来什么后果。公开的游戏演示绕开这一点的办法是：由代码把每个选项都模拟一遍，再把结果写进输入。

本仓库验证另一条路：把“往前看”的能力放进权重里。我们在 ViZDoom Predict Position 上用**以整个回合为单位的 GRPO** 训练 NanoJev（Qwen3-0.6B 加决策头）：不用专家动作，输入里也不写模拟结果，只用回合最后是否击杀这一个信号。

| 权重 | 专家标签 | 640 个回合的成功率（贪心 / 采样） | 每次决策 8 tick 的条件（128） | ViZDoom Basic（128） |
|---|---|---:|---:|---:|
| NanoJev 已发布的权重 | — | 19.4% / 18.0% | 14 | 128 |
| 用专家动作做 SFT | 使用 | 68.6% / 66.1% | 57 | 128 |
| **从已发布权重做 GRPO** | **不使用** | **79.1% / 77.2%** | 34 | 107 |
| 从 SFT 的平台期继续做 GRPO | 仅 SFT 部分 | 81.4% / 80.8% | 69 | 128 |

- 640 个回合由已发布的测试集（128）和 512 个未参与训练的种子组成。
- SFT 和 GRPO 从同一份权重出发，使用相同的 512 个训练种子。在这 640 个回合中，只有 SFT 成功的有 72 个，只有 GRPO 成功的有 139 个（精确符号检验，p < 0.00001）。
- SFT 在约 400 次更新后进入平台期，再训练 800 次也没有提升。GRPO 在没有标签的情况下超过了这个水平；从 SFT 平台期出发的 GRPO 还能继续上升。

![训练过程中在 512 个留出种子上的成功率：SFT 停在 65% 到 69%，单独的 GRPO 最终为 77%，从 SFT 平台期出发的 GRPO 最终为 81%](docs/assets/curves.png)

### 已知的弱点

- **学到的是适合训练条件的打法。** GRPO 的策略是先等待再开火。把每次决策从 4 tick 改成 8 tick 后，128 个回合里有 75 个一枪未开（开了火的回合命中率为 64%，SFT 为 45%）；在 ViZDoom Basic 上的 21 次失败全部是超时。
- **提高的是成功率；没有证据表明“往前看”本身比 SFT 更好。** 用 AUC 衡量开火概率能否区分“此刻开火会命中”和“会打偏”的状态（怪物可见且还有时间）：训练前 0.64，SFT 后 0.80，GRPO 后 0.76。
- **贪心选择曾短暂崩溃**，发生在 GRPO 的第 20 步左右；采样评估没有崩溃。
- **最后一行与 GRPO 那一行不是受控对比。** 它使用 1,600 个新种子、每步 32 个种子、每步更新 2 次、学习率为三分之一。
- **规模很小：** 每行只训练一次，只有一个游戏，模型为 0.6B。

## 原因：模型会读，但不会往前看

同一个任务上，我们也测了托管的 Jev 模型（`typesafe/jev-1.13`）。

![同一个种子由 Jev 游玩两次：输入保持原样时它一直选择开火直到超时，候选项里写入代码模拟的结果后则命中](docs/assets/pp_jev.gif)

| 由谁选择 | 候选项里写了什么 | 成功数（128 个测试回合） |
|---|---|---:|
| 均匀随机 | — | 12 |
| Jev | 原始文本，例如 `Hold attack for 4 Doom ticks.` | 0 |
| Jev | 原始文本 + 代码模拟出的结果 | 112 |
| Jev | 同上，再加一句解释标注含义的指令 | 126 |
| 只用代码：选择文本里写着“击杀”的选项 | — | 126 |

- 带标注时 Jev 成功的 126 个回合，与只用代码的规则成功的 126 个回合完全相同。
- 对于读一下状态就能回答的问题（怪物是否在屏幕中心右侧），Jev 的正确率为 97.7%。它的开火概率不能区分命中和打偏（AUC 0.49）。
- 在所有规则都已知的 Snake 上，Jev 在查找类问题上为 100%，把规则往前应用一步时为 63.5%，往前五步时为 70.0%。先推理再回答的对话 LLM 分别达到 98–100% 和 72–99%；同样的 LLM 只用一个词回答时，五步问题降到 39.5–50.5%。差别来自回答之前能不能先思考。

## GRPO 如何用在决策模型上

![一个种子游玩 16 次，每个回合根据成败得到组内相对优势，该回合中的每一次决策都用这个优势更新](docs/assets/grpo.png)

它就是用于 LLM 的 GRPO，只做了两处替换：一个动作是在候选项中做的一次选择，而不是一个 token；一条轨迹是一个回合（20 到 30 次决策），而不是一段回答。

```text
A_i     = (r_i - mean(r)) / std(r)                      r_i 为 0 或 1，在同一种子的 16 个回合内计算
rho_i,t = pi_theta(a_i,t | s_i,t) / pi_old(a_i,t | s_i,t)
L       = - 对所有决策取平均  min(rho_i,t * A_i, clip(rho_i,t, 1 - eps, 1 + eps) * A_i),   eps = 0.2
```

16 个回合结果全部相同的组不含任何信号，直接丢弃。没有 critic，没有 KL 项，也没有参考模型。对应的两个函数是 [`scripts/pp_rl.py`](scripts/pp_rl.py) 里的 `group_advantages` 和 `clipped_loss`；完整说明见 [`docs/PP_RL.md`](docs/PP_RL.md)。

`scripts/pp_tree.py` 让这件事在计算上可行：每次决策只对状态和问题编码一次，各候选项共享这段前缀，每次决策约 530 个 token，而不是约 1,950 个，输出不变（可用 `pp_rl.py check-forward` 验证）。

## 快速开始

需要一块 CUDA GPU（上面的实验用了一块 A100 40 GB），以及运行 ViZDoom 的 CPU 核心。

```bash
git clone https://github.com/Hajime-Y/nanojev-grpo && cd nanojev-grpo
bash scripts/pp_setup.sh        # .venv、固定版本的依赖、从 Hugging Face 下载 NanoJev 的权重和数据集
source .venv/bin/activate
```

评估已发布的权重（应得到 27/128）：

```bash
python scripts/pp_rl.py eval --checkpoint checkpoints/NanoJev-unified \
  --cases configs/sonic_predict_supervision_v1_cases.jsonl --splits test --mode greedy \
  --workers 24 --output runs_pp/base_test_greedy.jsonl
```

从已发布权重做 GRPO，以及作为对照的 SFT：

```bash
scripts/pp_train.sh grpo_main grpo --steps 50 --tasks-per-step 16 --group-size 16 --minibatches 8 \
  --clip 0.2 --temperature 1.0 --backbone-lr 1e-5 --head-lr 1e-4 --eval-every 5 --save-every 10 --workers 24

scripts/pp_train.sh sft_main sft --updates 1200 --batch-questions 64 --backbone-lr 1e-5 --head-lr 1e-4 \
  --eval-every 100 --save-every 100 --workers 24
```

从最好的 SFT 快照（`python scripts/pp_pick_best.py runs_pp/sft_main/eval.jsonl update` 会给出它的名字）出发，在新种子上继续做 GRPO：

```bash
INIT=runs_pp/sft_main/ckpt/update-1100 CASES=configs/pp_fresh1600_cases.jsonl \
scripts/pp_train.sh grpo_from_sft grpo --steps 50 --tasks-per-step 32 --group-size 16 --minibatches 2 \
  --clip 0.2 --temperature 1.0 --backbone-lr 3e-6 --head-lr 3e-5 --eval-every 5 --save-every 10 \
  --workers 16 --max-microbatch-tokens 17000 --max-forward-tokens 70000
```

表中的评估、逐回合的配对比较，以及回放视频：

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

有两项测量不需要模型，只用 CPU 就能跑：`python scripts/pp_probe.py play --engine random ...` 和 `--engine oracle --annotate lookahead ...`（只用代码的规则）。`--engine jev` 会把请求发给本地的中转服务（`scripts/pp_jev_relay.py`，它从 macOS 钥匙串读取密钥并转发到 OpenRouter）。Snake 的问题由 `scripts/pp_snake_probe.py` 生成并提问。

测试：

```bash
cd scripts && python -m unittest test_pp_rl test_pp_probe test_pp_snake_probe \
  test_unified_pipeline test_unified_doom_env test_unified_grid_envs test_composed_maze test_question_contract
```

## 仓库里有什么

| 路径 | 来源 | 用途 |
|---|---|---|
| `scripts/pp_rl.py`、`pp_envpool.py`、`pp_tree.py` | 新增 | GRPO、对照用的 SFT、批量评估、共享前缀的前向计算 |
| `scripts/pp_probe.py`、`pp_snake_probe.py`、`pp_jev_relay.py` | 新增 | 探针测量、带结果标注的游玩、Jev / 只用代码 / 随机三种引擎、Snake 问题 |
| `scripts/pp_*.sh`、`pp_compare.py`、`pp_pick_best.py`、`pp_probe_auc.py`、`pp_other_games.py`、`pp_video.py` | 新增 | 环境搭建、可续跑的训练、表中的评估、分析、回放视频 |
| `configs/pp_*_cases.jsonl` | 新增 | 上面数字所用的留出种子、探针种子和新的训练种子 |
| `scripts/train_toy_decisions.py`、`predict_toy_decisions.py` | NanoJev | 决策模型及其加载器 |
| `scripts/unified_*.py`、`snake_game.py`、`scaled_maze.py`、`evaluate_*.py` | NanoJev | 环境、请求格式和 rollout 流程 |
| `configs/sonic_predict_supervision_v1_cases.jsonl`、`configs/appo_basic_supervision_v1_cases.jsonl` | NanoJev | 已发布的训练 / 开发 / 测试 / OOD 种子 |

NanoJev 的浏览器演示、研究日志、结果文件、数据构建脚本和其他实验在这里用不到，已通过普通提交删除。它们都保留在 [NanoJev 的 `76fdfc9`](https://github.com/TianyuCodings/NanoJev/tree/76fdfc9)；模型本身、它的训练流程和文档也请到那里查看。本实验训练出的权重没有发布。

## 许可与致谢

与 NanoJev 相同，采用 MIT 许可；见 [`LICENSE`](LICENSE)，其中保留了原始的版权声明。

- TianyuCodings 的 [NanoJev](https://github.com/TianyuCodings/NanoJev)：模型、已发布的[权重](https://huggingface.co/C-Tianyu/NanoJev)和[数据集](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data)、环境。
- 游戏来自 [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)；SFT 对照所用的 NanoJev 数据集中的专家动作由 [Sonic Doom](https://github.com/thainv0212/sonic_doom) 的策略产生。
- GRPO 出自 [DeepSeekMath](https://arxiv.org/abs/2402.03300)。
- TypeSafe 的 [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 是探针中测量的托管决策模型。
