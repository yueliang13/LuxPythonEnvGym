# RL_HW LuxAI 2021 Evaluation Notes

本目录用于 LuxAI 2021 强化学习大作业，目前包含两个已经可以直接对抗评测的 agent：

- `LuxAI-G4/`：模仿学习 agent，简称 **G4**。
- `LuxPythonEnvGym/`：基于 LuxPythonEnvGym 的 naive PPO baseline，简称 **PPO**。

当前在可控 seed 和 both-sides 对抗评测中，G4 已经可以稳定超过 PPO。后续 RL 训练建议以这里的评测脚本作为 baseline 对比入口。

## 目录结构

```text
RL_HW/
  LuxAI-G4/                 # G4 agent 代码与权重
  LuxPythonEnvGym/          # PPO baseline 代码与权重
  run_lux_match.sh          # 单局对战，适合生成 replay
  eval_lux_winrate.py       # 多轮对抗，统计胜率，支持并行
  run_lux_tournament.sh     # 官方 lux-ai-2021 tournament 模式
```

## 1. 环境要求

需要按照 LuxAI 2021 官方说明安装 Python 环境和官方 runner。
当前的测试目前只支持纯cpu测试，因为本人拥有的显卡(A6000和4090)均与lux-ai不兼容，G4队伍官方使用的是1080是兼容的。不过cpu测试的速度也并不慢。
我自己的G4的训练是在4090上面训练的，因为lux-ai在训练的时候使用不到。

## 2. 单局对战：run_lux_match.sh

`run_lux_match.sh` 用于让 G4 和 PPO 跑一场官方 LuxAI 2021 对局，适合生成 replay 并人工看回放。

默认对战顺序：

```text
Player 0: LuxAI-G4/main.py
Player 1: LuxPythonEnvGym/kaggle_submissions/main_lux-ai-2021.py
```

基础用法：

```bash
cd /home/liuxiaoyang/data/RL_HW
./run_lux_match.sh --seed 42 --storeReplay=true --storeLogs=true
```

交换先后手：

```bash
./run_lux_match.sh --seed 42 --swap --storeReplay=true --storeLogs=true
```

常用参数：

```text
--seed SEED             指定地图随机 seed，默认 42。
--swap                  交换先后手，让 PPO 为 player 0，G4 为 player 1。
--storeReplay=true      保存 replay JSON。
--storeReplay=false     不保存 replay。
--storeLogs=true        保存 agent stderr/error logs。
--storeLogs=false       不保存 error logs。
--loglevel=0            压低官方 runner 输出。
--loglevel=2            显示 warning，例如碰撞、非法移动等。
```

如果开启 replay，文件会保存在 `replays/` 目录下，格式是 `.json`。可以把 replay JSON 上传到 LuxAI replay viewer 网站查看可视化回放。

## 3. 多轮胜率评测：eval_lux_winrate.py

`eval_lux_winrate.py` 用于对 G4 和 PPO 进行多轮对抗，计算胜率。它比官方 tournament 更适合作为报告里的主实验，因为它能明确控制：

- seed 范围
- 总局数
- 是否正反手都跑
- 胜率统计方式
- 并行进程数

基础用法：

```bash
cd /home/liuxiaoyang/data/RL_HW
./eval_lux_winrate.py --games 20
```

更推荐的公平评测方式是每个 seed 正反手各打一局：

```bash
./eval_lux_winrate.py --games 20 --both-sides
```

这会实际跑 40 场，因为每个 seed 会跑：

```text
G4 vs PPO
PPO vs G4
```

支持多进程并行和线程控制：

```bash
./eval_lux_winrate.py --games 50 --both-sides --workers 4 --threads-per-match 16
```

参数含义：

```text
--games N                  评测 N 个 seed。默认 10。
--seed-start S             起始 seed。默认 42。
--both-sides               每个 seed 正反手各跑一局。
--fixed-order              不交替先后手，始终 G4 为 player 0，PPO 为 player 1。
--workers N                同时跑 N 场 match。
--threads-per-match N      每场 match 给 PyTorch/BLAS 使用的 CPU 线程数。
--store-replay             保存 replay。
--store-logs               保存 logs。
--maxtime MS               每回合时间限制，默认 10000。
--loglevel LEVEL           官方 runner 日志等级，默认 0。
```

`workers` 和 `threads-per-match` 的直观理解：

```text
总 CPU 压力大致约等于 workers * threads-per-match
```

例如：

```bash
./eval_lux_winrate.py --games 50 --both-sides --workers 4 --threads-per-match 16
```

大致表示同时跑 4 场，每场内部最多使用 16 个计算线程。

如果 CPU 占用太高或机器卡顿，可以降低：

```bash
./eval_lux_winrate.py --games 50 --both-sides --workers 2 --threads-per-match 8
```

如果单局太慢，可以提高：

```bash
./eval_lux_winrate.py --games 50 --both-sides --workers 2 --threads-per-match 32
```

建议先小规模测试：

```bash
./eval_lux_winrate.py --games 4 --both-sides --workers 2 --threads-per-match 16
```

## 4. 官方锦标赛模式：run_lux_tournament.sh

`run_lux_tournament.sh` 调用官方 `lux-ai-2021 --tournament`，用于做官方风格的 tournament 排名。

基础用法：

```bash
cd /home/liuxiaoyang/data/RL_HW
./run_lux_tournament.sh
```

指定 wins 排名和并发：

```bash
./run_lux_tournament.sh --rank-system wins --max-concurrent 4
```

保存 replay/log：

```bash
./run_lux_tournament.sh --rank-system trueskill --store-replay true --store-logs true
```

参数说明：

```text
--rank-system wins|elo|trueskill
    控制 tournament 排名系统。
    wins 最直观，适合两个 agent 简单比较。
    elo/trueskill 更适合多个 agent 长期排名。

--max-concurrent N
    官方 tournament 同时最多跑多少场 match。
    类似 eval_lux_winrate.py 里的 --workers。
```

注意：官方 `--tournament` 模式可控参数较少，不能方便地指定总共跑多少场、seed 范围、以及 both-sides 策略。官方 help 也说明，更多 tournament 配置建议复制官方 runner 源码自行修改。因此它更适合作为补充验证，不建议作为最终主要实验表格数据。

运行时会动态刷新类似如下表格：

```text
Total Matches: 70 | Matches Queued: 8
Name | ID | W | T | L | Points | Matches
```

含义：

```text
Total Matches      当前已经完成/统计的比赛数量。
Matches Queued     当前仍在队列中等待或调度的比赛数量。
W                  胜场数。
T                  平局数。
L                  负场数。
Points             积分。wins 模式下一般是 win=3, tie=1, loss=0。
Matches            该 agent 已经参与统计的比赛数。
```

## 5. 迁移到其他机器时需要改的路径

当前脚本里有服务器绝对路径。如果队友把项目解压到其他机器或其他目录，需要检查并修改这些位置。

三个评测脚本：

```text
run_lux_match.sh
run_lux_tournament.sh
eval_lux_winrate.py
```

重点修改：

```text
ROOT=/home/liuxiaoyang/data/RL_HW
PYTHON=/data/user/liuxiaoyang/.conda/envs/lux/bin/python
```

PPO 入口也要确认模型路径：

```text
LuxPythonEnvGym/kaggle_submissions/main_lux-ai-2021.py
```

当前 PPO 权重路径：

```text
LuxPythonEnvGym/kaggle_submissions/model.zip
```

G4 权重当前位于：

```text
LuxAI-G4/model.pth
```

G4 入口已经做了兼容：优先尝试配置里的 `MODEL_DIR/model.pth`，不存在时回退到 `LuxAI-G4/model.pth`。

## 6. 已知注意事项

- LuxAI 官方 runner 的 warning，例如 unit collided、off map，不一定表示程序崩溃，只表示 agent 输出了碰撞或非法移动动作。
- 如果 agent 崩溃或启动失败，优先打开 `--storeLogs=true` 查看 `errorlogs/`。
- 如果需要看策略行为，优先用 `run_lux_match.sh --storeReplay=true` 生成 replay。
- 如果需要报告里的定量结果，优先用 `eval_lux_winrate.py --both-sides`。
- 官方 tournament 输出是动态排名，不如 `eval_lux_winrate.py` 的 seed sweep 透明。

## 7. 推荐实验命令

快速生成一局 replay：

```bash
./run_lux_match.sh --seed 42 --storeReplay=true --storeLogs=true
```

快速 sanity check：

```bash
./eval_lux_winrate.py --games 4 --both-sides --workers 2 --threads-per-match 16
```

正式一点的胜率评测：

```bash
./eval_lux_winrate.py --games 50 --both-sides --workers 4 --threads-per-match 16
```

官方 tournament 补充验证：

```bash
./run_lux_tournament.sh --rank-system wins --max-concurrent 2
```
