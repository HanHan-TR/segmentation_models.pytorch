"""
对比 Optuna 中 log=True 与 step=0.1 在区间 [0.1, 100] 上的采样分布。

运行：
    pip install optuna numpy matplotlib
    python compare_log_vs_step.py

输出：
    终端打印三个数量级的概率表
    弹出 / 保存 log_vs_step.png（2x2 直方图）
"""
import numpy as np
import matplotlib.pyplot as plt
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

LOW, HIGH, N, SEED = 0.1, 100.0, 100_000, 0


def sample(kwargs, n=N):
    """用 Optuna 的 RandomSampler 真实采样，返回一维 numpy 数组。"""
    def objective(trial):
        trial.suggest_float("x", LOW, HIGH, **kwargs)
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=SEED))
    study.optimize(objective, n_trials=n)
    return np.array([t.params["x"] for t in study.trials])


def decade_prob(x):
    """统计 [0.1,1)、[1,10)、[10,100] 三个数量级的占比。"""
    return [
        float(np.mean((x >= 0.1) & (x < 1.0))),
        float(np.mean((x >= 1.0) & (x < 10.0))),
        float(np.mean((x >= 10.0) & (x <= 100.0))),
    ]


# ---------- 1. 采样 ----------
print(f"采样中 (N={N}) ...")
x_log = sample({"log": True})     # 对数均匀
x_step = sample({"step": 0.1})     # 线性网格，步长 0.1

# ---------- 2. 打印概率表 ----------
labels = ["[0.1, 1)", "[1, 10)", "[10, 100]"]
p_log, p_step = decade_prob(x_log), decade_prob(x_step)

print("\n每个数量级被采中的概率：")
print(f"{'区间':<12}{'log=True':>10}{'step=0.1':>12}")
for name, a, b in zip(labels, p_log, p_step):
    print(f"{name:<12}{a:>9.2%}{b:>11.2%}")

print("\n样本落在 x < 1 的数量：")
print(f"  log=True : {(x_log < 1).sum():>7d} / {N}")
print(f"  step=0.1 : {(x_step < 1).sum():>7d} / {N}")

# ---------- 3. 2x2 直方图 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle(f"Optuna 在 [0.1, 100] 上的采样分布对比 (N={N})", fontsize=14)

bins_log = np.logspace(-1, 2, 61)      # 对数等距 bin
bins_lin = np.linspace(0.1, 100, 61)   # 线性等距 bin

# 左上：log=True，对数横轴
axes[0, 0].hist(x_log, bins=bins_log, color="#3b82f6", edgecolor="black", alpha=0.85)
axes[0, 0].set_xscale("log")
axes[0, 0].set_title("log=True ｜ 对数横轴 → 几乎均匀")
axes[0, 0].set_xlabel("x (log scale)")
axes[0, 0].set_ylabel("count")

# 右上：step=0.1，对数横轴
axes[0, 1].hist(x_step, bins=bins_log, color="#f97316", edgecolor="black", alpha=0.85)
axes[0, 1].set_xscale("log")
axes[0, 1].set_title("step=0.1 ｜ 对数横轴 → 左疏右密")
axes[0, 1].set_xlabel("x (log scale)")
axes[0, 1].set_ylabel("count")

# 左下：log=True，线性横轴
axes[1, 0].hist(x_log, bins=bins_lin, color="#3b82f6", edgecolor="black", alpha=0.85)
axes[1, 0].set_title("log=True ｜ 线性横轴 → 左密右疏")
axes[1, 0].set_xlabel("x (linear scale)")
axes[1, 0].set_ylabel("count")

# 右下：step=0.1，线性横轴
axes[1, 1].hist(x_step, bins=bins_lin, color="#f97316", edgecolor="black", alpha=0.85)
axes[1, 1].set_title("step=0.1 ｜ 线性横轴 → 几乎均匀")
axes[1, 1].set_xlabel("x (linear scale)")
axes[1, 1].set_ylabel("count")

for ax in axes.flat:
    for v in (1, 10):   # 标出数量级分界
        ax.axvline(v, color="red", ls="--", lw=0.8, alpha=0.6)

plt.tight_layout()
plt.savefig("log_vs_step.png", dpi=130, bbox_inches="tight")
print("\n图已保存为 log_vs_step.png")
