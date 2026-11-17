"""Optuna + SwanLab 冒烟测试：搜索 x、y 两个超参数，并把指标记录到 SwanLab。

用法（在 ultraseg 环境、已 swanlab login 的前提下）::

    python UltraSeg/tmp/sweep_test.py
"""

import optuna
import swanlab

PROJECT = "test-project"
STUDY_NAME = "optuna-swanlab-smoke-test"


def objective(trial: optuna.Trial):
    x = trial.suggest_float("x", 0.01, 0.1)
    y = trial.suggest_categorical("y", [1, 3, 7])

    swanlab.init(project=PROJECT,
                 name=f"{STUDY_NAME}-trial{trial.number}",
                 config={"x": x, "y": y, "trial_number": trial.number},
                 reinit=True)
    try:
        for epoch in range(5):
            swanlab.log({"loss": x ** 2 + y ** 2}, step=epoch)
        acc = x ** 2 + y ** 2
        recall = x ** 3 + y
        swanlab.log({"metric/acc": acc, "metric/recall": recall})
        score = x ** 3 + y
        swanlab.log({"score": score})
    except Exception as error:
        swanlab.finish(state="crashed", error=str(error))
        raise
    swanlab.finish()
    return score


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, load_if_exists=True)
    study.optimize(objective, n_trials=10)
    print(f"best trial: {study.best_trial.number}, score={study.best_value:.6f}")
    print(f"best params: {study.best_params}")
