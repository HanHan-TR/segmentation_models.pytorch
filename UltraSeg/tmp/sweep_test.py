# Import the W&B Python Library and log into W&B
import wandb

# 1: Define objective/training function


def objective(config):
    for epoch in range(5):
        for batch in range(10):
            loss = config.x**2 + config.y**2
            wandb.log({"loss": loss, "batch": batch, "epoch": epoch})
    score = config.x**3 + config.y
    acc = config.x**2 + config.y**2
    recall = config.x**3 + config.y

    wandb.log({"metric": {"acc": acc, "recall": recall}})
    return score


def main():
    with wandb.init(project="my-first-sweep3") as run:
        score = objective(run.config)
        run.log({"score": score})


# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# 3: Start the sweep

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep3")
    wandb.agent(sweep_id, function=main, count=10)
