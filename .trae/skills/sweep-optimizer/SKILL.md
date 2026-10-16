---
name: "sweep-optimizer"
description: "Analyze W&B sweep results to narrow hyperparameter search ranges. Invoke when user wants to refine sweep config based on completed experiments, optimize parameter ranges, or analyze sweep outcomes."
---

# Sweep Optimizer

This skill systematically analyzes results from completed W&B sweep experiments and provides data-driven recommendations for narrowing hyperparameter search ranges. It is designed for iterative sweep optimization — each round narrows the search space based on measured performance.

## When to Use

Invoke this skill when the user:
- Has completed a round of sweep experiments and wants to narrow the search space
- Wants to analyze which hyperparameters matter most
- Asks to "optimize sweep ranges", "narrow sweep parameters", "refine sweep config"
- Wants a data-driven analysis of sweep results to guide the next round

## Required Inputs

The user must provide (or the agent must determine):

1. **Sweep config file** (YAML) — the wandb sweep configuration used for the experiments
2. **Metric specification** — how to identify the best model. Format: `<model_type>|<reduction>|<metric_name>`
   - `<model_type>`: `ori` or `ema`
   - `<reduction>`: `micro`, `macro`, `weighted`, or `macro-imagewise`
   - `<metric_name>`: `dice`, `iou`, `accuracy`, `precision`, `recall`, `f2score`
   - Example: `ema|macro|dice` means "best model is the one with highest EMA model macro Dice"
3. **Results directory** — path to the folder containing experiment subdirectories, each with `wandb/latest-run/files/output.log` and `wandb/latest-run/files/config.yaml`

## Workflow

### Phase 1: Understand the Sweep Config

Read the sweep YAML config file to extract all hyperparameters and their original ranges/distributions. Key parameters typically include:

| Parameter | Type | Meaning |
|-----------|------|---------|
| `optimizer` | categorical | [AdamW, SGD] |
| `batch_size` | categorical | [16, 32] |
| `loss_type` | categorical | [[dice,bce], [focal,bce], [dice,focal]] |
| `epochs` | q_uniform | training epochs |
| `decoder_lr` | uniform | decoder learning rate |
| `encoder_lr_factor` | uniform | encoder LR = decoder_lr × factor |
| `ema_decay` | q_uniform | EMA decay rate |
| `momentum` | uniform | SGD momentum |
| `weight_decay` | uniform | weight decay |
| `alpha` | uniform | loss weighting coefficient |
| `warmup_epochs_ratio` | categorical | warmup ratio |

### Phase 2: Parse All Experiment Results

For each experiment subdirectory under the results directory:
1. Read `wandb/latest-run/files/config.yaml` to extract hyperparameter values
2. Read `wandb/latest-run/files/output.log` to extract validation metrics

**Parsing output.log:**

The output.log format varies across projects. The key task is to locate and extract the user-specified metric from the log for each epoch. Follow this general approach:

**Step 1 — Explore the format.** First read one experiment's output.log to understand how metrics are logged. Common patterns include:
- ASCII tables with multi-reduction rows (micro/macro/weighted/macro-imagewise)
- Single-line key-value logs like `Epoch 10: val_loss=0.123, val_dice=0.456`
- JSON or structured log lines
- Mixed formats with different sections for different model types (e.g., `ori` vs `ema`)

**Step 2 — Identify the target metric.** Based on the user's metric specification `<model_type>|<reduction>|<metric_name>`:
- Locate the section that corresponds to `<model_type>` (e.g., look for lines containing "ori" or "ema")
- Within that section, locate the row/key that matches `<reduction>` (e.g., "macro", "micro", "macro-imagewise")
- Extract the value for `<metric_name>` (e.g., "dice", "iou", "accuracy")

**Step 3 — Adapt the extraction strategy to the format:**

| Log Format | Extraction Strategy |
|------------|-------------------|
| **ASCII table** | Use regex to match the reduction row and capture column values. Be careful with name collisions (e.g., `macro` vs `macro-imagewise` — use word boundaries or spacing patterns to distinguish). |
| **Key-value lines** | Use regex like `Epoch (\d+).*?val_(\w+)=([\d.]+)` to capture epoch and metric values. |
| **JSON lines** | Parse each line as JSON, extract fields by key. |
| **Mixed sections** | Use section markers (e.g., `=== Validation ===`) to locate the right block, then apply the appropriate extraction within that block. |

**Step 4 — Find the best epoch.** For each experiment, scan all epochs and find the one that maximizes (or minimizes, depending on the metric) the target metric value. Record:
- The best epoch number
- All metric values at that epoch (for richer analysis)
- The last train loss (for convergence assessment)

**Example — ASCII table format (common in segmentation projects):**
```
Epoch N - ori Model Validation Metrics:
+-------+----------+-----------+--------+--------+--------+---------+
|  red  | accuracy | precision | recall |  iou   |  dice  | f2score |
+-------+----------+-----------+--------+--------+--------+---------+
| micro |  0.9908  |   0.9129  | 0.9129 | 0.8397 | 0.9129 |  0.9129 |
| macro |  0.9908  |   0.5902  | 0.1579 | 0.1007 | 0.1029 |  0.1043 |
+-------+----------+-----------+--------+--------+--------+---------+
```
For this format, use `macro\s{2,}\|` (not `macro\s*\|`) to match the "macro" row without matching "macro-imagewise".

### Phase 3: Build Ranking

Build a ranked table of all experiments, sorted by the specified metric (descending). For each experiment, include:
- Rank, experiment name
- All hyperparameter values
- Best EMA metric value and epoch
- Best Ori metric value and epoch
- Last train loss

### Phase 4: Analyze by Hyperparameter

For each hyperparameter, analyze:

**Categorical parameters** (optimizer, batch_size, loss_type, warmup_ratio):
- Group results by each value
- Compute average, max, min of the target metric
- Determine if one value is clearly superior

**Discrete numerical parameters** (epochs):
- Group by value
- Compute average, max, min
- Check if the best metric epoch suggests sufficient training length

**Continuous parameters** (decoder_lr, encoder_lr_factor, ema_decay, weight_decay, momentum, alpha):
- Compare Top 1/3 vs Bottom 1/3:
  - Split results into top 1/3 and bottom 1/3 by the target metric
  - Compute average, min, max for each group
  - Identify parameters where the top group has significantly different values
- Examine the range of values in the Top 10 experiments
- Check if extreme values (very low or very high) correlate with poor performance

### Phase 5: Generate Recommendations

For each parameter, recommend one of:

| Action | Criteria | Example |
|--------|----------|---------|
| **Drop value** | A categorical value is consistently worse | Remove AdamW if SGD dominates top 5 |
| **Narrow range** | Extreme values of a continuous parameter perform poorly | momentum: 0.85→0.99 → 0.92→0.99 |
| **Fix value** | One value is clearly best and sampling is wasteful | Fix batch_size=32 |
| **Keep** | No clear pattern or sufficient diversity needed | Keep alpha: 0.25-0.75 |

**Key analysis principles:**
1. Look at BOTH average and maximum — a parameter might have similar average but one variant produces the best model
2. Check the Top 10 experiments for the "winning recipe" of parameter combinations
3. Consider interactions: e.g., SGD benefits from high momentum, AdamW doesn't use momentum
4. Assess whether the best metric epoch is near the end of training — if so, more epochs may help
5. The "macro" reduction provides better discrimination than "macro-imagewise" for imbalanced datasets

### Phase 6: Output the Optimized Sweep Config

Produce a YAML config snippet with the recommended narrowed ranges. Each parameter should include a comment explaining the rationale.

Also provide a summary table:

| Parameter | Original Range | Recommended Range | Rationale |
|-----------|---------------|-------------------|-----------|
| ... | ... | ... | ... |

## Important Notes

- **Always read the actual sweep config** — never assume parameter names or ranges
- **Use the user-specified metric** — don't default to any particular reduction or model type
- **Always explore the output.log format first** — read a sample output.log before writing any parsing code. Different projects log metrics in different ways (tables, key-value lines, JSON, etc.)
- **Be careful with name collisions in regex** — e.g., when matching `macro`, ensure it doesn't accidentally match `macro-imagewise` (use word boundaries or spacing patterns)
- **Config values may be nested** — in wandb config.yaml, values are often under a `value` key: `{"value": 0.001}`
- **When suggesting range narrowing**, estimate search space reduction to help the user understand the benefit
- **Present the analysis in the user's language** — match the language of the conversation