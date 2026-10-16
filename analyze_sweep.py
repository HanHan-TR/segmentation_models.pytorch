#!/usr/bin/env python3
"""Analyze sweep results from all experiments under res/zhou-seg/"""

import os
import re
import yaml
import glob
from collections import defaultdict

RES_DIR = "/home/t_wanghan/work/segmentation_models.pytorch/res/zhou-seg"


def parse_config(config_path):
    """Parse wandb config.yaml to extract hyperparameters."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    params = {}
    for key in ['alpha', 'batch_size', 'decoder_lr', 'ema_decay',
                'encoder_lr_factor', 'epochs', 'loss_type', 'momentum',
                'optimizer', 'warmup_epochs_ratio', 'weight_decay']:
        if key in config:
            val = config[key]
            if isinstance(val, dict) and 'value' in val:
                params[key] = val['value']
            else:
                params[key] = val

    return params


def parse_output_log(log_path):
    """Parse output.log to extract best metrics for both ori and ema models."""
    with open(log_path, 'r') as f:
        content = f.read()

    # Find all validation metric blocks
    # Pattern for ori model macro metrics (macro row, NOT macro-imagewise)
    # "macro" row has 6 spaces padding, "macro-imagewise" only has 1 → use \s{2,} to distinguish
    ori_pattern = re.compile(
        r'Epoch (\d+) - ori Model Validation Metrics:.*?'
        r'macro\s{2,}\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)',
        re.DOTALL
    )

    ema_pattern = re.compile(
        r'Epoch (\d+) - ema Model Validation Metrics:.*?'
        r'macro\s{2,}\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)',
        re.DOTALL
    )

    # Also find micro metrics
    ori_micro_pattern = re.compile(
        r'Epoch (\d+) - ori Model Validation Metrics:.*?'
        r'micro\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)',
        re.DOTALL
    )

    ema_micro_pattern = re.compile(
        r'Epoch (\d+) - ema Model Validation Metrics:.*?'
        r'micro\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)',
        re.DOTALL
    )

    # Train loss pattern
    train_loss_pattern = re.compile(r'Train Epoch (\d+)/\d+:.*?train loss=([\d.]+)')

    # Find best epoch by macro dice (column index 4 in macro)
    # macro: accuracy, precision, recall, iou, dice, f2score

    ori_matches = ori_pattern.findall(content)
    ema_matches = ema_pattern.findall(content)
    ori_micro_matches = ori_micro_pattern.findall(content)
    ema_micro_matches = ema_micro_pattern.findall(content)
    train_losses = train_loss_pattern.findall(content)

    # Find best epoch for ori (by macro dice)
    best_ori = None
    best_ori_dice = -1
    for m in ori_matches:
        epoch = int(m[0])
        dice = float(m[4])
        if dice > best_ori_dice:
            best_ori_dice = dice
            best_ori = {
                'epoch': epoch,
                'accuracy': float(m[1]),
                'precision': float(m[2]),
                'recall': float(m[3]),
                'iou': float(m[4]),
                'dice': float(m[5]),
                'f2score': float(m[6]),
            }

    # Find best epoch for ema (by macro dice)
    best_ema = None
    best_ema_dice = -1
    for m in ema_matches:
        epoch = int(m[0])
        dice = float(m[4])
        if dice > best_ema_dice:
            best_ema_dice = dice
            best_ema = {
                'epoch': epoch,
                'accuracy': float(m[1]),
                'precision': float(m[2]),
                'recall': float(m[3]),
                'iou': float(m[4]),
                'dice': float(m[5]),
                'f2score': float(m[6]),
            }

    # Find best micro metrics for ori
    best_ori_micro = None
    best_ori_micro_dice = -1
    for m in ori_micro_matches:
        epoch = int(m[0])
        dice = float(m[4])
        if dice > best_ori_micro_dice:
            best_ori_micro_dice = dice
            best_ori_micro = {
                'epoch': epoch,
                'accuracy': float(m[1]),
                'precision': float(m[2]),
                'recall': float(m[3]),
                'iou': float(m[4]),
                'dice': float(m[5]),
                'f2score': float(m[6]),
            }

    # Find best micro metrics for ema
    best_ema_micro = None
    best_ema_micro_dice = -1
    for m in ema_micro_matches:
        epoch = int(m[0])
        dice = float(m[4])
        if dice > best_ema_micro_dice:
            best_ema_micro_dice = dice
            best_ema_micro = {
                'epoch': epoch,
                'accuracy': float(m[1]),
                'precision': float(m[2]),
                'recall': float(m[3]),
                'iou': float(m[4]),
                'dice': float(m[5]),
                'f2score': float(m[6]),
            }

    # Last train loss
    last_train_loss = float(train_losses[-1][1]) if train_losses else None
    first_train_loss = float(train_losses[0][1]) if train_losses else None

    return {
        'best_ori_macro': best_ori,
        'best_ema_macro': best_ema,
        'best_ori_micro': best_ori_micro,
        'best_ema_micro': best_ema_micro,
        'last_train_loss': last_train_loss,
        'first_train_loss': first_train_loss,
        'total_epochs_trained': len(ori_matches),
    }


def main():
    # Find all experiment dirs
    exp_dirs = sorted(glob.glob(os.path.join(RES_DIR, "*/")))

    results = []
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(os.path.dirname(exp_dir))
        # Find wandb run dir
        wandb_dirs = glob.glob(os.path.join(exp_dir, "wandb", "run-*"))
        if not wandb_dirs:
            print(f"WARNING: No wandb run found for {exp_name}")
            continue

        for run_dir in wandb_dirs:
            config_path = os.path.join(run_dir, "files", "config.yaml")
            log_path = os.path.join(run_dir, "files", "output.log")

            if not os.path.exists(config_path) or not os.path.exists(log_path):
                print(f"WARNING: Missing files for {run_dir}")
                continue

            params = parse_config(config_path)
            metrics = parse_output_log(log_path)

            results.append({
                'exp_name': exp_name,
                'run_dir': os.path.basename(run_dir),
                **params,
                **metrics,
            })

    # Sort by best ema macro dice
    results.sort(key=lambda x: (x.get('best_ema_macro') or {}).get('dice', 0), reverse=True)

    # Print summary
    print("=" * 200)
    print(f"{'Rank':<5} {'Experiment':<65} {'Opt':<7} {'BS':<5} {'LR':<10} {'EncLR':<8} {'EMA':<8} {'WD':<12} {'Mom':<8} {'Alpha':<8} {'Loss':<18} {'Ep':<5} {'Warm':<6} {'Best_EMA_Dice':<14} {'Best_EMA_IoU':<14} {'Best_Ori_Dice':<14} {'Best_Ori_IoU':<14} {'BestEp':<8} {'TrainLoss':<10}")
    print("=" * 200)

    for i, r in enumerate(results):
        ema_best = r.get('best_ema_macro') or {}
        ori_best = r.get('best_ori_macro') or {}

        loss_type_str = str(r.get('loss_type', 'N/A'))
        if isinstance(loss_type_str, list):
            loss_type_str = '+'.join(loss_type_str)

        print(f"{i+1:<5} {r['exp_name']:<65} {str(r.get('optimizer', 'N/A')):<7} {str(r.get('batch_size', 'N/A')):<5} {r.get('decoder_lr', 0):<10.6f} {r.get('encoder_lr_factor', 0):<8.4f} {r.get('ema_decay', 0):<8.4f} {r.get('weight_decay', 0):<12.8f} {r.get('momentum', 0):<8.4f} {r.get('alpha', 0):<8.4f} {loss_type_str:<18} {str(r.get('epochs', 'N/A')):<5} {str(r.get('warmup_epochs_ratio', 'N/A')):<6} {ema_best.get('dice', 0):<14.6f} {ema_best.get('iou', 0):<14.6f} {ori_best.get('dice', 0):<14.6f} {ori_best.get('iou', 0):<14.6f} {str(ema_best.get('epoch', 'N/A')):<8} {r.get('last_train_loss', 0) or 0:<10.4f}")

    print("=" * 200)

    # Summary statistics by hyperparameter
    print("\n\n========== ANALYSIS BY HYPERPARAMETER ==========\n")

    # Top 10 experiments
    print("\n--- Top 10 Experiments (by EMA macro Dice) ---")
    for i, r in enumerate(results[:10]):
        ema_best = r.get('best_ema_macro') or {}
        ori_best = r.get('best_ori_macro') or {}
        loss_type_str = str(r.get('loss_type', 'N/A'))
        if isinstance(loss_type_str, list):
            loss_type_str = '+'.join(loss_type_str)
        print(f"  Rank {i+1}: {r['exp_name']}")
        print(f"    Optimizer={r.get('optimizer')}, BS={r.get('batch_size')}, LR={r.get('decoder_lr'):.4f}, EncLR={r.get('encoder_lr_factor'):.4f}")
        print(f"    EMA={r.get('ema_decay'):.4f}, WD={r.get('weight_decay'):.8f}, Mom={r.get('momentum'):.4f}, Alpha={r.get('alpha'):.4f}")
        print(f"    Loss={loss_type_str}, Epochs={r.get('epochs')}, Warmup={r.get('warmup_epochs_ratio')}")
        print(f"    EMA Best: Dice={ema_best.get('dice'):.4f}, IoU={ema_best.get('iou'):.4f} @ Epoch {ema_best.get('epoch')}")
        print(f"    Ori  Best: Dice={ori_best.get('dice'):.4f}, IoU={ori_best.get('iou'):.4f} @ Epoch {ori_best.get('epoch')}")
        print()

    # Analysis by optimizer
    print("\n--- Performance by Optimizer ---")
    for opt in ['AdamW', 'SGD']:
        opt_results = [r for r in results if r.get('optimizer') == opt]
        if opt_results:
            ema_dices = [(r.get('best_ema_macro') or {}).get('dice', 0) for r in opt_results]
            ori_dices = [(r.get('best_ori_macro') or {}).get('dice', 0) for r in opt_results]
            print(f"  {opt}: count={len(opt_results)}, EMA Dice: avg={sum(ema_dices)/len(ema_dices):.4f}, max={max(ema_dices):.4f}, min={min(ema_dices):.4f}")
            print(f"        Ori Dice: avg={sum(ori_dices)/len(ori_dices):.4f}, max={max(ori_dices):.4f}, min={min(ori_dices):.4f}")

    # Analysis by loss type
    print("\n--- Performance by Loss Type ---")
    for lt in [['dice', 'bce'], ['focal', 'bce'], ['dice', 'focal']]:
        lt_str = '+'.join(lt)
        lt_results = [r for r in results if r.get('loss_type') == lt]
        if lt_results:
            ema_dices = [(r.get('best_ema_macro') or {}).get('dice', 0) for r in lt_results]
            ori_dices = [(r.get('best_ori_macro') or {}).get('dice', 0) for r in lt_results]
            print(f"  {lt_str}: count={len(lt_results)}, EMA Dice: avg={sum(ema_dices)/len(ema_dices):.4f}, max={max(ema_dices):.4f}, min={min(ema_dices):.4f}")
            print(f"          Ori Dice: avg={sum(ori_dices)/len(ori_dices):.4f}, max={max(ori_dices):.4f}, min={min(ori_dices):.4f}")

    # Analysis by batch_size
    print("\n--- Performance by Batch Size ---")
    for bs in [16, 32]:
        bs_results = [r for r in results if r.get('batch_size') == bs]
        if bs_results:
            ema_dices = [(r.get('best_ema_macro') or {}).get('dice', 0) for r in bs_results]
            ori_dices = [(r.get('best_ori_macro') or {}).get('dice', 0) for r in bs_results]
            print(f"  BS={bs}: count={len(bs_results)}, EMA Dice: avg={sum(ema_dices)/len(ema_dices):.4f}, max={max(ema_dices):.4f}, min={min(ema_dices):.4f}")
            print(f"          Ori Dice: avg={sum(ori_dices)/len(ori_dices):.4f}, max={max(ori_dices):.4f}, min={min(ori_dices):.4f}")

    # Analysis by epoch range
    print("\n--- Performance by Epochs ---")
    for ep in sorted(set(r.get('epochs') for r in results if r.get('epochs'))):
        ep_results = [r for r in results if r.get('epochs') == ep]
        if ep_results:
            ema_dices = [(r.get('best_ema_macro') or {}).get('dice', 0) for r in ep_results]
            ori_dices = [(r.get('best_ori_macro') or {}).get('dice', 0) for r in ep_results]
            print(f"  Epochs={ep}: count={len(ep_results)}, EMA Dice: avg={sum(ema_dices)/len(ema_dices):.4f}, max={max(ema_dices):.4f}, min={min(ema_dices):.4f}")
            print(f"            Ori Dice: avg={sum(ori_dices)/len(ori_dices):.4f}, max={max(ori_dices):.4f}, min={min(ori_dices):.4f}")

    # Correlation analysis for continuous parameters
    print("\n--- Parameter Range Analysis for Top 30% vs Bottom 30% ---")
    n = len(results)
    top_n = max(1, n // 3)
    bottom_n = max(1, n // 3)

    top_results = results[:top_n]
    bottom_results = results[-bottom_n:]

    for param in ['decoder_lr', 'encoder_lr_factor', 'ema_decay', 'weight_decay', 'momentum', 'alpha']:
        top_vals = [r.get(param) for r in top_results if r.get(param) is not None]
        bottom_vals = [r.get(param) for r in bottom_results if r.get(param) is not None]
        if top_vals and bottom_vals:
            print(f"  {param}:")
            print(f"    Top 1/3:    avg={sum(top_vals)/len(top_vals):.6f}, min={min(top_vals):.6f}, max={max(top_vals):.6f}")
            print(f"    Bottom 1/3: avg={sum(bottom_vals)/len(bottom_vals):.6f}, min={min(bottom_vals):.6f}, max={max(bottom_vals):.6f}")

    # Print all results for csv export
    print("\n\n========== CSV FORMAT ==========")
    print("rank,exp_name,optimizer,batch_size,decoder_lr,encoder_lr_factor,ema_decay,weight_decay,momentum,alpha,loss_type,epochs,warmup_epochs_ratio,ema_dice,ema_iou,ema_epoch,ori_dice,ori_iou,ori_epoch,train_loss")
    for i, r in enumerate(results):
        ema_best = r.get('best_ema_macro') or {}
        ori_best = r.get('best_ori_macro') or {}
        loss_type_str = str(r.get('loss_type', 'N/A'))
        if isinstance(loss_type_str, list):
            loss_type_str = '+'.join(loss_type_str)
        print(f"{i+1},{r['exp_name']},{r.get('optimizer')},{r.get('batch_size')},{r.get('decoder_lr'):.6f},{r.get('encoder_lr_factor'):.4f},{r.get('ema_decay'):.4f},{r.get('weight_decay'):.8f},{r.get('momentum'):.4f},{r.get('alpha'):.4f},{loss_type_str},{r.get('epochs')},{r.get('warmup_epochs_ratio')},{ema_best.get('dice',0):.6f},{ema_best.get('iou',0):.6f},{ema_best.get('epoch','N/A')},{ori_best.get('dice',0):.6f},{ori_best.get('iou',0):.6f},{ori_best.get('epoch','N/A')},{r.get('last_train_loss',0) or 0:.4f}")


if __name__ == '__main__':
    main()
