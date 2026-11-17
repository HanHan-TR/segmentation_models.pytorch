"""基于 Optuna 的超参数搜索空间工具。

支持两种配置写法，便于从 wandb sweep 平滑迁移到 Optuna：

1. wandb sweep 风格（已有 yaml 无需修改即可继续使用）::

    epochs:
      distribution: q_uniform
      min: 80
      max: 180
      q: 20

    batch_size:
      values: [16, 32]

    seed:
      value: 42

2. Optuna 原生风格::

    decoder_lr:
      type: float
      low: 0.01
      high: 0.1
      log: false

    loss_type:
      type: categorical
      choices: [[dice, bce], [focal, bce]]
"""

from typing import Any, Dict, List

import optuna

# wandb 的 distribution 名称 -> 统一的参数类型
_DISTRIBUTION_TYPE_MAP = {
    'uniform': 'float',
    'uniform_values': 'float',
    'q_uniform': 'float',
    'quantized_uniform': 'float',
    'log_uniform': 'float',
    'log_uniform_values': 'float',
    'q_log_uniform': 'float',
    'q_log_uniform_values': 'float',
    'int_uniform': 'int',
    'int': 'int',
    'q_int_uniform': 'int',
    'categorical': 'categorical',
}

_LOG_DISTRIBUTIONS = {'log_uniform', 'log_uniform_values', 'q_log_uniform', 'q_log_uniform_values'}
_QUANTIZED_DISTRIBUTIONS = {'q_uniform', 'quantized_uniform', 'q_log_uniform', 'q_log_uniform_values',
                            'q_int_uniform'}


def _is_hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _is_integral(value: Any) -> bool:
    """判断值是否为整数（int 或整数值的 float）；None 视为非整数。"""
    if value is None:
        return False
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def normalize_param_spec(spec: Any) -> Dict[str, Any]:
    """把单个参数的配置归一化成统一格式。

    统一后的 spec 一定是 dict，且 ``type`` 取值为
    ``fixed`` / ``categorical`` / ``float`` / ``int`` 之一。
    """
    if not isinstance(spec, dict):  # 直接写标量时视为固定值，例如 ``epochs: 120``
        return {'type': 'fixed', 'value': spec}

    spec = dict(spec)

    # ---------------- Optuna 原生写法 ----------------
    if 'type' in spec:
        param_type = str(spec.pop('type')).lower()
        if param_type in ('fixed', 'value'):
            return {'type': 'fixed', 'value': spec['value']}
        if param_type in ('categorical', 'cat'):
            return {'type': 'categorical', 'choices': list(spec['choices'])}
        if param_type in ('bool', 'boolean'):
            return {'type': 'categorical', 'choices': [True, False]}
        if param_type in ('float', 'int'):
            return {'type': param_type,
                    'low': spec.get('low', spec.get('min')),
                    'high': spec.get('high', spec.get('max')),
                    'log': bool(spec.get('log', False)),  # True表示取对数采样，例如 0.01~0.1 采样到 0.02, 0.03, 0.05, 0.08 等，而不是均匀采样
                    'step': spec.get('step', spec.get('q'))}
        raise ValueError(f"Unsupported param type: {param_type}, spec: {spec}")

    # ---------------- wandb sweep 写法 ----------------
    if 'value' in spec:
        return {'type': 'fixed', 'value': spec['value']}

    distribution = spec.get('distribution')
    if distribution is None:  # 只给了 values 或 min/max
        distribution = 'categorical' if 'values' in spec else 'uniform'
    distribution = str(distribution).lower()
    if distribution not in _DISTRIBUTION_TYPE_MAP:
        raise ValueError(f"Unsupported distribution: {distribution}, spec: {spec}")

    param_type = _DISTRIBUTION_TYPE_MAP[distribution]
    if param_type == 'categorical':
        return {'type': 'categorical', 'choices': list(spec['values'])}

    low = spec.get('min', spec.get('low'))
    high = spec.get('max', spec.get('high'))
    if low is None or high is None:
        raise ValueError(f"Missing min/max (low/high) for param, spec: {spec}")

    step = spec.get('q', spec.get('step')) if distribution in _QUANTIZED_DISTRIBUTIONS else None

    # q_uniform 的 min/max/q 若都是整数（例如 epochs: 80~180, q=20），
    # 说明该参数本质是整数（wandb 里对应 int_uniform），按 int 采样，
    # 否则 suggest_float 会返回 120.0 这类浮点值，无法直接用于 range()
    if param_type == 'float' and _is_integral(low) and _is_integral(high) and _is_integral(step):
        param_type, low, high, step = 'int', int(low), int(high), int(step)

    return {'type': param_type,
            'low': low,
            'high': high,
            'log': distribution in _LOG_DISTRIBUTIONS,
            'step': step}


def build_search_space(sweep_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """把 sweep yaml 的内容转换成统一的搜索空间描述。"""
    if not isinstance(sweep_cfg, dict) or len(sweep_cfg) == 0:
        raise ValueError('Sweep config must be a non-empty dict.')
    return {name: normalize_param_spec(spec) for name, spec in sweep_cfg.items()}


def suggest_hyperparams(trial: optuna.Trial, search_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """根据搜索空间为一个 trial 采样一组超参数。"""
    params = {}
    for name, spec in search_space.items():
        param_type = spec['type']

        if param_type == 'fixed':
            params[name] = spec['value']
        elif param_type == 'categorical':
            choices = spec['choices']
            if all(_is_hashable(choice) for choice in choices):
                params[name] = trial.suggest_categorical(name, choices)
            else:
                # Optuna 的 categorical 只接受可哈希的候选值（例如 loss_type 的 [dice, bce] 是列表），
                # 因此以候选值索引作为搜索维度，再映射回原始候选值，并记录到 user_attrs 方便回读。
                index = trial.suggest_categorical(f'{name}__index', list(range(len(choices))))
                params[name] = choices[index]
                trial.set_user_attr(name, choices[index])
        elif param_type == 'float':
            params[name] = trial.suggest_float(name,
                                               float(spec['low']),
                                               float(spec['high']),
                                               step=None if spec['log'] else spec['step'],
                                               log=spec['log'])
        elif param_type == 'int':
            # Optuna 的 suggest_int 不允许 log 与 step 同时使用
            params[name] = trial.suggest_int(name,
                                             int(spec['low']),
                                             int(spec['high']),
                                             step=None if spec['log'] else (int(spec['step']) if spec['step'] else 1),
                                             log=spec['log'])
        else:
            raise ValueError(f"Unsupported param type: {param_type} (param: {name})")
    return params


def get_trial_params(trial: optuna.Trial, search_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """从 trial 中还原完整的超参数（包含不可哈希候选值对应的原始取值）。"""
    params = {}
    for name, spec in search_space.items():
        if spec['type'] == 'fixed':
            params[name] = spec['value']
        elif name in trial.user_attrs:
            params[name] = trial.user_attrs[name]
        else:
            params[name] = trial.params[name]
    return params


def export_trials_dataframe(study: optuna.Study, search_space: Dict[str, Dict[str, Any]]):
    """导出 trial 结果表，并把不可哈希候选（如 ``loss_type``）还原成原始取值。

    ``study.trials_dataframe()`` 对列表形式的候选值只会导出
    ``params_loss_type__index`` 索引列，直接分析无法解读；这里把它替换成
    真实取值（例如 ``['dice', 'bce']``），与 wandb sweep 导出的口径保持一致。
    """
    df = study.trials_dataframe()
    for name, spec in search_space.items():
        index_col = f'params_{name}__index'
        if index_col not in df.columns or spec.get('choices') is None:
            continue
        choices = spec['choices']
        # 被剪枝/失败的 trial 该列为 NaN，用 index == index 过滤
        df[name] = df[index_col].apply(
            lambda index: str(choices[int(index)]) if index == index and index >= 0 else None)
        df = df.drop(columns=[index_col])
    return df


def _distribution_kind(dist: Any) -> str:
    """把 Optuna 的分布对象归类为统一的参数类型名称。"""
    if isinstance(dist, optuna.distributions.CategoricalDistribution):
        return 'categorical'
    if isinstance(dist, optuna.distributions.IntDistribution):
        return 'int'
    if isinstance(dist, optuna.distributions.FloatDistribution):
        return 'float'
    return 'unknown'


def find_incompatible_params(study: optuna.Study,
                             search_space: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """比较已有 study 中记录的参数分布与当前搜索空间，返回不兼容项。

    Optuna 不允许在同一个 study 内给同名参数更换分布类型（例如 ``epochs``
    从 ``FloatDistribution`` 改成 ``IntDistribution``）。当搜索空间 yaml 被调整后
    继续复用旧 study，会在采样阶段抛出
    ``ValueError: Cannot set different distribution kind to the same parameter name.``，
    且此时 trial 尚未开始训练，报错信息对使用者很不直观。这里提前检查，
    以便在 ``sweep_main`` 中给出可操作的提示。
    """
    recorded: Dict[str, Any] = {}
    for trial in study.trials:
        for name, dist in trial.distributions.items():
            recorded.setdefault(name, dist)

    mismatches = []
    for name, spec in search_space.items():
        if spec['type'] == 'fixed':  # 固定值不参与采样，没有记录分布
            continue
        dist = recorded.get(name)
        if dist is None:
            continue
        old_kind = _distribution_kind(dist)
        if old_kind != spec['type']:
            mismatches.append({'param': name, 'recorded': old_kind, 'current': spec['type']})
    return mismatches


def derive_epoch_metric_key(opt_metric: str) -> str:
    """把训练结束时的指标名映射为训练过程中每个 epoch 记录的同名指标键。

    例如 ``best_ema/metric(macro)/dice`` -> ``ema_metric(macro)/dice``，
    ``best_ema/cls_macro/c3`` -> ``ema_cls_metric(macro)/c3``。
    用于在每个 epoch 向 Optuna 上报中间结果，从而支持剪枝（pruning）。
    """
    if not opt_metric.startswith('best_'):
        return opt_metric

    model_type, metric_part = opt_metric[len('best_'):].split('/', 1)
    if metric_part == 'epoch':
        return 'epoch'
    if metric_part.startswith('cls_'):
        reduction, class_id = metric_part[len('cls_'):].split('/', 1)
        return f'{model_type}_cls_metric({reduction})/{class_id}'
    return f'{model_type}_{metric_part}'
