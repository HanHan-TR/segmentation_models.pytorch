import base64
import io
import shutil
import sys
import os
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from typing import Any, Dict, List, Tuple, Optional

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from UltraSeg.core.fileio import yaml_load


def rgb_label_to_binary_masks(rgb_label: np.ndarray, color_map: List[List[int]]) -> List[np.ndarray]:
    """
    将RGB彩色标签图像转换为每个类别对应的二值掩码列表。

    Args:
        rgb_label: RGB彩色标签图像，形状为 [H, W, 3]，RGB格式
        color_map: 颜色映射表，每个元素是 [R, G, B] 列表，索引对应类别编号

    Returns:
        binary_masks: 二值掩码列表，binary_masks[i] 对应类别 i
    """
    color_map_np = np.array(color_map, dtype=np.uint8)
    binary_masks = []

    for color in color_map_np:
        mask = np.all(rgb_label == color, axis=-1).astype(np.uint8)
        binary_masks.append(mask)

    return binary_masks


def check_single_connected_region(
    rgb_label: np.ndarray,
    color_map: List[List[int]],
    connectivity: int = 8,
    skip_background: bool = True,
    min_area: int = 0
) -> Tuple[bool, List[int], List[int]]:
    """
    判断RGB彩色标签图像中每种颜色的区域是否只有一个连通域。

    对于每个类别，检查其对应颜色的像素是否构成单一的连通区域。
    如果某个颜色出现两个或两个以上的独立区域，则说明标注错误。

    Args:
        rgb_label: RGB彩色标签图像，形状为 [H, W, 3]，RGB格式
        color_map: 颜色映射表，每个元素是 [R, G, B] 列表
        connectivity: 连通域分析的连接性，4 或 8
        skip_background: 是否跳过背景类（类别0），背景通常有多个不连通区域
        min_area: 最小连通域面积阈值，小于此面积的连通域将被忽略

    Returns:
        is_valid: 所有类别是否都只有一个连通域
        error_classes: 存在多个连通域的类别索引列表
        component_counts: 每个类别的连通域数量列表
    """
    color_map_np = np.array(color_map, dtype=np.uint8)
    num_classes = len(color_map_np)
    error_classes = []
    component_counts = [0] * num_classes

    start_idx = 1 if skip_background else 0

    for class_idx in range(start_idx, num_classes):
        color = color_map_np[class_idx]
        mask = np.all(rgb_label == color, axis=-1).astype(np.uint8)

        if mask.sum() == 0:
            component_counts[class_idx] = 0
            continue

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=connectivity
        )

        # num_labels 包含背景标签（label 0），实际前景连通域数量 = num_labels - 1
        foreground_components = num_labels - 1

        if min_area > 0:
            valid_components = 0
            for comp_id in range(1, num_labels):
                area = stats[comp_id, cv2.CC_STAT_AREA]
                if area >= min_area:
                    valid_components += 1
            foreground_components = valid_components

        component_counts[class_idx] = foreground_components

        if foreground_components > 1:
            error_classes.append(class_idx)

    is_valid = len(error_classes) == 0
    return is_valid, error_classes, component_counts


def clean_dataset_by_connected_components(
    config_path: str,
    split: str,
    connectivity: int = 8,
    skip_background: bool = True,
    min_area: int = 0,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    基于连通域分析清洗数据集。

    读取 img_dir/split_name 下的原始图片与 class_rgb_dir/split_name 下的RGB彩色标签图像，
    检查每种颜色是否只形成一个连通区域。如果某个颜色存在两个或以上的独立区域，
    则将对应的原始图片和标签图像移动到 mistake_split_name 文件夹。

    Args:
        config_path: 数据集配置文件路径（YAML格式）
        split: 数据集分割名称（如 'train'、'val' 等）
        connectivity: 连通域分析的连接性，4 或 8
        skip_background: 是否跳过背景类
        min_area: 最小连通域面积阈值（像素），小于此面积的碎片将被忽略
        dry_run: 如果为 True，仅进行检查而不实际移动文件

    Returns:
        result: 包含清洗结果信息的字典
    """
    dataset_cfg = yaml_load(config_path)

    data_root = dataset_cfg['data_root']
    img_dir_name = dataset_cfg['img_dir']
    class_rgb_dir_name = dataset_cfg['class_rgb_dir']
    color_map = dataset_cfg['color_map']
    classes = dataset_cfg.get('classes', None)
    img_suffix = dataset_cfg.get('img_suffix', '.png')

    img_dir = Path(data_root) / img_dir_name / split
    class_rgb_dir = Path(data_root) / class_rgb_dir_name / split

    start_time = time.time()
    if img_dir.exists() and class_rgb_dir.exists():
        # 创建 mistake 文件夹
        mistake_img_dir = Path(data_root) / img_dir_name / f'mistake_{split}'
        mistake_rgb_dir = Path(data_root) / class_rgb_dir_name / f'mistake_{split}'

        if not dry_run:
            mistake_img_dir.mkdir(parents=True, exist_ok=True)
            mistake_rgb_dir.mkdir(parents=True, exist_ok=True)

        # 收集所有图片文件
        img_paths = sorted(img_dir.glob(f'*{img_suffix}'))
        if len(img_paths) == 0:
            img_paths = sorted(img_dir.glob('*.*'))

        num_total = len(img_paths)

        print(f'图片目录: {img_dir}')
        print(f'RGB标签目录: {class_rgb_dir}')
        print(f'图像总数: {num_total}')
        print(f'类别数量: {len(color_map)}')
        if skip_background:
            print('跳过背景类（类别0）的连通域检查')
        if min_area > 0:
            print(f'最小连通域面积阈值: {min_area} 像素')
        if dry_run:
            print('[DRY RUN 模式] 仅检查，不移动文件')
        print()

        pbar = tqdm(img_paths, desc='检查标注')
        mistake_count = 0
        error_details = []

        for img_path in pbar:
            img_stem = img_path.stem
            class_rgb_path = class_rgb_dir / f'{img_stem}.png'

            if not class_rgb_path.exists():
                pbar.set_postfix_str('跳过: 缺少RGB标签')
                continue

            rgb_label = cv2.imread(str(class_rgb_path), cv2.IMREAD_COLOR)
            if rgb_label is None:
                pbar.set_postfix_str('跳过: 无法读取')
                continue

            # cv2 读取为 BGR，转换为 RGB 以与 color_map 匹配
            rgb_label = cv2.cvtColor(rgb_label, cv2.COLOR_BGR2RGB)

            is_valid, error_classes, component_counts = check_single_connected_region(
                rgb_label, color_map, connectivity, skip_background, min_area
            )

            if not is_valid:
                if classes is not None:
                    class_names = [classes[c] for c in error_classes]
                else:
                    class_names = [str(c) for c in error_classes]

                comp_info = {c: component_counts[c] for c in error_classes}
                pbar.set_postfix_str(f'错误: {class_names}')

                error_details.append({
                    'filename': img_path.name,
                    'error_classes': error_classes,
                    'class_names': class_names,
                    'component_counts': comp_info
                })

                if not dry_run:
                    shutil.move(str(img_path), str(mistake_img_dir / img_path.name))
                    shutil.move(str(class_rgb_path), str(mistake_rgb_dir / class_rgb_path.name))

                mistake_count += 1
    else:
        print(f'图片目录或RGB标签目录不存在: {img_dir} 或 {class_rgb_dir}')
        num_total = 0
        mistake_count = 0

    # 输出统计结果
    elapsed_time = time.time() - start_time
    print(f'\n{"=" * 60}')
    print('数据清洗完成')
    print(f'总图像数: {num_total}')
    print(f'错误图像数: {mistake_count}')
    print(f'正确图像数: {num_total - mistake_count}')
    print(f'错误率: {mistake_count / num_total * 100:.2f}%' if num_total > 0 else '错误率: N/A')
    print(f'耗时: {elapsed_time:.1f} 秒')
    print(f'{"=" * 60}')

    if error_details:
        print('\n错误详情:')
        for detail in error_details:
            print(f'  {detail["filename"]}:')
            for cls_name, comp_count in zip(detail['class_names'], detail['component_counts'].values()):
                print(f'    {cls_name} -> {comp_count} 个连通域')

    if not dry_run and mistake_count > 0:
        print('\n错误文件已移动到:')
        print(f'  图片: {mistake_img_dir}')
        print(f'  标签: {mistake_rgb_dir}')

    return {
        'mistake_count': mistake_count,
        'num_total': num_total,
        'error_details': error_details,
        'img_dir': img_dir,
        'class_rgb_dir': class_rgb_dir,
        'config_path': config_path,
        'split': split,
        'connectivity': connectivity,
        'skip_background': skip_background,
        'min_area': min_area,
        'dry_run': dry_run,
        'elapsed_time': elapsed_time,
        'num_classes': len(color_map),
        'classes': classes,
    }


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f'{minutes}:{secs:02d}'
    return f'{secs} 秒'


def _generate_blended_thumbnail(
    orig_path: Path, label_path: Path, thumb_width: int = 200, alpha: float = 0.6
) -> str:
    orig = Image.open(str(orig_path)).convert('RGB')
    label = Image.open(str(label_path)).convert('RGB')

    orig_w, orig_h = orig.size

    label_resized = label.resize((orig_w, orig_h), Image.LANCZOS)

    blended = Image.blend(orig, label_resized, alpha=alpha)

    thumb_h = int(orig_h * thumb_width / orig_w)
    blended_thumb = blended.resize((thumb_width, thumb_h), Image.LANCZOS)

    buf = io.BytesIO()
    blended_thumb.save(buf, format='PNG', optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{b64}'


def _load_report_css() -> str:
    css_path = FILE.parent / 'report_style.css'
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f'警告: 无法加载 CSS 文件 {css_path}: {e}')
        return ''


def export_html_report(
    report_path: str,
    results: List[Dict[str, Any]],
    title: str = '数据清洗报告',
) -> None:
    THUMB_WIDTH = 200
    BLEND_ALPHA = 0.6

    for result in results:
        result['_base64_map'] = {}
        for detail in result['error_details']:
            fname = detail['filename']
            img_dir = result['img_dir']
            class_rgb_dir = result['class_rgb_dir']
            orig_path = img_dir / fname
            label_stem = Path(fname).stem
            label_path = class_rgb_dir / f'{label_stem}.png'  # 标签图像的格式为 png 格式
            split = result['split']
            dry_run = result['dry_run']
            if not dry_run:  # 非 dry_run 模式，即已经移动了错误文件到 mistake 文件夹
                mistake_img_dir = img_dir.parent / f'mistake_{split}'
                mistake_rgb_dir = class_rgb_dir.parent / f'mistake_{split}'
                if not orig_path.exists():  # 检查图像是否已经在原始路径中不存在，不存在则从 mistake 文件夹中获取
                    orig_path = mistake_img_dir / fname
                if not label_path.exists():  # 检查标签图像是否已经在原始路径中不存在，不存在则从 mistake 文件夹中获取
                    label_path = mistake_rgb_dir / f'{label_stem}.png'
            if orig_path.exists() and label_path.exists():
                try:
                    b64 = _generate_blended_thumbnail(
                        orig_path, label_path, THUMB_WIDTH, BLEND_ALPHA
                    )
                    result['_base64_map'][fname] = b64
                except Exception as e:
                    print(f'警告: 无法生成缩略图 {fname}: {e}')

    css = _load_report_css()

    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{css}</style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
""")

    for idx, result in enumerate(results):
        split = result['split']
        config_path = result['config_path']
        img_dir = result['img_dir']
        class_rgb_dir = result['class_rgb_dir']
        num_total = result['num_total']
        num_classes = result['num_classes']
        mistake_count = result['mistake_count']
        # connectivity = result['connectivity']
        skip_background = result['skip_background']
        min_area = result['min_area']
        dry_run = result['dry_run']
        elapsed_time = result['elapsed_time']
        error_details = result['error_details']
        # classes = result['classes']
        base64_map = result['_base64_map']

        error_rate = f'{mistake_count / num_total * 100:.2f}%' if num_total > 0 else 'N/A'
        dry_run_text = '仅检查，不移动文件' if dry_run else '正常模式，移动错误文件'
        bg_text = '是' if skip_background else '否'

        html_parts.append(f"""
        <div class="section">
            <div class="section-title">配置参数 - {split}</div>
            <table>
                <tr><td class="key-col">配置文件</td><td class="value-col"><code>{config_path}</code></td></tr>
                <tr><td class="key-col">图片目录</td><td class="value-col"><code>{img_dir}</code></td></tr>
                <tr><td class="key-col">RGB标签目录</td><td class="value-col"><code>{class_rgb_dir}</code></td></tr>
                <tr><td class="key-col">图像总数</td><td class="value-col"><strong>{num_total}</strong></td></tr>
                <tr><td class="key-col">类别数量</td><td class="value-col"><strong>{num_classes}</strong></td></tr>
                <tr><td class="key-col">跳过背景类（类别0）的连通域检查</td><td class="value-col">{bg_text}</td></tr>
                <tr><td class="key-col">最小连通域面积阈值</td><td class="value-col"><strong>{min_area}</strong> 像素</td></tr>
                <tr><td class="key-col">运行模式</td><td class="value-col"><span class="badge">{dry_run_text}</span></td></tr>
            </table>
        </div>
""")

        progress_pct = 100.0
        time_str = _format_time(elapsed_time)
        error_class_examples = set()
        for detail in error_details:
            for cn in detail['class_names']:
                error_class_examples.add(cn)
        error_class_str = ', '.join(sorted(error_class_examples)[:5])

        html_parts.append(f"""
        <div class="section">
            <div class="section-title">统计摘要 - {split}</div>
            <div class="stat-grid">
                <div class="stat-item"><span class="stat-label">总图像数：</span><span class="stat-number">{num_total}</span></div>
                <div class="stat-item"><span class="stat-label">错误图像数：</span><span class="stat-number error">{mistake_count}</span></div>
                <div class="stat-item"><span class="stat-label">正确图像数：</span><span class="stat-number success">{num_total - mistake_count}</span></div>
                <div class="stat-item"><span class="stat-label">错误率：</span><span class="stat-number rate">{error_rate}</span></div>
            </div>
            <div style="margin-top: 12px; font-size: 14px; color: #475569;">
                <span class="badge" style="background:#f1f5f9; color:#334155;">检查进度</span> {progress_pct:.0f}% ({num_total}/{num_total})
                &nbsp;|&nbsp; 耗时 {time_str}
                &nbsp;|&nbsp; 错误类别示例: {error_class_str if error_class_str else '无'}
            </div>
            <div class="note-box">
                <strong>错误检查方法：</strong><br>
                默认每种肌骨类型在一张超声图像中仅存在至多一个独立的区域。因此，标注错误的检查方法为：判断 RGB
                彩色标签中的每种颜色的区域是不是只有一个，如果出现两个或两个以上的独立区域具有相同的颜色，则说明该类别标注错误（例如出现多个连通域）。
            </div>
        </div>
""")

        html_parts.append(f"""
        <div class="section">
            <div class="section-title">错误详情 - {split}（共 {mistake_count} 项）</div>
            <table class="error-table" id="error-table-{split}">
                <thead>
                    <tr>
                        <th style="width:25%;">图像文件名</th>
                        <th style="width:20%;">错误类别</th>
                        <th style="width:15%;">描述</th>
                        <th style="width:40%;">图像预览（原图 + 标签叠加）</th>
                    </tr>
                </thead>
                <tbody>
""")

        for detail in error_details:
            fname = detail['filename']
            class_names = detail['class_names']
            comp_counts = detail['component_counts']
            category_str = ', '.join(class_names)
            desc_str = ', '.join(f'{comp_counts[c]} 个连通域' for c in detail['error_classes'])
            html_parts.append(f"""
                    <tr>
                        <td class="filename">{fname}</td>
                        <td class="category">{category_str}</td>
                        <td class="desc">{desc_str}</td>
                    </tr>
""")

        html_parts.append("""
                </tbody>
            </table>
        </div>
""")

        if base64_map:
            html_parts.append("""
    <script>
(function() {
    var base64Map = {
""")
            for fname, b64 in sorted(base64_map.items()):
                html_parts.append(f'        "{fname}": "{b64}",\n')

            html_parts.append(f"""    }};
    var rows = document.querySelectorAll('#error-table-{split} tbody tr');
    rows.forEach(function(row) {{
        var filenameCell = row.querySelector('.filename');
        if (!filenameCell) return;
        var filename = filenameCell.textContent.trim();
        var b64 = base64Map[filename];
        if (!b64) return;
        var td = document.createElement('td');
        td.style.textAlign = 'center';
        td.style.padding = '4px 8px';
        var img = document.createElement('img');
        img.src = b64;
        img.style.maxWidth = '{THUMB_WIDTH}px';
        img.style.borderRadius = '4px';
        img.style.border = '1px solid #e2e8f0';
        td.appendChild(img);
        row.appendChild(td);
    }});
}})();
</script>
""")

    html_parts.append("""
    </div>
</body>
</html>
""")

    html_content = ''.join(html_parts)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    size_mb = os.path.getsize(report_path) / (1024 * 1024)
    print(f'\nHTML报告已生成: {report_path}')
    print(f'文件大小: {size_mb:.2f} MB')


def main():
    parser = argparse.ArgumentParser(
        description='数据清洗脚本：基于连通域分析检测标注错误'
    )
    parser.add_argument(
        '-cfg', '--config', type=str, default='UltraSeg/config/dataset/wanguan/wan_guanBG.yaml',
        help='数据集配置文件路径'
    )
    parser.add_argument('-s', '--split',
                        type=str,
                        nargs='+',
                        default=['all', 'train', 'val'],
                        help='需要清洗的数据集的分割部分，如：all表示全部数据集，train表示训练集。可以指定多个部分的数据集')
    parser.add_argument(
        '--connectivity', type=int, default=8, choices=[4, 8],
        help='连通域分析的连接性（4 或 8），默认 8'
    )
    parser.add_argument(
        '--skip-background', action='store_true', default=True,
        help='跳过背景类（类别0）的连通域检查（默认开启）'
    )
    parser.add_argument(
        '--no-skip-background', action='store_false', dest='skip_background',
        help='不跳过背景类，对所有类别进行检查'
    )
    parser.add_argument(
        '--min-area', type=int, default=10,
        help='最小连通域面积阈值（像素），小于此面积的碎片将被忽略，默认 10'
    )
    parser.add_argument(
        '-dr', '--dry-run', action='store_true', default=False,
        help='仅检查而不实际移动文件'
    )
    parser.add_argument('-rep', '--report_name', type=str, default='踝部数据集清洗报告',
                        help='报告文件名，默认 腕部数据集清洗报告')
    parser.add_argument('--html', action='store_true', default=False,
                        help='生成HTML报告，默认不生成')
    args = parser.parse_args()

    config_path = args.config

    if not Path(config_path).exists():
        print(f'配置文件不存在: {config_path}')
        sys.exit(1)

    print(f'配置文件: {config_path}\n')

    splits = args.split
    all_results = []

    for s in splits:
        result = clean_dataset_by_connected_components(
            config_path=config_path,
            split=s,
            connectivity=args.connectivity,
            skip_background=args.skip_background,
            min_area=args.min_area,
            dry_run=args.dry_run
        )
        all_results.append(result)

    if args.html and all_results:
        report_path = f'{args.report_name}.html'
        export_html_report(report_path, all_results, title=f'{args.report_name}')


if __name__ == '__main__':
    main()
