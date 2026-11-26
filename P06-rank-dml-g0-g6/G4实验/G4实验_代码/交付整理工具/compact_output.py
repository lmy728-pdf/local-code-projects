"""G6 delivery: one full CSV, one brief report, four vector PDF figures."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_tables(summary, source, axis, value, laws):
    frames = []
    for path in sorted(summary.glob('*.csv')):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        if 'law' in df:
            df = df[df.law.isin(laws)]
        if 'n' in df:
            df = df[df.n == 1000]
        if {'d', 'rho'}.issubset(df.columns):
            df = df[(df.d == 1) | ((df.d == 3) & (df.rho == .5))]
        if not df.empty:
            frames.append(df.assign(table=path.stem, source=source, **{axis: value}))
    return frames


def collect_comparison(work, archive, group):
    is_a = group == 'G6A'
    axis = 'tau' if is_a else 'c'
    laws = ['sln100'] if is_a else ['sln075', 'sln100']
    frames = []
    for baseline in ('g1_scalar_n1000', 'g2_vector_n1000'):
        summary = archive / 'runs' / baseline / 'summary'
        if not (summary / 'coordinates.csv').exists():
            raise FileNotFoundError(summary / 'coordinates.csv')
        frames.extend(read_tables(summary, baseline, axis, .2 if is_a else 1., laws))
    stages = [('g6a_tau050', .5)] if is_a else [('g6b_c0.5', .5), ('g6b_c2.0', 2.)]
    for stage, value in stages:
        folder = work / stage
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        if config['status'] != 'completed':
            raise RuntimeError(f'{stage}: complete both G6B steps before final delivery')
        frames.extend(read_tables(folder / 'summary', stage, axis, value,
                                  ['sln100_tau050'] if is_a else laws))
    return pd.concat(frames, ignore_index=True, sort=False)


def draw_core_figures(comparison, plots, group):
    # deliver configures the non-interactive backend and Chinese font first.
    from deliver import plt, METHODS, COLORS
    plots.mkdir(parents=True, exist_ok=True)
    is_a = group == 'G6A'
    axis = 'tau' if is_a else 'c'
    ticks = [.2, .5] if is_a else [.5, 1., 2.]
    laws = ['sln100'] if is_a else ['sln075', 'sln100']
    learned = comparison[comparison.pipeline == 'learned'].copy()
    learned['plot_law'] = learned.law.replace({'sln100_tau050': 'sln100'})
    specs = [
        ('01_scalar_RMSE.pdf', 'coordinates', 1, 'RMSE', '标量 RMSE'),
        ('02_scalar_coverage.pdf', 'coordinates', 1, 'coverage', '标量区间覆盖率'),
        ('03_vector_RMSE.pdf', 'vectors', 3, 'vector_RMSE', '多元整体 RMSE'),
        ('04_joint_coverage.pdf', 'vectors', 3, 'coverage', '多元联合覆盖率'),
    ]
    index = []
    for filename, table, dimension, metric, title in specs:
        fig, axes = plt.subplots(1, len(laws), figsize=(11, 8) if is_a else (16, 8),
                                 squeeze=False, sharey=True)
        fig.subplots_adjust(left=.10, right=.97, bottom=.19, top=.76, wspace=.18)
        values_for_limits = []
        for ax, law in zip(axes[0], laws):
            selected = learned[(learned.table == table) & (learned.d == dimension)
                               & (learned.plot_law == law)]
            for method, color in zip(METHODS, COLORS):
                q = selected[selected.method == method].sort_values(axis)
                if len(q) != len(ticks) or not np.allclose(q[axis].to_numpy(), ticks):
                    raise ValueError(f'Incomplete or duplicate comparison: {group}/{law}/{table}/{method}')
                y = q[metric].to_numpy(dtype=float)
                error = 1.96 * q.coverage_mcse.to_numpy(dtype=float) if metric == 'coverage' else None
                ax.errorbar(np.arange(len(ticks)), y, yerr=error, fmt='o-', color=color,
                            label=method, capsize=4, markersize=7, linewidth=2)
                values_for_limits.extend((y - error if error is not None else y).tolist())
            ax.set_xticks(np.arange(len(ticks)), [f'{v:g}' for v in ticks])
            ax.set_xlim(-.15, len(ticks) - .85)
            ax.set_xlabel('平滑参数 τ' if is_a else '步长系数 c', labelpad=12)
            ax.set_title('平滑 lognormal，σ=' + ('0.75' if law == 'sln075' else '1'), pad=16)
            ax.grid(axis='y', alpha=.22)
            if metric == 'coverage':
                ax.axhline(.95, color='#b91c1c', linestyle='--', linewidth=1.5)
        finite = [v for v in values_for_limits if np.isfinite(v)]
        if metric == 'coverage':
            axes[0, 0].set_ylim(max(0, min(.8, min(finite, default=.8) - .03)), 1.04)
        else:
            axes[0, 0].set_ylim(bottom=0)
        axes[0, 0].set_ylabel(title, labelpad=12)
        fig.suptitle(f'{group} · {title}', fontsize=23, y=.96)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .91),
                   ncol=5, frameon=False)
        note = 'learned 流程；n=1000；每条件 R=1000。'
        note += (' 红虚线：95%；误差条：±1.96 Monte Carlo SE；无效区间计未覆盖。'
                 if metric == 'coverage' else ' RMSE 按有效估计计算；完整指标与 Oracle 见 CSV。')
        fig.text(.5, .055, note, ha='center', fontsize=11)
        if not is_a:
            fig.text(.5, .025, 'c=1 复用 G1/G2；LS/H/HT 不受 c 影响，作为基准保留。', ha='center', fontsize=11)
        fig.savefig(plots / filename)
        plt.close(fig)
        index.append((filename, title))
    (plots / '图表索引.md').write_text(
        f'# {group} 核心图（4 张 PDF 矢量图）\n\n' +
        '\n'.join(f'- [{title}](<{filename}>)' for filename, title in index) +
        '\n\n仅展示 learned 流程；全部方法、Oracle、逐坐标、contrast 和配对比较保留于汇总 CSV。\n', encoding='utf-8')
    return index


def export_compact(folder, target, stage, archive):
    from deliver import audit_records
    folder, target, archive = map(Path, (folder, target, archive))
    group = 'G6A' if stage == 'g6a_tau050' else 'G6B'
    if stage not in ('g6a_tau050', 'g6b_c0.5', 'g6b_c2.0'):
        raise ValueError(stage)
    audit = audit_records(folder)
    audit['failure_counts'] = json.loads((folder / 'summary' / 'failures.json').read_text(encoding='utf-8'))
    audit_path = folder / 'delivery_audit.json'
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    if stage == 'g6b_c0.5':
        print('G6B c=0.5 saved; four combined figures will be exported after c=2.', flush=True)
        return
    comparison = collect_comparison(folder.parent, archive, group)
    tables, results, plots = (target / name for name in ('汇总表_CSV', '结果_MD', '图'))
    for path in (tables, results, plots):
        path.mkdir(parents=True, exist_ok=True)
    csv = tables / '汇总表.csv'
    comparison.to_csv(csv, index=False, encoding='utf-8-sig')
    index = draw_core_figures(comparison, plots, group)
    stages = ['g6a_tau050'] if group == 'G6A' else ['g6b_c0.5', 'g6b_c2.0']
    checks = []
    for name in stages:
        check = json.loads((folder.parent / name / 'delivery_audit.json').read_text(encoding='utf-8'))
        checks.append(f'- {name}：{check["datasets"]} 份记录，缓存复算 {check["replayed_records"]} 条通过；失败分类：{check["failure_counts"]}。')
    description = ('τ=0.2（G1/G2）与 τ=0.5 对照；平滑 lognormal σ=1。'
                   if group == 'G6A' else 'c=0.5、1、2 合并对照；平滑 lognormal σ=0.75、1，c=1 复用 G1/G2。')
    report = (f'# {group} 结果\n\n{description} n=1000，每条件 R=1000；含标量和三维（ρ=0.5）实验。\n\n'
              '默认仅输出 4 张核心 PDF，展示 LS/H/HT/W/N 的 learned 结果。覆盖率图含 95% 参考线和 Monte Carlo 误差条。\n\n'
              '[完整汇总 CSV](../汇总表_CSV/汇总表.csv) · [4 张核心图索引](../图/图表索引.md)\n\n'
              'CSV 用 table、source 和参数列区分指标与来源，保留 Oracle、逐坐标、contrast、配对比较和耗时。'
              'RMSE 按有效估计计算；覆盖率将无效区间计为未覆盖，另保留有效区间分母结果。\n\n' + '\n'.join(checks) + '\n')
    if group == 'G6B':
        report += '\nG6B 复用残差与 LS/H/HT 结果；W/N 耗时仅为本次二阶段重算。\n'
    (results / (group + '_结果.md')).write_text(report, encoding='utf-8')
    audit.update(pdf_figures=len(index), csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest())
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{group}: 1 CSV, 1 Markdown report, 4 PDF figures exported to {target}', flush=True)
