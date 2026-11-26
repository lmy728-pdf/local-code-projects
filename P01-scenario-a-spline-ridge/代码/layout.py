"""One condition per user-run notebook; common scientific code is shared."""
from pathlib import Path
from methods import LAWS
ROOT=Path(__file__).resolve().parents[1]
def paths(n,law,stage):
    if n not in (1000,500,2000) or law not in LAWS:raise ValueError("Unknown formal condition")
    ident=(1000,500,2000).index(n)*9+LAWS.index(law)+1
    group=ROOT/"分组实验"/f"A{ident:02d}_n{n}_{law}"
    code=group/f"A{ident:02d}实验_代码"
    return dict(root=group,code=code,work=code/"_work"/("replications" if stage=="main" else stage),
                logs=code/"logs",tables=group/"汇总表_CSV"/stage,figures=group/"图"/stage,reports=group/"结果_MD")
