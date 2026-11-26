"""Collect completed groups only; does not start or resume simulations."""
import argparse
from layout import paths
from methods import LAWS
from reporting import summarize
def collect(stage):
    if stage not in ("main","pilot"):raise ValueError("Only collect main or pilot; original validation smoke is separate")
    sources=[paths(n,law,stage)["work"]/"records.sqlite" for n in (1000,500,2000) for law in LAWS]
    sources=[p for p in sources if p.exists()]
    if not sources:print("No group results yet.");return
    summarize(stage,record_sources=sources)
    print("Collected",len(sources),"condition databases; no simulation was run.")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("main","pilot"),default="main")
    collect(p.parse_args().stage)
