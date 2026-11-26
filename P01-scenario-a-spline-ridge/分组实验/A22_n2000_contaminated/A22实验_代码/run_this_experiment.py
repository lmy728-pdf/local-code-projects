"""Run only A22: n=2000, law=contaminated. Never chains other groups."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"代码"))
if __name__=="__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    if any(x=="--n" or x=="--law" or x.startswith("--n=") or x.startswith("--law=") for x in sys.argv[1:]):
        raise SystemExit("This launcher fixes n and law; choose a different group notebook.")
    sys.argv.extend(["--n","2000","--law","contaminated"])
    from runner import main
    main()
