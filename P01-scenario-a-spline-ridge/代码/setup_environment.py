"""Check current notebook Python; install only missing requirements; save environment."""
import importlib.util,importlib.metadata as md,json,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PACKAGES={"numpy":"numpy","scipy":"scipy","pandas":"pandas","matplotlib":"matplotlib",
          "tqdm":"tqdm","psutil":"psutil","threadpoolctl":"threadpoolctl","nbformat":"nbformat","ipykernel":"ipykernel"}
def main():
    if sys.version_info<(3,11):raise RuntimeError("Python >= 3.11 required; choose environment2 on this computer.")
    print("Python:",sys.executable,flush=True)
    missing=[name for name,module in PACKAGES.items() if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing missing packages only:",missing,flush=True)
        subprocess.check_call([sys.executable,"-m","pip","install",*missing])
    versions={name:md.version(name) for name in PACKAGES}
    print(json.dumps(versions,indent=2),flush=True)
    dest=ROOT/"config/environment";dest.mkdir(parents=True,exist_ok=True)
    (dest/"versions.json").write_text(json.dumps(dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=versions),indent=2),encoding="utf-8")
    with (dest/"requirements_freeze.txt").open("w",encoding="utf-8") as f:
        subprocess.run([sys.executable,"-m","pip","freeze"],stdout=f,check=True)
    print("Environment OK. Snapshot:",dest,flush=True)
if __name__=="__main__":main()

