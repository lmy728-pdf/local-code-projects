"""Execute only the new Stage2 notebook with its existing environment2 kernel."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parent
path=root.parent/'DML_Stage2_Robust_M_Pilot.ipynb'
nb=nbformat.read(path,as_version=4)
nb.metadata.kernelspec.name='environment2'
client=NotebookClient(nb,timeout=600,kernel_name='environment2',resources={'metadata':{'path':str(root)}})
try:
    client.execute()
finally:
    nbformat.write(nb,path)
print('Notebook execution completed:',path)
