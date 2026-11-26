import ast,hashlib,json,pathlib,shutil
ROOT=pathlib.Path(__file__).resolve().parent
record=json.loads((ROOT/'r200_quantile_update.json').read_text(encoding='utf-8'))
backup=pathlib.Path(record['backup'])
p=ROOT/'source_manifest.json'
shutil.copy2(p,backup/'source_manifest.json')
m=json.loads(p.read_text(encoding='utf-8'))
old=m['notebook'];new=ROOT.parent/'毕设'/'DML_Dukes_n1000_R1000.ipynb'
nb=json.loads(new.read_text(encoding='utf-8'))
code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
assert hashlib.sha256(ast.dump(ast.parse(code),include_attributes=False).encode()).hexdigest()==m['notebook_code_ast_sha256']
m['notebook']=str(new);m['files_sha256'][str(new)]=m['files_sha256'].pop(old)
m['relocation_note']='2026-11-02: original notebook found under Desktop/毕设; AST matches frozen audit.'
p.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
alternate=ROOT.parent/'毕设'/'Huber_DML_pilot.ipynb'
if alternate.exists():
    shutil.copy2(alternate,backup/'Huber_DML_pilot_from_bishe.ipynb')
    updated=json.loads((ROOT.parent/'Huber_DML_pilot.ipynb').read_text(encoding='utf-8'))
    for c in updated['cells']:
        if c['cell_type']=='code':
            c['source']=''.join(c['source']).replace('NOTEBOOK = ROOT.parent / "Huber_DML_pilot.ipynb"','NOTEBOOK = ROOT.parent / "毕设" / "Huber_DML_pilot.ipynb"').splitlines(keepends=True)
    alternate.write_text(json.dumps(updated,ensure_ascii=False,indent=2),encoding='utf-8')
print('Relocated source verified; both notebook copies updated.')
