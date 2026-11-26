"""Repair whole-notebook hash checks; never execute a simulation."""
import ast,hashlib,json,pathlib,shutil
from datetime import datetime
ROOT=pathlib.Path(__file__).resolve().parent
NB=ROOT.parent/'Huber_DML_pilot.ipynb'
def ast_hash(source):
    return hashlib.sha256(ast.dump(ast.parse(source),include_attributes=False).encode()).hexdigest()
def main():
    backup=ROOT/'backups'/('source_check_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    backup.mkdir(parents=True,exist_ok=False)
    for p in [NB,ROOT/'run_huber_pilot.py',ROOT/'source_manifest.json']:
        shutil.copy2(p,backup/p.name)
    manifest=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
    original=pathlib.Path(manifest['notebook'])
    snapshot=pathlib.Path(manifest['legacy_run'])/'diagnostics_abc_20261030'/'notebook_source_snapshot.py'
    source=json.loads(original.read_text(encoding='utf-8'))
    current='\n'.join(''.join(c['source']) for c in source['cells'] if c['cell_type']=='code')
    expected=ast_hash(snapshot.read_text(encoding='utf-8'))
    assert ast_hash(current)==expected,'Actual original code changed; do not rebaseline'
    manifest['notebook_code_ast_sha256']=expected
    manifest['notebook_code_ast_reference']=str(snapshot)
    manifest['notebook_validation_policy']='Validate code AST; allow output, execution count, metadata and formatting changes.'
    (ROOT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    replacement='''import ast
for path,digest in manifest['files_sha256'].items():
    source_path=pathlib.Path(path)
    if source_path.suffix.lower()=='.ipynb':
        original_nb=json.loads(source_path.read_text(encoding='utf-8'))
        original_code='\\n'.join(''.join(c['source']) for c in original_nb['cells'] if c['cell_type']=='code')
        code_hash=hashlib.sha256(ast.dump(ast.parse(original_code),include_attributes=False).encode()).hexdigest()
        if code_hash!=manifest['notebook_code_ast_sha256']:
            raise RuntimeError('Original notebook computation code changed: '+path)
    elif sha(source_path)!=digest:
        raise RuntimeError('Source changed since packaging: '+path)
'''
    notebook=json.loads(NB.read_text(encoding='utf-8'));changed=0
    for cell in notebook['cells']:
        source=''.join(cell['source'])
        if cell['cell_type']=='code' and "for path,digest in manifest['files_sha256'].items():" in source:
            start=source.index("for path,digest in manifest['files_sha256'].items():")
            end=source.index("expected=config['hgbr_params_seed_zero']",start)
            cell['source']=(source[:start]+replacement+source[end:]).splitlines(keepends=True)
            cell['outputs']=[];cell['execution_count']=None;changed+=1
    assert changed==1
    NB.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    runner=ROOT/'run_huber_pilot.py';source=runner.read_text(encoding='utf-8')
    start=source.index("    for path,digest in manifest['files_sha256'].items():")
    end=source.index("    expected=config['hgbr_params_seed_zero']",start)
    runner.write_text(source[:start]+''.join('    '+line+'\n' for line in replacement.splitlines())+source[end:],encoding='utf-8')
    (ROOT/'source_check_fix.md').write_text('修复：仅运行输出、元数据、格式变化不再阻断原notebook校验；全部计算代码AST与旧审计快照一致。冻结模块与原配置仍严格检查SHA256。没有更新旧文件哈希来绕过检查，没有修改DGP/学习器/SE，也未运行模拟。备份：'+str(backup),encoding='utf-8')
    print('Patched notebook and script. Backup:',backup)
if __name__=='__main__':main()
