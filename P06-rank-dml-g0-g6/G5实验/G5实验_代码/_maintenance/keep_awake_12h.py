import ctypes,json,os,re,subprocess,time
from pathlib import Path
from datetime import datetime,timedelta
ROOT=Path(__file__).resolve().parent
STATUS=ROOT/'power_guard_status.json'
def run(*args):
    p=subprocess.run(['powercfg',*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('powercfg failed: '+' '.join(args)+' '+p.stdout.decode(errors='replace'))
    return p.stdout.decode(errors='replace')
def read(scheme,sub,key):
    values=re.findall(r'0x([0-9a-fA-F]{8})',run('/qh',scheme,sub,key))
    if len(values)<2:raise RuntimeError('Could not read power setting '+key)
    return dict(ac=int(values[-2],16),dc=int(values[-1],16))
def save(data):
    temp=STATUS.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,STATUS)
scheme=re.search(r'[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}',run('/getactivescheme')).group()
settings=[('SUB_SLEEP','STANDBYIDLE'),('SUB_SLEEP','HIBERNATEIDLE'),('SUB_BUTTONS','LIDACTION')]
original=[dict(sub=sub,key=key,values=read(scheme,sub,key)) for sub,key in settings]
state=dict(pid=os.getpid(),scheme=scheme,start=datetime.now().astimezone().isoformat(),end=(datetime.now().astimezone()+timedelta(hours=12)).isoformat(),status='starting',original=original,low_battery_protection='unchanged',display_sleep='unchanged')
save(state);applied=[]
try:
    for item in original:
        for mode,value in item['values'].items():
            if value:
                run('/set'+mode+'valueindex',scheme,item['sub'],item['key'],'0')
                applied.append((item,mode,value))
    run('/setactive',scheme)
    set_state=ctypes.windll.kernel32.SetThreadExecutionState
    set_state.argtypes=[ctypes.c_uint];set_state.restype=ctypes.c_uint
    if not set_state(0x80000001):raise ctypes.WinError()
    state.update(status='active',verified=[dict(sub=s,key=k,values=read(scheme,s,k)) for s,k in settings]);save(state)
    deadline=time.monotonic()+12*3600
    while time.monotonic()<deadline:time.sleep(min(30,deadline-time.monotonic()))
    state['status']='finished'
except BaseException as e:
    state.update(status='failed',error=repr(e));raise
finally:
    restore_errors=[]
    for item,mode,value in applied:
        try:
            if read(scheme,item['sub'],item['key'])[mode]==0:run('/set'+mode+'valueindex',scheme,item['sub'],item['key'],str(value))
        except Exception as e:restore_errors.append(str(e))
    try:run('/setactive',scheme)
    except Exception as e:restore_errors.append(str(e))
    ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000000))
    state['restore_errors']=restore_errors;save(state)
