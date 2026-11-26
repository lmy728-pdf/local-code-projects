# -*- coding: utf-8 -*-
"""使用本机 Jupyter 控制通道挂接已验证的暂停钩子，不发送中断或终止信号。"""
import argparse, json, sys, time
from pathlib import Path
from jupyter_client import BlockingKernelClient, KernelManager

def request(client, code, expression="None", timeout=20):
    message=client.session.msg("execute_request",content=dict(
        code=code,silent=True,store_history=False,user_expressions={"result":expression},
        allow_stdin=False,stop_on_error=False))
    client.control_channel.send(message)
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        reply=client.get_control_msg(timeout=max(.1,until-time.monotonic()))
        if reply.get('parent_header',{}).get('msg_id')==message['header']['msg_id']:
            assert reply['content']['status']=='ok',reply['content']
            return reply['content']
    raise TimeoutError("本机控制通道未响应")

def selftest():
    manager=KernelManager(kernel_name='environment2')
    manager.start_kernel()
    client=manager.blocking_client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=45)
        message=client.execute("import time\n_control_test_finished=False\ntime.sleep(4)\n_control_test_finished=True",silent=True,store_history=False)
        time.sleep(.5)
        started=time.monotonic()
        response=request(client,"_control_test_seen_while_busy = not _control_test_finished","_control_test_seen_while_busy")
        elapsed=time.monotonic()-started
        assert response['user_expressions']['result']['data']['text/plain']=='True',response
        assert elapsed<3,elapsed
        while True:
            reply=client.get_shell_msg(timeout=10)
            if reply.get('parent_header',{}).get('msg_id')==message:
                assert reply['content']['status']=='ok',reply
                break
        result=dict(control_during_execution_passed=True,reply_seconds=elapsed)
        print(json.dumps(result))
        return result
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)

def arm(connection, root, deadline, config_hash):
    client=BlockingKernelClient(connection_file=str(connection))
    client.load_connection_file()
    client.start_channels()
    try:
        helper=Path(root)/'运行控制'/'safe_pause_guard.py'
        # 自己的模块命名空间，不覆盖 Notebook 的统计变量。
        payload = (
            "import importlib.util as _dml_pause_importlib, sys as _dml_pause_sys\n"
            "_dml_pause_spec=_dml_pause_importlib.spec_from_file_location('dml_scheduled_pause_guard',"+repr(str(helper))+")\n"
            "_dml_pause_module=_dml_pause_importlib.module_from_spec(_dml_pause_spec)\n"
            "_dml_pause_sys.modules['dml_scheduled_pause_guard']=_dml_pause_module\n"
            "_dml_pause_spec.loader.exec_module(_dml_pause_module)\n"
            "_dml_pause_state=_dml_pause_module.install(_dml_pause_sys.modules['spline_p10_lognormal_worker'],"
            +repr(str(root))+","+repr(deadline)+","+repr(config_hash)+")\n"
        )
        result=request(client,payload,"_dml_pause_state",timeout=30)
        print(json.dumps(result,ensure_ascii=False))
    finally:
        client.stop_channels()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--connection')
    parser.add_argument('--root')
    parser.add_argument('--deadline',type=float)
    parser.add_argument('--config-hash')
    args=parser.parse_args()
    if args.selftest:
        result=selftest()
        root=Path(__file__).resolve().parent
        (root/'kernel_control_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    else:
        assert all((args.connection,args.root,args.deadline,args.config_hash))
        arm(args.connection,args.root,args.deadline,args.config_hash)

