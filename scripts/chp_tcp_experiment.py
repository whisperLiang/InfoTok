#!/usr/bin/env python3
"""Actual ordered loopback transfer between isolated encoder/decoder processes."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import torch

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--decoder',required=True)
p.add_argument('--video',required=True)
p.add_argument('--output',required=True)
p.add_argument('--block-budgets')
a=p.parse_args()
root=Path(a.output)
root.mkdir(parents=True,exist_ok=False)
with socket.socket() as probe:
    probe.bind(('127.0.0.1',0))
    port=probe.getsockname()[1]
endpoint=f'tcp://127.0.0.1:{port}'
prefix=root/'transmission'
receiver=[sys.executable,'scripts/chp_isolated_receiver.py','--checkpoint',a.decoder,
          '--stream',endpoint,'--output',str(root/'receiver')]
sender=[sys.executable,'-m','cosmos_predict1.tokenizer.progressive.cli','send',
        '--checkpoint',a.checkpoint,'--stream',endpoint,'--video',a.video,'--frames','17',
        '--log-prefix',str(prefix)]
if a.block_budgets:
    sender += ['--block-budgets',a.block_budgets]
t0=time.monotonic_ns()
with (root/'receiver.log').open('w') as receiver_log,(root/'sender.log').open('w') as sender_log:
    receive=subprocess.Popen(receiver,stdout=receiver_log,stderr=subprocess.STDOUT)
    send=subprocess.run(sender,stdout=sender_log,stderr=subprocess.STDOUT)
    if send.returncode:
        receive.terminate()
        raise RuntimeError(f'sender failed: {send.returncode}')
    receive_code=receive.wait(timeout=300)
    if receive_code:
        raise RuntimeError(f'receiver failed: {receive_code}')
wall_seconds=(time.monotonic_ns()-t0)/1e9
sent=json.loads(Path(str(prefix)+'.sender.json').read_text())
received=json.loads((root/'receiver/receiver.json').read_text())
ids=torch.load(str(prefix)+'.sender_ids.pt',weights_only=True)
recovered=torch.load(root/'receiver/received_ids.pt',weights_only=True)
assert all(torch.equal(p,ids[:,i,:p.shape[1]]) for i,p in enumerate(recovered))
assert sent['actual_bytes']==received['actual_bytes']
rows=[]
for tx,rx in zip(sent['packets'],received['packets']):
    assert (tx['block_id'],tx['budget'])==(rx['block_id'],rx['budget'])
    rows.append(dict(block_id=tx['block_id'],budget=tx['budget'],new_ids=tx['new_ids'],
                     packet_bytes=tx['packet_bytes'],encode_seconds=tx['encode_seconds'],
                     decode_seconds=rx['decode_seconds'],
                     end_to_end_seconds=(rx['reconstruction_ready_ns']-tx['encode_start_ns'])/1e9))
result=dict(transport='tcp-loopback',protocol_version=received['protocol_version'],sender_command=sender,
            receiver_command=receiver,actual_bytes=sent['actual_bytes'],all_received_ids_equal=True,
            actual_budgets=[p.shape[1] for p in recovered],elapsed_including_startup_seconds=wall_seconds,
            peak_sender_bytes=sent['peak_allocated_bytes'],peak_receiver_bytes=received['peak_allocated_bytes'],
            end_to_end_definition='same-host monotonic clock: block encode start to RGB ready, including queueing',
            source_pacing='17 file frames available at sender before encode; not live 30fps capture',
            packets=rows)
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
