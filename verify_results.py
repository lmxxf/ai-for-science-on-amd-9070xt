#!/usr/bin/env python3
"""Validate synced immutable evidence and source checksums; does not rerun GPUs."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
load=lambda p:json.loads((R/p).read_text())
for item in load('results/phase2_sync_manifest.json')['files']:
 p=R/item['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
b=load('results/module6_results.json');assert b['pairbias']['pass'] and len(b['epochs'])==20
for v in b['pairbias']['comparisons'].values():
 assert v['CUDA_vs_HIP_max_abs']<1e-3 and v['HIP_vs_official_PyTorch_max_abs']<1e-3
f=load('results/module7_results.json');assert f['validation']['force_relative_l2']<1e-12
assert f['validation']['finite_difference_max_abs_error']<1e-7
for s in f['systems']:
 assert s['hip_vs_cuda']['force_relative_l2']<1e-12
 for v in s['backends'].values():assert v['force_relative_l2_vs_cpu']<1e-12
p=load('results/module8_results.json');assert p['status']=='PASS' and len(p['history'])==12
print('PASS AMD mirror: all synced source/evidence hashes, CUDA/HIP/PyTorch core checks, forcefield physics/precision, protein training evidence. Raw full-array checks were run in source workspace.')
