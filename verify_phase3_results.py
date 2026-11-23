#!/usr/bin/env python3
"""Validate compact Phase 3 evidence and synced source hashes; no GPU rerun."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent
load=lambda p:json.loads((R/p).read_text())
for entry in load('results/phase3_sync_manifest.json')['files']:
 p=R/entry['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],p
p=load('results/module9_results.json')
assert p['validation']['mesh_sizes']['128']['force_relative_l2']<p['validation']['mesh_sizes']['64']['force_relative_l2']
for b in p['backends'].values():
 assert b['initial_force_relative_l2_vs_cpu']<1e-10 and b['nve_dt_halving_drift_ratio']>2
 assert b['trajectories']['nve']['steps']==1000 and b['trajectories']['nvt']['steps']==1000
r=load('results/module10_results.json')
for b in r['backends'].values():assert b['eigenvalue_max_abs_error']<1e-10 and b['independent_full_H_max_residual']<1e-7
r=load('results/module11_results.json');assert r['hip_vs_cuda_visibility_relative_l2']<1e-12 and r['hip_vs_cuda_clean_relative_l2']<1e-12
r=load('results/module12_results.json');assert r['cross_vendor_byte_identical'] and r['input']['sequences']==100000
assert len({v['score_sha256'] for v in r['backends'].values()})==1
r=load('results/module13_results.json');c=r['official_fp32_compatibility']
assert r['parameter_count']==47331139 and r['layers']==15 and r['test_samples']==112
assert r['AMD_vs_GB10_FP64_max_abs_logS']<1e-5 and c['pass']
assert max(abs(a-b) for a,b in zip(c['AMD_pred_logS'],c['reference_pred_logS']))<1e-5
print('PASS Phase3 mirror: every source/evidence hash and numerical acceptance summary. Full raw arrays/traces independently checked in primary workspace; this is not a GPU rerun.')
