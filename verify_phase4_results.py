#!/usr/bin/env python3
"""Verify copied Phase4 source/evidence hashes and acceptance summary; no numerical rerun."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
load=lambda p:json.loads((R/p).read_text())
for x in load('results/phase4_sync_manifest.json')['files']:
 p=R/x['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],p
r=load('results/module14_results.json');assert r['status']=='PASS' and r['gradient_validation']['max_abs_error']<1e-8
for b in r['backends'].values():assert b['nonbonded_collision_pairs_final']==0 and b['nonincrease_count']==10000
assert r['comparison']['hip_vs_cuda']['coordinate_relative_l2']==0 and r['comparison']['hip_vs_cuda']['energy_relative_l2']==0
f=r['failure_case'];assert not f['collision_threshold_changed'] and f['original_contact_final']>f['original_contact_initial']
assert f['initial_coordinate_sha256_before']==f['initial_coordinate_sha256_after']
r=load('results/module15_results.json');assert r['iteration_host_transfers']==r['iteration_host_eigensolves']==0
for s in r['systems']:
 for name,t in s['backends'].items():
  assert t['eigenvalue_max_abs_error']<1e-13
  if name!='cpu':assert t['iteration_host_transfers']==t['iteration_host_eigensolves']==0 and t['max_residual']<1e-10
r=load('results/module16_results.json');assert r['article_sha256_before']==r['article_sha256_after'] and not r['article_modified']
assert r['status']=='AUDIT_COMPLETED_WITH_FINDINGS' and r['findings_by_severity']=={'P0':2,'P1':11,'P2':11}
print('PASS Phase4 mirror hashes/acceptance summaries; article audit has findings and is not publication approval. Full raw numerical checks executed in primary workspace.')
