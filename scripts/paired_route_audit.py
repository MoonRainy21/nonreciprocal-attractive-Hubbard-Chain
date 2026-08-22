#!/usr/bin/env python3
"""Decisive paired continuation-history audit at prescribed targets only."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy.optimize import linear_sum_assignment
from nhbdg.model import Chain, bdg_matrix, Numerics
from nhbdg.solver import MeanFieldSolver, HFBState
from nhbdg.fixed_filling import solve_fixed_filling
from nhbdg.observables import metric_violation, pair_deformation, gamma_max, complex_fraction, spectrum_distance
ROOT=Path(__file__).resolve().parents[1]
def rec(st,L):
 out=[]
 for p in (ROOT/'data/raw'/st).glob('*/metadata.json'):
  m=json.loads(p.read_text());q=json.loads((p.parent/'status.json').read_text())
  if m['kind']=='branch_trial' and m.get('accepted') and q['status']=='SUCCESS' and m['physical']['L']==L and abs(m['physical']['g']-.05)<1e-12 and abs(m['physical']['U']-2)<1e-12:out.append((m,q,p.parent))
 return sorted(out,key=lambda z:z[0]['physical']['lambda'])
def load(x):
 m,q,d=x;z=np.load(d/'state.npz');p=m['physical'];c=Chain(p['L'],p['U'],g=p['g'],lambda_=p['lambda'],filling=p['filling'],t=p['t']);e,_=MeanFieldSolver._eigensystem(bdg_matrix(c,float(z['mu']),z['n_up'],z['n_down'],z['delta_plus'],z['delta_minus']),None);a,b=linear_sum_assignment(abs(z['eigenvalues'][:,None]-e.values[None,:]));occ=np.zeros(len(e.values),bool);occ[b]=z['occupied'][a];e=replace(e,occupied=occ);return HFBState(c,float(z['mu']),z['delta_plus'],z['delta_minus'],z['n_up'],z['n_down'],e,0,0,0,0,True,'SUCCESS'),m
def run(seed,target,num):
 st,m=load(seed);c=replace(st.chain,lambda_=target);return solve_fixed_filling(MeanFieldSolver(c,num,'rescaled'),initial_state=replace(st,chain=c),occupied_reference=st.eigensystem),m
def cmp(a,b):
 P,Q=a.pair_product,b.pair_product;Ra=a.eigensystem.right[:,a.eigensystem.occupied];La=a.eigensystem.left[:,a.eigensystem.occupied];Rb=b.eigensystem.right[:,b.eigensystem.occupied];Lb=b.eigensystem.left[:,b.eigensystem.occupied];Ca=Ra@La.conj().T;Cb=Rb@Lb.conj().T
 return np.linalg.norm(P-Q)/max(np.linalg.norm(Q),1e-30),np.linalg.norm(a.density-b.density)/max(np.linalg.norm(b.density),1e-30),spectrum_distance(a.eigensystem.values,b.eigensystem.values)/a.chain.t,np.linalg.norm(Ca-Cb)/max(np.linalg.norm(Cb),1e-30)
def main():
 cfg=yaml.safe_load((ROOT/'configs/paper.yaml').read_text());num=Numerics(**cfg['numerics']);rows=[]
 for L,targets in [(24,[.05,.075,.10,.15,.20]),(40,[.012,.020,.025,.030])]:
  A,B=rec('fig3',L),rec('fig4',L)
  for t in targets:
   sa=max((x for x in A if x[0]['physical']['lambda']<t),key=lambda x:x[0]['physical']['lambda']);sb=max((x for x in B if x[0]['physical']['lambda']<t),key=lambda x:x[0]['physical']['lambda']);a,ma=run(sa,t,num);b,mb=run(sb,t,num);dp,dn,de,dc=cmp(a,b);oa=metric_violation(a)[0];ob=metric_violation(b)[0];da=pair_deformation(a,load(sa)[0],True);db=pair_deformation(b,load(sb)[0],True);ga,gb=gamma_max(a),gamma_max(b);ok=max(dp,dn,de,dc,abs(a.mu-b.mu))<1e-7 and (ga>1e-8)==(gb>1e-8);ver='PASS' if ok else ('FAIL' if max(dp,dn,de,dc)>1e-4 or ((ga>1e-8)!=(gb>1e-8)) else 'AMBIGUOUS')
   rows.append({'L':L,'g':.05,'lambda':t,'route_A_seed_lambda':sa[0]['physical']['lambda'],'route_B_seed_lambda':sb[0]['physical']['lambda'],'route_A_run_id':ma['run_id'],'route_B_run_id':mb['run_id'],'d_P':dp,'d_n':dn,'d_E':de,'d_C':dc,'mu_A':a.mu,'mu_B':b.mu,'Mmc_A':oa,'Mmc_B':ob,'deltaP_A':da,'deltaP_B':db,'gamma_A':ga,'gamma_B':gb,'complex_fraction_A':complex_fraction(a),'complex_fraction_B':complex_fraction(b),'field_residual_A':a.field_residual,'field_residual_B':b.field_residual,'number_residual_A':a.number_residual,'number_residual_B':b.number_residual,'verdict':ver})
 df=pd.DataFrame(rows);df.to_csv(ROOT/'data/processed/paired_route_audit.csv',index=False);final='FAIL' if (df.verdict=='FAIL').any() else ('AMBIGUOUS' if (df.verdict=='AMBIGUOUS').any() else 'PASS');out={'final_verdict':final,'rows':rows};(ROOT/'data/processed/paired_route_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
