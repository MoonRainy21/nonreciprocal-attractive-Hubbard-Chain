#!/usr/bin/env python3
"""Read-only Fig. 3/4 route-consistency audit; writes only processed audit files."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from nhbdg.model import Chain, bdg_matrix, Numerics
from nhbdg.solver import MeanFieldSolver, HFBState
from nhbdg.observables import metric_violation, pair_deformation, gamma_max, complex_fraction

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads('{}')
def records(st,L):
 out=[]
 for p in (ROOT/'data/raw'/st).glob('*/metadata.json'):
  m=json.loads(p.read_text()); q=json.loads((p.parent/'status.json').read_text())
  if m['kind']=='branch_trial' and m.get('accepted') and q['status']=='SUCCESS' and m['physical']['L']==L and abs(m['physical']['U']-2)<1e-12 and abs(m['physical']['g']-.05)<1e-12: out.append((m,q,p.parent))
 return sorted(out,key=lambda z:z[0]['physical']['lambda'])
def state(x):
 m,q,d=x; z=np.load(d/'state.npz'); p=m['physical']; c=Chain(p['L'],p['U'],g=p['g'],lambda_=p['lambda'],filling=p['filling'],t=p['t'])
 e,_=MeanFieldSolver._eigensystem(bdg_matrix(c,float(z['mu']),z['n_up'],z['n_down'],z['delta_plus'],z['delta_minus']),None)
 old,new=linear_sum_assignment(abs(z['eigenvalues'][:,None]-e.values[None,:])); occ=np.zeros(len(e.values),bool);occ[new]=z['occupied'][old];e=replace(e,occupied=occ)
 return HFBState(c,float(z['mu']),z['delta_plus'],z['delta_minus'],z['n_up'],z['n_down'],e,0,0,0,0,True,'SUCCESS'),m
def dist(a,b):
 P=a.pair_product;Q=b.pair_product; C1=a.eigensystem.right[:,a.eigensystem.occupied]@a.eigensystem.left[:,a.eigensystem.occupied].conj().T; C2=b.eigensystem.right[:,b.eigensystem.occupied]@b.eigensystem.left[:,b.eigensystem.occupied].conj().T
 rows,cols=linear_sum_assignment(abs(a.eigensystem.values[:,None]-b.eigensystem.values[None,:]))
 return [np.linalg.norm(P-Q)/max(np.linalg.norm(Q),1e-30),np.linalg.norm(a.density-b.density)/max(np.linalg.norm(b.density),1e-30),np.max(abs(a.eigensystem.values[rows]-b.eigensystem.values[cols]))/a.chain.t,np.linalg.norm(C1-C2)/max(np.linalg.norm(C2),1e-30)]
def main():
 rows=[]; brackets=[]
 for L in (24,40):
  A,B=records('fig3',L),records('fig4',L)
  for ma,qa,da in A:
   lam=ma['physical']['lambda']; match=[x for x in B if np.isclose(lam,x[0]['physical']['lambda'],rtol=1e-12,atol=0)]
   if not match or lam==0:continue
   a,am=state((ma,qa,da));b,bm=state(match[0]); dp,dn,de,dc=dist(a,b); oa=json.loads((da/'observables.json').read_text());ob=json.loads((match[0][2]/'observables.json').read_text()); verdict='PASS' if max(dp,dn,de,dc,abs(a.mu-b.mu))<1e-7 and complex_fraction(a)==complex_fraction(b) else 'FAIL'
   rows.append({'L':L,'g':.05,'lambda':lam,'route_A_run_id':am['run_id'],'route_B_run_id':bm['run_id'],'d_pair':dp,'d_density':dn,'d_spectrum':de,'d_projector':dc,'delta_mu':abs(a.mu-b.mu),'delta_Mmc':abs(oa['metric_violation']-ob['metric_violation']),'delta_deltaP':abs(oa.get('delta_P_bulk',np.nan)-ob.get('delta_P_bulk',np.nan)),'delta_gamma':abs(oa['gamma_max_over_t']-ob['gamma_max_over_t']),'complex_fraction_A':oa['complex_fraction'],'complex_fraction_B':ob['complex_fraction'],'verdict':verdict})
  for name,X in [('fig3',A),('fig4',B)]:
   gam=[(m['physical']['lambda'],json.loads((d/'observables.json').read_text())['gamma_max_over_t']) for m,q,d in X]; below=[v for v in gam if v[1]<1e-4];above=[v for v in gam if v[1]>=1e-4]
   brackets.append(dict(L=L,g=.05,route=name,lambda_below=max(below)[0] if below else np.nan,lambda_above=min(above)[0] if above else np.nan))
 df=pd.DataFrame(rows);df.to_csv(ROOT/'data/processed/consistency_audit.csv',index=False); out={'verdict':'INCONCLUSIVE' if len(df)<6 else ('PASS' if (df.verdict=='PASS').all() else 'FAIL'),'comparisons':rows,'gamma_brackets':brackets};(ROOT/'data/processed/consistency_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
