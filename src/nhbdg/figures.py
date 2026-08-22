"""Publication plots from processed data only."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C=["#0072B2","#D55E00","#009E73","#CC79A7"]; M=["o","s","^","D"]
def R(p):
    try:return pd.read_csv(p)
    except:return pd.DataFrame()
def V(x): return x[(x.kind=="branch_trial")&(x.status=="SUCCESS")&x.accepted.astype(bool)&(x.s_min>=.7)]
def S(f,o,n):
    plt.rcParams.update({"font.size":8.5,"axes.labelsize":8.5,"legend.fontsize":7,"pdf.fonttype":42});
    for i,a in enumerate(f.axes):
     a.text(-.12,1.04,f"({chr(97+i)})",transform=a.transAxes,fontweight="bold")
     h,l=a.get_legend_handles_labels()
     if h:
      if not (n=="fig02_obc_covariance" and i==0):a.legend(frameon=False,fontsize=7)
    f.tight_layout();f.savefig(o/(n+".pdf"));f.savefig(o/(n+".png"),dpi=300);plt.close(f)
def figure02(d,o):
 x,p=R(d/"fig2.csv"),R(d/"profiles.csv");f,a=plt.subplots(2,2,figsize=(7.2,5.7));g=6/79;p=p[(p.study=="fig2")&(p.L==80)&np.isclose(p.g,g)]
 for i,col in enumerate(["delta_plus_abs","delta_minus_abs"]):
  lab=r"$|\Delta_+|/t$" if i==0 else r"$|\Delta_-|/t$";z=p[p.kind=="mapped"];a[0,0].semilogy(z.j,z[col],color=C[i],label="mapped "+lab);[a[0,0].semilogy(q.j,q[col],ls="None",marker=m,markevery=5,mfc="white",color=C[i],label=k+" "+lab) for k,m in [("direct","o"),("rescaled","s")] for q in [p[p.kind==("raw" if k=="direct" else k)]]]
 a[0,0].text(.50,.91,r"blue: $\Delta_+$     orange: $\Delta_-$",transform=a[0,0].transAxes,ha="center",va="top",fontsize=7.2,bbox={"facecolor":"white","edgecolor":"none","alpha":.92,"pad":1.5})
 a[0,0].text(.50,.83,"solid: mapped     ○: direct     □: rescaled",transform=a[0,0].transAxes,ha="center",va="top",fontsize=7.2,bbox={"facecolor":"white","edgecolor":"none","alpha":.92,"pad":1.5})
 z=p[p.kind=="mapped"];a[0,1].plot(z.j,z.P_real,label=r"mapped $\mathrm{Re}\,P_j/t^2$");a[1,0].plot(z.j,z.density,label="mapped OBC")
 h=R(d/"profiles.csv");h=h[(h.study=="fig2")&(h.L==80)&(h.kind=="hermitian")];
 if not h.empty:a[0,1].plot(h.j,h.delta_plus_abs*h.delta_minus_abs,"--",color=".3",label=r"Hermitian $|\tilde\Delta_j|^2/t^2$");a[1,0].plot(h.j,h.density,"--",color=".3",label="Hermitian reference")
 for i,(k,q) in enumerate(x[x.kind.isin(["raw","rescaled"])].groupby("kind")):
  for j,col in enumerate(["map_error","pair_product_error","density_error","spectrum_error"]):a[1,1].scatter(j+(i-.5)*.2,q[col].max(),color=C[i],marker=M[i],label=k if j==0 else None)
 a[1,1].axhline(1e-8,color=".4",ls="--");a[1,1].set(yscale="log",xticks=range(4),xticklabels=["gap map","pair","density","spectrum"])
 a[0,0].set(xlabel="site $j$",ylabel=r"$|\Delta_\pm|/t$");a[0,1].set(xlabel="site $j$",ylabel=r"pairing invariant / $t^2$");a[1,0].set(xlabel="site $j$",ylabel=r"$n_j$");a[1,1].set(xlabel="validation category",ylabel="relative discrepancy")
 S(f,o,"fig02_obc_covariance")
def figure03(d,o):
 x,t=V(R(d/"fig3.csv")),R(d/"thresholds.csv");f,a=plt.subplots(2,2,figsize=(7.2,5.7))
 for i,((L,g),q) in enumerate(x.groupby(["L","g"])):
  q=q[q["lambda"]>1e-9].sort_values("lambda");a[0,0].loglog(q["lambda"],q.metric_violation,color=C[i],marker=M[i],label=f"L={L}, g={g:g}");a[0,1].loglog(q.chi,q.metric_violation,color=C[i],marker=M[i])
 for q in a[0]:q.set(ylabel=r"$M_{mc}$");q.axhspan(1e-16,1e-9,color=".92")
 tt=t[(t.study=="fig3")&t.quantity.str.startswith("metric_")]
 for i,(k,q) in enumerate(tt.groupby("quantity")):a[1,0].scatter(q.g*q.L,q.chi_c,color=C[i],marker=M[i],label=k)
 z=t[(t.study=="fig3")&np.isclose(t.g,.05)&t.quantity.isin(["metric_1e-2","pair"])]
 for i,(k,q) in enumerate(z.groupby("quantity")):a[1,1].scatter(q.L,q.chi_c,color=C[i],marker=M[i],label=k)
 a[1,0].set(xlabel="$gL$",ylabel=r"$\chi_c$",yscale="log");a[1,1].set(xlabel="$L$",ylabel=r"threshold $\chi_x$",yscale="log");S(f,o,"fig03_weak_link_crossover")
def figure04(d,o):
 x,p,e,s=V(R(d/"fig4.csv")),R(d/"profiles.csv"),R(d/"spectra.csv"),R(d/"fig4_snapshots.csv");f,a=plt.subplots(2,2,figsize=(7.2,5.7));q=x[(x.L==40)&(x.chi>1e-9)].sort_values("chi")
 for i,k in enumerate(["metric_violation","delta_P_bulk","gamma_max_over_t"]):a[0,0].loglog(q.chi,q[k],color=C[i],marker=M[i],label=k)
 for i,(_,r) in enumerate(s[s.L==40].iterrows()):
  z=p[p.run_id==r.run_id];a[0,1].plot(z.j,z.P_real,color=C[i],label=r.snapshot);z=e[e.run_id==r.run_id];a[1,0].scatter(z.E_real,z.E_imag,s=6,color=C[i],label=r.snapshot)
 end=x[np.isclose(x["lambda"],1)].groupby("L").first().reset_index();ctrl=R(d/"fig4.csv");ctrl=ctrl[ctrl.kind=="bandwidth_control"]
 for i,L in enumerate([24,40]):
  a[1,1].scatter(i-.1,end[end.L==L].bulk_pair_product_real,color=C[0],marker="o",label="NH pair" if i==0 else None);a[1,1].scatter(i+.1,ctrl[ctrl.L==L].bulk_pair_product_real,color=".3",marker="s",label="Hermitian pair" if i==0 else None)
 a[1,1].set(xticks=[0,1],xticklabels=["L=24","L=40"],ylabel=r"$P_{bulk}/t^2$");S(f,o,"fig04_pbc_endpoint")
def figure_s1(d,o):
 x=R(d/"conditioning.csv");f,a=plt.subplots(1,2,figsize=(7.2,3));
 for i,(k,q) in enumerate(x[x.kind.isin(["raw","rescaled"])].groupby("kind")):a[0].semilogy(q.q,q.similarity_condition,color=C[i],marker=M[i],label=r"$\kappa(V)$ "+k);a[0].semilogy(q.q,q.right_condition,color=C[i],marker=M[i],ls="--",label=r"$\kappa(R)$ "+k);a[1].semilogy(q.q,q.covariance_error,color=C[i],marker=M[i],label=k)
 a[1].axhline(1e-8,color=".4",ls="--");S(f,o,"figS1_conditioning")
def figure_s2(d,o):
 x=R(d/"green_frequency.csv");f,a=plt.subplots(1,3,figsize=(7.2,3));
 for i,(q,z) in enumerate(x.groupby("q")):a[0].plot(z.omega,z.center_ldos,color=C[i],label=f"q={q:g}");a[1].semilogy(z.omega,z.G_1L_abs,color=C[i]);a[1].semilogy(z.omega,z.G_L1_abs,color=C[i],ls="--");a[2].semilogy(z.omega,z.stripped_1L_abs,color=C[i])
 S(f,o,"figS2_green_covariance")
def figure_s3(d,o):
 x=V(R(d/"fig3.csv"));f,a=plt.subplots(2,2,figsize=(7.2,5.2));a=a.flatten();cols=["s_min","minimum_eigenvalue_separation","right_condition","field_residual"]
 for i,(k,q) in enumerate(x.groupby(["L","g"])):
  for j,c in enumerate(cols):a[j].plot(q["lambda"],q[c],color=C[i%4],marker=M[i%4],ms=2,label=str(k) if j==0 else None)
 for j in [1,2,3]:a[j].set_yscale("log")
 for j,l in enumerate([r"$s_{\min}$","minimum eigenvalue separation",r"$\kappa(R)$","SCF field residual"]):a[j].set(xscale="log",xlabel=r"$\lambda$",ylabel=l)
 a[0].axhline(.7,color=".4",ls="--");a[3].axhline(1e-10,color=".4",ls="--");S(f,o,"figS3_branch_quality")
def make(d,o,selected="all"):
 for k,f in {"fig02":figure02,"fig03":figure03,"fig04":figure04,"figS1":figure_s1,"figS2":figure_s2,"figS3":figure_s3}.items():
  if selected in ["all",k]:f(d,o)
