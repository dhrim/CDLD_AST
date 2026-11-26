from paths import *
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
roots={'original':B/'inputs',**{c:W/'inputs'/c for c in ['all_existing','matched_four','exclude_family_four']}}
def metrics(y,p):
 pr=p>=.5;tp=((y==1)&pr).sum();fn=((y==1)&~pr).sum();tn=((y==0)&~pr).sum();fp=((y==0)&pr).sum();se=tp/(tp+fn) if tp+fn else np.nan;sp=tn/(tn+fp) if tn+fp else np.nan;q=np.clip(p.astype(float),1e-7,1-1e-7)
 return dict(AUROC=roc_auc_score(y,p) if len(set(y))==2 else np.nan,accuracy=(tp+tn)/len(y),sensitivity_S=se,specificity=sp,BA=(se+sp)/2,F1_S=2*tp/(2*tp+fn+fp) if 2*tp+fn+fp else np.nan,BCE=np.mean(-y*np.log(q)-(1-y)*np.log1p(-q)))
rows=[]
for condition,root in roots.items():
 for inp in sorted(root.iterdir()):
  if not inp.is_dir():continue
  c=inp.name
  for ds in ['mimic','eicu']:
   ix=json.loads((inp/f'{ds}_index.json').read_text());labels=pd.read_csv(B/'labels'/ds/c/'scoring_index.csv.gz',dtype={'episode_id':str}).set_index('episode_id');eps=list(map(str,ix['query_episodes']));y=labels.loc[eps].y.to_numpy()
   pack=np.load(inp/('mimic_adapt_inputs.npz' if ds=='mimic' else 'eicu_observed_inputs.npz'));obs={}
   for (u,d),v in zip(pack['observed_x'].astype(int),pack['observed_y']):obs.setdefault(u,[]).append(float(v))
   score=np.array([np.mean(obs[int(u)]) for u in pack['query_x'][:,0]])
   for budget in [10,50]:rows.append(dict(dataset=ds,condition=condition,combination=c,model='AST mean',seed=0,repeat=0,support=budget,n=len(y),**metrics(y,score)))
   for model in ['MF','CDLD']:
    for seed in [965136,110623,615357]:
     dest=W/'predictions'/model/condition/c/f'seed{seed}'/ds
     if condition=='original':dest=B/'mf_runs'/c/f'seed{seed}'/ds if model=='MF' else B/'predictions/CDLD'/c/f'seed{seed}'/ds
     assert (dest/'completion.json').exists(),str(dest)
     for repeat in range(10):
      for budget in [10,50]:
       name=f'r{repeat}_'+model+f'_b{budget}.npz';p=np.load(dest/name)['probability'];assert len(p)==len(y)
       rows.append(dict(dataset=ds,condition=condition,combination=c,model=model,seed=seed,repeat=repeat,support=budget,n=len(y),**metrics(y,p)))
f=pd.DataFrame(rows);cols=['AUROC','accuracy','sensitivity_S','specificity','BA','F1_S','BCE'];g=f.groupby(['dataset','condition','combination','model','support'],as_index=False)[cols].mean()
f.to_json(W/'all_condition_metrics.json',orient='records',indent=2,double_precision=12);g.to_json(W/'all_combination_metrics.json',orient='records',indent=2,double_precision=12)
summary=g.groupby(['dataset','condition','model','support'])[cols].mean();summary['combinations']=g.groupby(['dataset','condition','model','support']).size();summary.reset_index().to_json(W/'all_summary.json',orient='records',indent=2,double_precision=12)
print(summary.round(4).to_string())
