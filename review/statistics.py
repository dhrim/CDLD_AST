from paths import *
import time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
threadpool_limits(2)
protocol=json.loads((PACKAGE/'config/review_statistics.json').read_text());R=protocol['replicates'];seeds=[965136,110623,615357];budgets=[0,10,20,30,40,50]
configs=[('CDLD',0),('CDLD',10),('CDLD',50),('MF',0),('MF',10),('MF',30),('MF',50),('AST mean',0)]
metrics=['AUROC','accuracy','sensitivity_S','specificity','BA','F1_S','BCE']
def div(a,b):
 with np.errstate(divide='ignore',invalid='ignore'):return np.where(b>0,a/b,np.nan)
def calculate(y,p,weights):
 pos=y==1;neg=~pos;np_=weights[:,pos].sum(1);nn=weights[:,neg].sum(1);total=np_+nn
 pred=(p>=.5).astype(float);tp=weights[:,pos]@pred[pos];fp=weights[:,neg]@pred[neg];tn=nn[:,None]-fp;fn=np_[:,None]-tp
 sens=div(tp,np_[:,None]);spec=div(tn,nn[:,None])
 out=np.empty((len(weights),len(metrics)))
 out[:,1]=np.mean(div(tp+tn,total[:,None]),axis=1);out[:,2]=np.mean(sens,axis=1);out[:,3]=np.mean(spec,axis=1);out[:,4]=np.mean((sens+spec)/2,axis=1);out[:,5]=np.mean(div(2*tp,2*tp+fp+fn),axis=1)
 pp=np.clip(p,1e-7,1-1e-7);loss=-(y[:,None]*np.log(pp)+(1-y[:,None])*np.log1p(-pp));out[:,6]=div(weights@loss.mean(axis=1),total)
 if pos.any() and neg.any():
  # The denominator is common to all seed/repeat predictions. Average pair concordance first.
  concordance=np.zeros((pos.sum(),neg.sum()))
  for k in range(p.shape[1]):
   diff=p[pos,k,None]-p[neg,k][None,:];concordance+=(diff>0)+.5*(diff==0)
  concordance/=p.shape[1];out[:,0]=div(np.sum((weights[:,pos]@concordance)*weights[:,neg],axis=1),np_*nn)
  assert abs(out[0,0]-np.mean([roc_auc_score(y,p[:,k]) for k in range(p.shape[1])]))<1e-10
 else:out[:,0]=np.nan
 return out
all_ci=[];all_diff=[];threshold=[];point_rows=[];validity=[];start=time.time()
for ds in ['mimic','eicu']:
 folders=sorted(x for x in (B/'inputs').iterdir() if x.is_dir());labels={f.name:pd.read_csv(B/'labels'/ds/f.name/'scoring_index.csv.gz',dtype={'patient_id':str,'episode_id':str}) for f in folders}
 patients=sorted(set(x for l in labels.values() for x in l.patient_id));pid={p:i for i,p in enumerate(patients)}
 rng=np.random.default_rng(protocol['seed']+(ds=='eicu'));cw=np.vstack([np.ones(len(patients)),rng.multinomial(len(patients),np.full(len(patients),1/len(patients)),size=R)]).astype(float)
 values=[];counts=[]
 for f in folders:
  l=labels[f.name];y=l.y.to_numpy();weights=cw[:,[pid[p] for p in l.patient_id]];counts.append(((y==0).sum(),(y==1).sum()));predictions={}
  for model in ['CDLD','MF']:
   parts=[]
   for seed in seeds:
    dest=B/'predictions/CDLD'/f.name/f'seed{seed}'/ds if model=='CDLD' else B/'mf_runs'/f.name/f'seed{seed}'/ds
    part=np.stack([np.stack([np.load(dest/f'r{rep}_{model}_b{b}.npz')['probability'] for b in budgets]) for rep in range(10)])
    assert part.shape==(10,len(budgets),len(y));parts.append(part)
   predictions[model]=np.concatenate(parts,axis=0)
  pack=np.load(f/('mimic_adapt_inputs.npz' if ds=='mimic' else 'eicu_observed_inputs.npz'));obs={}
  for (u,d),v in zip(pack['observed_x'].astype(int),pack['observed_y']):obs.setdefault(u,[]).append(float(v))
  ast=np.array([np.mean(obs[int(u)]) for u in pack['query_x'][:,0]])[:,None];cv=[]
  for model,b in configs:
   p=ast if model=='AST mean' else predictions[model][:,budgets.index(b),:].T.astype(float)
   v=calculate(y,p,weights);cv.append(v)
   point_rows.append(dict(dataset=ds,combination=f.name,model=model,support=b,**dict(zip(metrics,v[0]))))
   if b!=30:
    for t in protocol['thresholds']:
     pr=p>=t;tp=((y[:,None]==1)&pr).sum(0);fn=((y[:,None]==1)&~pr).sum(0);tn=((y[:,None]==0)&~pr).sum(0);fp=((y[:,None]==0)&pr).sum(0)
     se=div(tp,tp+fn);sp=div(tn,tn+fp)
     threshold.append(dict(dataset=ds,combination=f.name,model=model,support=b,threshold=t,accuracy=np.mean((tp+tn)/len(y)),sensitivity_S=np.mean(se),specificity=np.mean(sp),BA=np.mean((se+sp)/2),F1_S=np.mean(div(2*tp,2*tp+fp+fn))))
  values.append(cv);print(ds,f.name,'done',round(time.time()-start,1),flush=True)
 values=np.array(values);counts=np.array(counts) # combinations, configs, reps+1, metrics
 for subset in ['original','nonS_ge5']:
  eligible=np.ones(len(folders),bool) if subset=='original' else (counts[:,0]>=5)&(counts[:,1]>0)
  for mi,metric in enumerate(metrics):
   use=eligible&np.isfinite(values[:,:,0,mi]).all(axis=1)
   aggregates=values[use,:,:,mi].mean(axis=0)
   for ci,(model,b) in enumerate(configs):
    v=aggregates[ci];good=np.isfinite(v[1:]);valid=int(good.sum());interval=np.quantile(v[1:][good],[.025,.975]).tolist() if valid>=.95*R else [None,None]
    all_ci.append(dict(dataset=ds,subset=subset,metric=metric,model=model,support=b,combinations=int(use.sum()),estimate=float(v[0]),valid_replicates=valid,total_replicates=R,lower=interval[0],upper=interval[1]))
   pairs=[(1,4,'CDLD-MF support10'),(2,6,'CDLD-MF support50'),(1,7,'CDLD-AST support10'),(2,7,'CDLD-AST support50'),(2,0,'CDLD support50-support0'),(6,3,'MF support50-support0'),(5,7,'MF-AST support30')]
   for left,right,name in pairs:
    v=aggregates[left]-aggregates[right];good=np.isfinite(v[1:]);valid=int(good.sum());interval=np.quantile(v[1:][good],[.025,.975]).tolist() if valid>=.95*R else [None,None]
    all_diff.append(dict(dataset=ds,subset=subset,metric=metric,contrast=name,combinations=int(use.sum()),estimate=float(v[0]),valid_replicates=valid,total_replicates=R,lower=interval[0],upper=interval[1]))
 np.savez_compressed(W/f'{ds}_bootstrap_metrics.npz',values=values,counts=counts,combinations=[f.name for f in folders],configs=np.array(configs,dtype=str),metrics=metrics)
for filename,rows in [('intervals.json',all_ci),('paired_differences.json',all_diff),('point_estimates.json',point_rows),('threshold_by_combination.json',threshold)]:pd.DataFrame(rows).to_json(W/filename,orient='records',indent=2,double_precision=12)
t=pd.DataFrame(threshold);t.groupby(['dataset','model','support','threshold'],as_index=False)[['accuracy','sensitivity_S','specificity','BA','F1_S']].mean().to_json(W/'threshold_summary.json',orient='records',indent=2,double_precision=12)
(W/'status.json').write_text(json.dumps(dict(phase='complete',seconds=time.time()-start,replicates=R),indent=2));print('COMPLETE',time.time()-start,flush=True)

rows=[]
for ds in ['mimic','eicu']:
 z=np.load(W/f'{ds}_bootstrap_metrics.npz')
 for name,counts,values in zip(z['combinations'],z['counts'],z['values']):
  for (model,b),v in zip(z['configs'],values):
   for i,metric in enumerate(z['metrics']):
    good=np.isfinite(v[1:,i]);n=int(good.sum());ci=np.quantile(v[1:,i][good],[.025,.975]) if n>=1900 else [None,None]
    rows.append(dict(dataset=ds,combination=str(name),n_nonS=int(counts[0]),n_S=int(counts[1]),model=str(model),support=int(b),metric=str(metric),estimate=float(v[0,i]),valid_replicates=n,lower=ci[0],upper=ci[1]))
pd.DataFrame(rows).to_json(W/'combination_intervals.json',orient='records',indent=2,double_precision=12)
print('Saved',len(rows),'combination-level metric intervals or explicit unavailable results.')

t=pd.read_json(W/'threshold_by_combination.json');t=t[t.specificity.notna()&t.sensitivity_S.notna()]
cols=['accuracy','sensitivity_S','specificity','BA','F1_S']
t.groupby(['dataset','model','support','threshold'],as_index=False)[cols].mean().to_json(W/'threshold_common_combinations.json',orient='records',indent=2,double_precision=12)
