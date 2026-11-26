from paths import *
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
source=json.loads((B/'results/all_conditions.json').read_text())
names={'CDLD':'CDLD','MF':'MF','LR':'LR','RF':'RF','XGB':'XGBoost','frequency':'Frequency'}
def ratio(a,b):return a/b if b else np.nan
def metrics(tp,fn,fp,tn):
 # Source confusion matrix has nonsusceptibility as positive.
 se=ratio(tn,tn+fp);sp=ratio(tp,tp+fn)
 return dict(accuracy=ratio(tp+tn,tp+fn+fp+tn),sensitivity_S=se,specificity=sp,BA=(se+sp)/2,F1_S=ratio(2*tn,2*tn+fp+fn),precision_nonS=ratio(tp,tp+fp),F1_nonS=ratio(2*tp,2*tp+fp+fn),false_susceptible_rate=ratio(fn,tp+fn))
rows=[];counts={}
for r in source:
 if r['model'] not in names:continue
 q=dict(dataset=r['dataset'],combination=r['combination'],model=names[r['model']],support=r['budget'],seed=r['seed'],repeat=r['repeat'],AUROC=r['AUROC'],**metrics(r['TN'],r['FP'],r['FN'],r['TP']))
 rows.append(q)
 c=(r['TN']+r['FP'],r['TP']+r['FN']);key=(r['dataset'],r['combination'])
 if key in counts:assert counts[key]==c
 counts[key]=c
for folder in sorted((B/'inputs').iterdir()):
 if not folder.is_dir():continue
 entity=json.loads((folder/'entities.json').read_text())
 for ds in ['mimic','eicu']:
  pack=np.load(folder/('mimic_adapt_inputs.npz' if ds=='mimic' else 'eicu_observed_inputs.npz'))
  y=pd.read_csv(B/'labels'/ds/folder.name/'scoring_index.csv.gz').y.to_numpy()
  obs={}
  for (u,d),val in zip(pack['observed_x'].astype(int),pack['observed_y']):obs.setdefault(u,[]).append(float(val))
  score=np.array([np.mean(obs[int(u)]) for u in pack['query_x'][:,0]])
  assert all(len(obs[int(u)])==4 for u in pack['query_x'][:,0]);pred=score>=.5
  tp=int(((y==0)&~pred).sum());fn=int(((y==0)&pred).sum());fp=int(((y==1)&~pred).sum());tn=int(((y==1)&pred).sum())
  assert counts[(ds,folder.name)]==(tp+fn,tn+fp)
  for k in [0,10,20,30,40,50]:rows.append(dict(dataset=ds,combination=folder.name,model='AST mean',support=k,seed=0,repeat=0,AUROC=roc_auc_score(y,score) if len(set(y))==2 else np.nan,**metrics(tp,fn,fp,tn)))
f=pd.DataFrame(rows);metric=['AUROC','accuracy','sensitivity_S','F1_S','specificity','BA','precision_nonS','F1_nonS','false_susceptible_rate']
combo=f.groupby(['dataset','combination','model','support'],as_index=False)[metric].mean()
combo['n_nonS']=[counts[(d,c)][0] for d,c in zip(combo.dataset,combo.combination)];combo['n_S']=[counts[(d,c)][1] for d,c in zip(combo.dataset,combo.combination)]
summary=[]
for ds in ['mimic','eicu']:
 for minimum in [0,2,5]:
  selected=combo[(combo.dataset==ds)&(combo.n_nonS>=minimum)]
  for (model,k),g in selected.groupby(['model','support']):
   r=dict(dataset=ds,min_nonS=minimum,model=model,support=int(k),combinations=len(g))
   for m in metric:r[m]=g[m].mean();r[m+'_defined']=int(g[m].notna().sum())
   summary.append(r)
summary=pd.DataFrame(summary)
combo.to_json(W/'combination_metrics.json',orient='records',indent=2,double_precision=12)
summary.to_json(W/'sensitivity_summary.json',orient='records',indent=2,double_precision=12)
coverage=[]
for ds in ['mimic','eicu']:
 for minimum in [0,2,5]:
  patients=set();episodes=set();n=0;cells=0
  for (d,c),(nr,ns) in counts.items():
   if d!=ds or nr<minimum:continue
   ix=json.loads((B/'inputs'/c/(ds+'_index.json')).read_text());patients.update(map(str,ix['query_patients']));episodes.update(map(str,ix['query_episodes']));n+=1;cells+=nr+ns
  coverage.append(dict(dataset=ds,min_nonS=minimum,combinations=n,patients=len(patients),episodes=len(episodes),target_observations=cells))
(W/'coverage.json').write_text(json.dumps(coverage,indent=2))
print('PRIMARY TABLE REPRODUCTION, support 50:')
print(summary[(summary.min_nonS==0)&(summary.support==50)&summary.model.isin(['CDLD','MF','AST mean'])][['dataset','model']+metric].round(4).to_string(index=False))
print('\neICU sensitivity analysis, support 50:')
print(summary[(summary.dataset=='eicu')&(summary.support==50)&summary.model.isin(['CDLD','MF','AST mean'])][['min_nonS','model','combinations','AUROC','BA','specificity']].round(4).to_string(index=False))
print('\nCoverage:',json.dumps(coverage))
print('\nAdaptation BA and nonsusceptibility:')
print(summary[(summary.min_nonS==0)&summary.support.isin([0,50])&summary.model.isin(['CDLD','MF'])][['dataset','model','support','BA','F1_nonS','false_susceptible_rate']].round(4).to_string(index=False))
# Compare the canonical source's accuracy against confusion-matrix reconstruction.
assert max(abs(metrics(r['TN'],r['FP'],r['FN'],r['TP'])['accuracy']-r['accuracy']) for r in source if r['model'] in names)<1e-8
