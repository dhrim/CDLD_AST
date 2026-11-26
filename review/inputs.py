from paths import *
import hashlib, numpy as np, pandas as pd
raw={d:pd.read_csv(OUTPUT/'preprocessing/cohort'/f'{d}_common_01.csv.gz') for d in ['mimic','eicu']}
audit=[]
for inp in sorted((B/'inputs').iterdir()):
 if not inp.is_dir():continue
 entity=json.loads((inp/'entities.json').read_text());nu=len(entity['episodes']);drugs={d:i for i,d in enumerate(entity['drugs'])}
 dest=W/'inputs/all_existing'/inp.name;dest.mkdir(parents=True,exist_ok=True)
 for ds in ['mimic','eicu']:
  ix=json.loads((inp/f'{ds}_index.json').read_text());eps=ix['episodes'] if ds=='mimic' else ix['query_episodes'];ids={e:nu+i for i,e in enumerate(eps)}
  filename='mimic_adapt_inputs.npz' if ds=='mimic' else 'eicu_observed_inputs.npz';old=np.load(inp/filename);arrays={k:old[k] for k in old.files}
  rows=raw[ds][(raw[ds].species==ix['species'])&raw[ds].episode_id.isin(ids)&raw[ds].drug.isin(drugs)].copy()
  assert not rows.duplicated(['episode_id','drug']).any()
  rows['u']=rows.episode_id.map(ids);rows['d']=rows.drug.map(drugs);rows=rows.sort_values(['u','d'])
  allvalues={(int(r.u),int(r.d)):float(r.y) for r in rows.itertuples()}
  for pair,y in zip(old['observed_x'],old['observed_y']):assert allvalues[tuple(map(int,pair))]==float(y)
  arrays['observed_x']=rows[['u','d']].to_numpy(dtype=old['observed_x'].dtype);arrays['observed_y']=rows.y.to_numpy(dtype=old['observed_y'].dtype)
  assert set(rows.u)==set(ids.values());assert rows.groupby('u').size().min()>=4
  np.savez_compressed(dest/filename,**arrays)
  (dest/f'{ds}_index.json').write_text(json.dumps(ix,indent=2))
  counts=rows.groupby('u').size();audit.append(dict(combination=inp.name,dataset=ds,episodes=len(eps),original_observations=len(old['observed_x']),all_observations=len(rows),minimum=int(counts.min()),median=float(counts.median()),maximum=int(counts.max()),support_and_query_arrays_unchanged=all(np.array_equal(arrays[k],old[k]) for k in old.files if k not in ['observed_x','observed_y'])))
(W/'all_existing_audit.json').write_text(json.dumps(audit,indent=2))
print('Prepared',len(audit),'packs. Original four observations retained; support/query arrays unchanged.')

family=json.loads((PACKAGE/'config/drug_families.json').read_text())
(W/'family_definition.json').write_text(json.dumps(family,indent=2))
lookup={d:f for f,drugs in family['families'].items() for d in drugs}
raw={'mimic':pd.read_csv(OUTPUT/'preprocessing/splits/mimic_split_cells_01.csv.gz'),'eicu':pd.read_csv(OUTPUT/'preprocessing/cohort/eicu_common_01.csv.gz')}
def rank(x,s):return hashlib.sha256(f'{s}|{x}'.encode()).hexdigest()
audit=[]
for inp in sorted((B/'inputs').iterdir()):
 if not inp.is_dir():continue
 en=json.loads((inp/'entities.json').read_text());nu=len(en['episodes']);drugs={d:i for i,d in enumerate(en['drugs'])};mi=json.loads((inp/'mimic_index.json').read_text());target=mi['target'];species=mi['species'];assert target in lookup
 forbidden={d for d in drugs if lookup[d]==lookup[target]};supports={}
 for ds in ['mimic','eicu']:
  ix=json.loads((inp/f'{ds}_index.json').read_text());g=raw[ds][raw[ds].species==species];known=g[g.drug.isin(drugs)].copy();outside=known[~known.drug.isin(forbidden)];counts=outside.groupby('episode_id').drug.nunique();eligible=set(counts[counts>=4].index)
  query=[e for e in ix['query_episodes'] if e in eligible];patient_map=dict(zip(ix['query_episodes'],ix['query_patients']))
  if ds=='mimic':
   pool=g[(g.drug==target)&(g.split=='adaptation')&g.episode_id.isin(eligible)].copy()
   assert pool.patient_id.nunique()>=50
   for r in range(10):
    ordered=pool.assign(order=pool.episode_id.map(lambda e:rank(e,f'episode-{r}'))).sort_values('order').drop_duplicates('patient_id')
    ordered=ordered.assign(order=ordered.patient_id.map(lambda p:rank(p,f'patient-{r}'))).sort_values('order').head(50)
    supports[r]=ordered
   eps=sorted(set(query)|set(e for frame in supports.values() for e in frame.episode_id))
   assert set(pool.patient_id).isdisjoint(patient_map.values())
  else:eps=query
  ids={e:nu+i for i,e in enumerate(eps)}
  for condition in ['matched_four','exclude_family_four']:
   source=known if condition=='matched_four' else outside
   ox=[];oy=[]
   for ep,rows in source[source.episode_id.isin(ids)].groupby('episode_id'):
    selected=rows.assign(order=rows.drug.map(lambda d:rank(f'{ep}|{d}','dev118-k4'))).sort_values('order').head(4)
    assert len(selected)==4
    for row in selected.itertuples():ox.append([ids[ep],drugs[row.drug]]);oy.append(row.y)
   arrays=dict(observed_x=np.array(ox,dtype='float32').reshape(-1,2),observed_y=np.array(oy,dtype='float32'),query_x=np.array([[ids[e],len(drugs)] for e in query],dtype='float32').reshape(-1,2),new_rows=np.int64(len(eps)))
   if ds=='mimic':
    for r,frame in supports.items():
     for budget in [10,20,30,40,50]:
      selected=frame.head(budget);arrays[f'x_{r}_{budget}']=np.array([[ids[e],len(drugs)] for e in selected.episode_id],dtype='float32');arrays[f'y_{r}_{budget}']=selected.y.to_numpy(dtype='float32')
   dest=W/'inputs'/condition/inp.name;dest.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(dest/('mimic_adapt_inputs.npz' if ds=='mimic' else 'eicu_observed_inputs.npz'),**arrays)
   (dest/f'{ds}_index.json').write_text(json.dumps(dict(species=species,target=target,episodes=eps,query_episodes=query,query_patients=[patient_map[e] for e in query]),indent=2))
   if condition=='matched_four':reference=arrays
   else:
    assert all(np.array_equal(reference[k],v) for k,v in arrays.items() if k not in ['observed_x','observed_y'])
    assert not set(arrays['observed_x'][:,1].astype(int))&{drugs[d] for d in forbidden}
  audit.append(dict(dataset=ds,combination=inp.name,family=lookup[target],excluded_drugs=sorted(forbidden),original_query=len(ix['query_episodes']),matched_query=len(query),matched_rows=len(eps),support_sizes=[10,50] if ds=='mimic' else [],paired_support_query_identical=True))
(W/'family_exclusion_audit.json').write_text(json.dumps(audit,indent=2));print('Prepared',len(audit)*2,'paired packs; family exclusion and shared support/query verified.')
