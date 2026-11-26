from paths import *
import sys, subprocess, concurrent.futures, time
os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_ENABLE_ONEDNN_OPTS='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',TF_CPP_MIN_LOG_LEVEL='3')
assert sys.platform.startswith('linux'), 'Use the Linux CPU environment in requirements.txt.'
settings=next(s for s in PROTOCOL['settings'] if s['id']=='CDLD')
os.environ.update(FINDER_L2=str(settings['finder_l2']),FINDER_DROPOUT=str(settings['finder_dropout_multiplier']))
def run(job):
 model,condition,context,seed=job
 inp=W/'inputs'/condition/context
 dest=W/'predictions'/model/condition/context/f'seed{seed}'
 if (dest/'eicu/completion.json').exists() and (dest/'mimic/completion.json').exists():return job
 log=W/'logs'/f'{model}_{condition}_{context}_{seed}.txt';log.parent.mkdir(exist_ok=True)
 if model=='CDLD':
  folder=B/'runs/CDLD'/context
  assert (folder/f'seed{seed}').is_dir(), 'Run 02_models.ipynb first.'
  args=[folder,seed,settings['cycle'],settings['epoch']]
  commands=[[B/'code/adapt_cdld.py',*args,inp/'mimic_adapt_inputs.npz',dest/'mimic',B/'cdld_adaptation.json'],[B/'code/external_cdld.py',*args,inp/'eicu_observed_inputs.npz',dest/'eicu',B/'cdld_adaptation.json',dest/'mimic']]
 else:
  checkpoint=B/'mf_runs'/context/f'seed{seed}/model/checkpoint_200'
  assert checkpoint.with_suffix('.npz').is_file(), 'Run 02_models.ipynb first.'
  commands=[[B/'mf_code/adapt_mf.py',checkpoint,inp/'mimic_adapt_inputs.npz',dest/'mimic',B/'mf_cases.json'],[B/'mf_code/external_mf.py',checkpoint,inp/'eicu_observed_inputs.npz',dest/'eicu',B/'mf_cases.json',dest/'mimic']]
 with log.open('w') as stream:
  for command in commands:subprocess.run([sys.executable,*map(str,command)],stdout=stream,stderr=subprocess.STDOUT,check=True)
 return job
jobs=[(m,c,x,s) for m in ['CDLD','MF'] for c in ['all_existing','matched_four','exclude_family_four'] for x in PROTOCOL['contexts'] for s in PROTOCOL['seeds']]
with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('CDLD_WORKERS','1'))) as pool:
 for i,future in enumerate(concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]),1):
  print(i,'/',len(jobs),future.result(),flush=True)
(W/'adaptation_complete.json').write_text(json.dumps({'jobs':len(jobs)}))
