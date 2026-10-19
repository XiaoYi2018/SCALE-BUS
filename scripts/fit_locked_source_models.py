from __future__ import annotations
import sys,time,json,hashlib,platform
from pathlib import Path
sys.dont_write_bytecode=True
from metrology_paths import ROOT, OUTPUT_ROOT
OUT=OUTPUT_ROOT/'v07_positive_control'
if OUT.exists():raise RuntimeError('New output directory already exists; refuse overwrite')
import numpy as np,pandas as pd,sklearn,scipy,skimage,joblib
from sklearn.metrics import roc_auc_score
from evaluate_v04_locked_external import external_frame,build_logistic
from screen_v04_embedding import prepare_metadata
from screen_v04_handcrafted import feature_columns
from run_busuclm_grouped_cv import patient_equal_weights,SEED
START=time.perf_counter()
OUT.mkdir(parents=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(name,value):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
sources={d:prepare_metadata(d) for d in ['busuclm','busi']};targets={d:external_frame(d) for d in ['busbra','breast']}
assert len(sources['busuclm'])==115 and sources['busuclm'].patient_id.nunique()==29
assert len(sources['busi'])==622 and sources['busi'].patient_id.nunique()==595
models={};columns={};tuning={};prediction={d:f[['image','patient_id','label','target']].copy() for d,f in targets.items()}
for group,n in [('advanced76',76),('frozen98',98)]:
 cols=feature_columns(sources['busuclm'],group);assert len(cols)==n;columns[group]=cols;models[group]={};tuning[group]={}
 for d,frame in {**sources,**targets}.items():assert feature_columns(frame,group)==cols,(d,group)
 for source,frame in sources.items():
  frame['domain_group']=frame['patient_id'];model,c,rows=build_logistic(frame,cols);models[group][source]=model;tuning[group][source]={'C':c,'rows':rows}
  print('FITTED',group,source,'C',c,flush=True)
  for d,f in targets.items():prediction[d][f'probability_{group}_{source}']=model.predict_proba(f[cols])[:,1]
 for d,f in prediction.items():f[f'probability_{group}_ensemble']=f[[f'probability_{group}_{x}' for x in sources]].mean(axis=1)
expected={('busbra','advanced76'):'0.8985',('breast','advanced76'):'0.8222',('busbra','frozen98'):'0.8538',('breast','frozen98'):'0.7879'}
records=[];max_delta=0.0
for d,f in prediction.items():
 old=pd.read_csv(ROOT/'results'/'v04_locked_external'/f'{d}_locked_predictions.csv');old['patient_id']=old['patient_id'].astype(str)
 keys=['image','patient_id','label','target'];assert not f.duplicated(keys).any() and not old.duplicated(keys).any()
 pair=f.merge(old,on=keys,validate='one_to_one',suffixes=('_new','_old'));assert len(pair)==len(f)==len(old)
 for group in ['advanced76','frozen98']:
  for suffix in ['busuclm','busi','ensemble']:
   newcol=f'probability_{group}_{suffix}';oldcol=f'probability_{group}_logistic_{suffix}'
   a=pair[newcol].to_numpy();b=pair[oldcol].to_numpy();delta=float(np.max(np.abs(a-b)));max_delta=max(max_delta,delta)
  auc=float(roc_auc_score(f.target,f[f'probability_{group}_ensemble'],sample_weight=patient_equal_weights(f)))
  records.append({'dataset':d,'group':group,'auc':auc,'rounded':f'{auc:.4f}','expected':expected[d,group],'passed':f'{auc:.4f}'==expected[d,group]})
 with (OUT/f'{d}_predictions.csv').open('x',newline='',encoding='utf-8') as handle:f.to_csv(handle,index=False)
passed=all(x['passed'] for x in records)
# Save source models only after successful reproduction; no target-label fitting.
if passed:
 with (OUT/'locked_source_models.joblib').open('xb') as f:joblib.dump({'models':models,'columns':columns,'seed':SEED},f,compress=3)
write_json('source_tuning.json',tuning)
report={'status':'PASS' if passed else 'BLOCKED','auc':records,'max_abs_probability_delta_vs_original':max_delta,'source_counts':{d:{'images':len(f),'groups':int(f.patient_id.nunique())} for d,f in sources.items()},'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{m.__name__:m.__version__ for m in [np,pd,sklearn,scipy,skimage,joblib]},'script':Path(__file__).name,'script_sha256':digest(Path(__file__)),'seconds':time.perf_counter()-START}
write_json('positive_control.json',report)
manifest=[{'file':p.name,'bytes':p.stat().st_size,'sha256':digest(p),'generator':Path(__file__).name,'generator_sha256':digest(Path(__file__))} for p in OUT.iterdir() if p.is_file()]
write_json('manifest.json',manifest)
print(json.dumps(report,indent=2),flush=True)
raise SystemExit(0 if passed else 3)
