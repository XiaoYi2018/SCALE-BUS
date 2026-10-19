from metrology_common import (
    CACHE,
    COLUMNS,
    Image,
    Path,
    ROOT,
    SEED,
    THRESHOLD,
    csv,
    ensemble,
    external_frame,
    extract_advanced_feature_dict,
    finish,
    hashlib,
    json,
    logit,
    newjson,
    np,
    patient_equal_weights,
    pd,
    platform,
    resultdir,
    rng_for,
    sha,
    time,
)
from concurrent.futures import ProcessPoolExecutor,as_completed
from extract_v06_random_boundary_perturbations import perturb_mask
from sklearn.metrics import roc_auc_score
import argparse,traceback
OUT=resultdir('uncertainty_propagation')
AMPS=(.01,.025,.05)
SPECS=[(s,0.,20) for s in ('speckle','gamma','resolution')]+[(s,a,n) for a in AMPS for s,n in [('contour',20),('joint',50)]]
COLS=COLUMNS['advanced76']

def raw(record):
 with Image.open(record['image_path']) as h:im=np.asarray(h.convert('L'))
 with Image.open(record['mask_path']) as h:mask=np.asarray(h.convert('L'))
 return im,mask

def perturbed(im,mask,record,source,amp,rep):
 rng=rng_for('E3',record['dataset'],record['image'],source,amp,rep)
 seed=int(rng.integers(1,2**32-2));sigma=float(rng.uniform(0,.10));gamma=float(rng.uniform(.8,1.2));factor=float(rng.uniform(.5,1.))
 target=perturb_mask(mask,amp,seed,str(record['image'])) if source in ('joint','contour') else mask
 x=im
 if source in ('joint','resolution'):
  p=Image.fromarray(im);small=p.resize((max(1,int(p.width*factor)),max(1,int(p.height*factor))),Image.Resampling.BILINEAR);x=np.asarray(small.resize(p.size,Image.Resampling.BILINEAR))
 if source in ('joint','speckle','gamma'):
  v=x.astype(float)/255
  if source in ('joint','speckle'):v=v+v*rng.normal(0,sigma,v.shape)
  if source in ('joint','gamma'):v=np.sign(v)*np.power(np.abs(v),gamma)
  x=np.clip(v*255,0,255).astype(np.uint8)
 a=mask>0;b=target>0;dice=2*np.logical_and(a,b).sum()/(a.sum()+b.sum())
 return x,target,[seed,sigma,gamma,factor,dice]

def extract(im,mask):
 f=extract_advanced_feature_dict(im,mask);return np.asarray([f[c] for c in COLS])

def one_benchmark(record):
 t=time.perf_counter();im,mask=raw(record);clean=extract(im,mask)
 delta=float(np.max(np.abs(clean-np.array([record[c] for c in COLS],float))))
 t1=time.perf_counter();x,m,_=perturbed(im,mask,record,'joint',.025,0);extract(x,m);elapsed=time.perf_counter()-t1
 return {'dataset':record['dataset'],'image':record['image'],'single_joint_seconds':elapsed,'clean_max_absolute_difference':delta,'total_seconds':time.perf_counter()-t}

def case_path(r):return OUT/'cases'/('case_'+r['dataset']+'_'+hashlib.sha256(str(r['image']).encode()).hexdigest()[:20]+'.npz')

def one_case(record):
 dest=case_path(record)
 if dest.exists():
  with np.load(dest,allow_pickle=False) as a:
   assert str(a['script_sha256'])==sha(__file__),'Case cache script mismatch';assert str(a['cache_sha256'])==sha(CACHE);assert str(a['common_sha256'])==sha(Path(__file__).with_name('metrology_common.py'));assert list(a['columns'])==COLS;assert a['X'].shape==(270,76)
  return {'dataset':record['dataset'],'image':record['image'],'reused':True,'file':dest.name}
 t=time.perf_counter();im,mask=raw(record);base=extract(im,mask);delta=float(np.max(np.abs(base-np.asarray([record[c] for c in COLS],float))))
 if delta>1e-7:raise RuntimeError('Frozen feature identity mismatch '+str(delta)+' '+str(record['image']))
 xs=[];meta=[]
 for source,amp,n in SPECS:
  for rep in range(n):
   x,m,pars=perturbed(im,mask,record,source,amp,rep);xs.append(extract(x,m));meta.append((source,amp,rep,*pars))
 X=np.asarray(xs);p=ensemble(X);z=logit(np.clip(p,1e-6,1-1e-6))
 with dest.open('xb') as h:np.savez_compressed(h,X=X,p=p,z=z,base=base,base_p=ensemble(base[None]),source=np.asarray([m[0] for m in meta]),amplitude=np.asarray([m[1] for m in meta]),rep=np.asarray([m[2] for m in meta]),parameters=np.asarray([m[3:] for m in meta]),script_sha256=np.asarray(sha(__file__)),cache_sha256=np.asarray(sha(CACHE)),common_sha256=np.asarray(sha(Path(__file__).with_name('metrology_common.py'))),columns=np.asarray(COLS),seconds=np.asarray(time.perf_counter()-t))
 return {'dataset':record['dataset'],'image':record['image'],'reused':False,'seconds':time.perf_counter()-t,'file':dest.name,'clean_max_absolute_difference':delta}

def collect(D,workers):
 start=time.perf_counter();(OUT/'cases').mkdir(exist_ok=True)
 bench=[]
 for dataset in D:
  f=D[dataset];n=100;ix=rng_for('E3benchmark',dataset).choice(len(f),n,replace=False);bench.extend(f.iloc[sorted(ix)].to_dict('records'))
 t=time.perf_counter()
 with ProcessPoolExecutor(max_workers=workers) as ex:b=list(ex.map(one_benchmark,bench,chunksize=1))
 wall=time.perf_counter()-t;csv(OUT/'benchmark_200.csv',pd.DataFrame(b))
 if max(x['clean_max_absolute_difference'] for x in b)>1e-7:raise RuntimeError('Benchmark clean feature identity gate failed')
 # Domain-weight single perturbation timing; measure parallel efficiency independently.
 per_domain={d:{'joint_mean':float(np.mean([r['single_joint_seconds'] for r in b if r['dataset']==d])),'per_case_overhead':float(np.mean([r['total_seconds']-r['single_joint_seconds'] for r in b if r['dataset']==d]))} for d in D}
 efficiency=min(1.,sum(r['total_seconds'] for r in b)/(workers*wall))
 expected_cpu=sum(len(f)*(270*per_domain[d]['joint_mean']+per_domain[d]['per_case_overhead']) for d,f in D.items())
 estimated_full_hours=expected_cpu/(workers*efficiency)/3600
 records=pd.concat(D.values(),ignore_index=True).to_dict('records');csv(OUT/'selected_cases.csv',pd.DataFrame(records)[['dataset','image','patient_id','target','device']])
 timing={'n_benchmark':200,'workers':workers,'benchmark_wall_seconds':wall,'mean_single_joint_seconds':float(np.mean([r['single_joint_seconds'] for r in b])),'estimated_full_hours':estimated_full_hours,'n_selected':len(records),'per_domain_timing':per_domain,'observed_parallel_efficiency':efficiency,'formula':'sum_d n_d*(270*mean_joint_d+mean_once_per_case_overhead_d)/(workers*observed_efficiency*3600)','cpu':platform.processor(),'seed':SEED,'draws_per_image':270}
 newjson(OUT/'runtime_decision.json',timing);print('RUNTIME_DECISION',json.dumps(timing),flush=True)
 with ProcessPoolExecutor(max_workers=workers) as ex:
  iterator=iter(records);pending={}
  for _ in range(workers):
   r=next(iterator,None)
   if r is not None:pending[ex.submit(one_case,r)]=r
  with (OUT/'case_progress.jsonl').open('x',encoding='utf-8') as log:
   k=0
   while pending:
    future=next(as_completed(pending));r=pending.pop(future);k+=1
    try:v=future.result()
    except BaseException as e:
     newjson(OUT/('failure_'+r['dataset']+'_'+str(k)+'.json'),{'record':{c:r[c] for c in ['dataset','image','patient_id']},'error':repr(e),'traceback':traceback.format_exc()})
     for f in pending:f.cancel()
     raise
    log.write(json.dumps(v)+'\n');log.flush()
    if k%10==0 or k==len(records):print('E3_CASES',k,'/',len(records),'elapsed',round(time.perf_counter()-start,1),flush=True)
    r=next(iterator,None)
    if r is not None:pending[ex.submit(one_case,r)]=r
 newjson(OUT/'collection_complete.json',{'status':'COLLECTED','images':len(records),'seconds':time.perf_counter()-start,'script_sha256':sha(__file__)})

def analyze():
 start=time.perf_counter();cases=pd.read_csv(OUT/'selected_cases.csv',dtype={'patient_id':str});features=[];scores=[];budgets=[];blocks=[];dices=[];aucs=[]
 blockids={'GF':[i for i,c in enumerate(COLS) if c.startswith('advanced_mf_')],'W':[i for i,c in enumerate(COLS) if c.startswith('advanced_wavelet_')],'S':[i for i,c in enumerate(COLS) if c.startswith('advanced_shape_')]}
 for dataset,frame in cases.groupby('dataset',sort=False):
  pjoint={a:[] for a in AMPS}
  for r in frame.to_dict('records'):
   with np.load(case_path(r),allow_pickle=False) as ar:
    assert str(ar['script_sha256'])==sha(__file__);assert str(ar['cache_sha256'])==sha(CACHE);assert str(ar['common_sha256'])==sha(Path(__file__).with_name('metrology_common.py'));assert list(ar['columns'])==COLS
    X=ar['X'];p=ar['p'];z=ar['z'];base=ar['base'];source=ar['source'];amplitude=ar['amplitude'];parameters=ar['parameters']
   identity={c:r[c] for c in ['dataset','image','patient_id','target']};sourcevar={}
   for s,a,_ in SPECS:
    ix=(source==s)&(amplitude==a);xx=X[ix];pp=p[ix];zz=z[ix];sourcevar[(s,a)]=float(np.var(zz,ddof=1))
    for j,c in enumerate(COLS):features.append({**identity,'source':s,'contour_sigma':a,'variable':c,'u':float(np.std(xx[:,j],ddof=1)),'mean':float(xx[:,j].mean()),'q025':float(np.quantile(xx[:,j],.025)),'q975':float(np.quantile(xx[:,j],.975))})
    for output,v in [('probability',pp),('logit',zz)]:
     lo,hi=np.quantile(v,[.025,.975]);th=THRESHOLD['advanced76'] if output=='probability' else logit(THRESHOLD['advanced76'])
     scores.append({**identity,'source':s,'contour_sigma':a,'output':output,'u':float(np.std(v,ddof=1)),'mean':float(v.mean()),'q025':float(lo),'q975':float(hi),'threshold_crossing':bool(lo<=th<=hi)})
    for block,ids in blockids.items():
     xb=np.repeat(base[None],len(xx),axis=0);xb[:,ids]=xx[:,ids];zb=logit(np.clip(ensemble(xb),1e-6,1-1e-6));blocks.append({**identity,'source':s,'contour_sigma':a,'block':block,'variance':float(np.var(zb,ddof=1)),'approximation':'only this feature block varies; all other blocks fixed at clean values'})
    if s in ('contour','joint'):
     for k,value in enumerate(parameters[ix,-1]):dices.append({**identity,'source':s,'contour_sigma':a,'repeat':k,'Dice':float(value)})
    if s=='joint':pjoint[a].append(pp)
   for a in AMPS:
    total=sourcevar[('joint',a)];single={s:sourcevar[(s,a if s=='contour' else 0.)] for s in ('contour','speckle','gamma','resolution')};single['interaction']=total-sum(single.values())
    for s,v in single.items():budgets.append({**identity,'contour_sigma':a,'source':s,'variance':v,'joint_variance':total,'variance_fraction':v/total if total>0 else np.nan})
  w=patient_equal_weights(frame);y=frame.target.to_numpy()
  for a in AMPS:
   matrix=np.asarray(pjoint[a]);assert matrix.shape==(len(frame),50)
   for rep in range(50):aucs.append({'dataset':dataset,'contour_sigma':a,'repeat':rep,'auc_group_weighted':roc_auc_score(y,matrix[:,rep],sample_weight=w),'auc_image':roc_auc_score(y,matrix[:,rep])})
 f=pd.DataFrame(features);s=pd.DataFrame(scores);b=pd.DataFrame(budgets);bl=pd.DataFrame(blocks);di=pd.DataFrame(dices);ac=pd.DataFrame(aucs)
 sampling=[]
 original=json.loads((ROOT/'results/v04_locked_external/locked_external_summary.json').read_text())
 for dataset,fr in cases.groupby('dataset',sort=False):
  ci=original['bootstrap_vs_frozen98'][dataset]['auc_intervals']['advanced76_logistic'];origin='existing full-cohort original grouped bootstrap'
  sampling.append({'dataset':dataset,'n_images':len(fr),'sampling_auc_low':ci['low'],'sampling_auc_high':ci['high'],'source':origin})
 csv(OUT/'sampling_auc_comparison.csv',pd.DataFrame(sampling))
 for name,df in [('per_image_feature_uncertainty',f),('per_image_score_uncertainty',s),('variance_budget_per_image',b),('feature_block_budget_per_image',bl),('contour_dice',di),('measurement_auc_draws',ac)]:csv(OUT/(name+'.csv'),df)
 bs=b.groupby(['dataset','contour_sigma','source']).agg(mean_variance=('variance','mean'),mean_joint_variance=('joint_variance','mean'),median_case_fraction=('variance_fraction','median')).reset_index();bs['ratio_of_mean_variances']=bs.mean_variance/bs.mean_joint_variance;csv(OUT/'variance_budget_summary.csv',bs)
 ds=di.groupby(['dataset','contour_sigma','source']).Dice.agg(['median',lambda x:x.quantile(.25),lambda x:x.quantile(.75)]).reset_index();ds.columns=['dataset','contour_sigma','source','median','q25','q75'];csv(OUT/'dice_summary.csv',ds)
 cs=s.loc[(s.source=='joint')&(s.output=='probability')].groupby(['dataset','contour_sigma','target']).agg(images=('image','size'),crossing_fraction=('threshold_crossing','mean'),median_u=('u','median')).reset_index();csv(OUT/'decision_uncertainty_summary.csv',cs)
 aas=ac.groupby(['dataset','contour_sigma']).auc_group_weighted.agg(['median',lambda x:x.quantile(.025),lambda x:x.quantile(.975)]).reset_index();aas.columns=['dataset','contour_sigma','median_auc','q025','q975'];csv(OUT/'measurement_auc_summary.csv',aas)
 csv(OUT/'feature_block_budget_summary.csv',bl.groupby(['dataset','contour_sigma','source','block']).variance.agg(['mean','median']).reset_index())
 finish(OUT,__file__,{'status':'DONE','analysis_seconds':time.perf_counter()-start,'runtime_decision':json.loads((OUT/'runtime_decision.json').read_text()),'collection':json.loads((OUT/'collection_complete.json').read_text()),'parameters':{'seed':SEED,'contour_amplitudes':AMPS,'primary_contour_sigma':.025,'single_draws':20,'joint_draws_each_sigma':50,'threshold':THRESHOLD['advanced76'],'threshold_source':'results/v04_locked_external/locked_external_summary.json source OOF Youden','image_order':'native resolution: PIL bilinear down/up -> multiplicative speckle -> gamma -> clip -> uint8 -> production extraction','gamma_negative_extension':'sign(v)*abs(v)**gamma before final clipping; identical to zero-domain gamma after final clipping','contour_function':'extract_v06_random_boundary_perturbations.perturb_mask unchanged','uncertainty_interval':'empirical 2.5-97.5 percentile; logit output for budgets','AUC':'patient-equal weighted, each of 50 draws; compare sampling interval separately','block_budget':'approximate only-block-vary scoring','interaction':'joint variance minus sum of four single-source variances; negative retained'},'sampling_intervals':sampling,'auc':aas.to_dict('records'),'dice':ds.to_dict('records'),'decision_crossings':cs.to_dict('records'),'budget':bs.to_dict('records')})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--analyze-only',action='store_true');a=ap.parse_args()
 if not a.analyze_only:
  D={d:external_frame(d) for d in ('busbra','breast')}
  for d,f in D.items():f['dataset']=d
  collect(D,a.workers)
 analyze()
if __name__=='__main__':main()
