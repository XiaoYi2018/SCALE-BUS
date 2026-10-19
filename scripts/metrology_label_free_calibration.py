from metrology_common import (
    COLUMNS,
    MODELS,
    ROOT,
    csv,
    expit,
    finish,
    frames,
    logit,
    np,
    pd,
    report_scores,
    resultdir,
    source_oof,
    time,
)
from sklearn.linear_model import LogisticRegression
from run_busuclm_grouped_cv import valid_group_splits,SEED as TRAIN_SEED
from run_v04_four_dataset_transfer import build_model
OUT=resultdir('label_free_calibration');START=time.perf_counter()
D=frames();rows=[];platt_records=[];em_records=[]
def fit_platt(p,y):
 m=LogisticRegression(C=1e6,solver='lbfgs',max_iter=5000,random_state=TRAIN_SEED)
 m.fit(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1),y);return m
def apply(m,p):return m.predict_proba(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1))[:,1]
def em(p,pi0):
 p=np.clip(np.asarray(p),1e-6,1-1e-6);pi=float(pi0)
 for i in range(1000):
  z=expit(logit(p)+logit(np.clip(pi,1e-6,1-1e-6))-logit(pi0));new=float(np.clip(z.mean(),1e-6,1-1e-6))
  if abs(new-pi)<1e-6:pi=new;break
  pi=new
 return expit(logit(p)+logit(pi)-logit(pi0)),{'estimated_prior':pi,'iterations':i+1,'converged':i<999,'reference_prior':pi0}
def evaluate(scope,d,f,group,raw,cal,prior_raw,prior_cal):
 pc,ec=em(raw,prior_raw);pd_,ed=em(cal,prior_cal);variants={'a':raw,'b':cal,'c':pc,'d':pd_};baseauc=report_scores(f,raw,group)['group_balanced_roc_auc']
 out=f[['image','patient_id','target']].copy()
 for v,p in variants.items():
  m=report_scores(f,p,group);m.update(scope=scope,dataset=d,group=group,variant=v,auc_difference_from_raw=m['group_balanced_roc_auc']-baseauc,true_image_prior=float(f.target.mean()),training_prior=prior_cal if v in 'bd' else prior_raw)
  if v in ['c','d']:m.update(ec if v=='c' else ed)
  rows.append(m);out['p_'+v]=p
 for v,e in [('c',ec),('d',ed)]:em_records.append({'scope':scope,'dataset':d,'group':group,'variant':v,**e})
 csv(OUT/f'{scope}_{d}_{group}_predictions.csv',out)
 print('CALIBRATION',scope,d,group,'a',rows[-4]['brier'],'d',rows[-1]['brier'],'dEce',rows[-1]['ece_10'],flush=True)
# Original grouped-nested source OOF means are read-only.
for group in ['advanced76','frozen98']:
 calibrators={};sourcepriors={}
 for source in ['busuclm','busi']:
  oof=source_oof(group+'_logistic',source);m=fit_platt(oof.probability.to_numpy(),oof.target.to_numpy());calibrators[source]=m;sourcepriors[source]=float(D[source].target.mean());platt_records.append({'scope':'locked','source':source,'group':group,'intercept':float(m.intercept_[0]),'slope':float(m.coef_[0,0]),'OOF_file':str(ROOT/'results/v04_confirmation'/f'{group}_logistic'/f'{source}_mean_oof_predictions.csv'),'reference_image_prior':sourcepriors[source]})
 for d in ['busbra','breast']:
  f=D[d];cols=COLUMNS[group];p=[MODELS[group][source].predict_proba(f[cols])[:,1] for source in ['busuclm','busi']];cal=np.mean([apply(calibrators[source],pp) for source,pp in zip(['busuclm','busi'],p)],axis=0);evaluate('locked',d,f,group,np.mean(p,axis=0),cal,.5,float(np.mean(list(sourcepriors.values()))))
csv(OUT/'locked_calibration_metrics.csv',pd.DataFrame(rows))
# LODO calibrators use only repeated grouped-nested OOF from their three source datasets.
for d in D:
 names=[x for x in D if x!=d];parts=[]
 for name in names:
  f=D[name].copy();f['dataset']=name;f['patient_id']=name+':'+f.patient_id.astype(str);f['domain_group']=f.patient_id;parts.append(f)
 source=pd.concat(parts,ignore_index=True);source['domain_group']=source.patient_id
 old=pd.read_csv(ROOT/'results/v04_four_dataset_transfer'/f'lodo_{d}_predictions.csv');old.patient_id=old.patient_id.astype(str)
 f=D[d].copy();old=old.set_index(['image','patient_id']).loc[pd.MultiIndex.from_frame(f[['image','patient_id']])].reset_index();assert np.array_equal(old.target,f.target)
 for group in ['advanced76','frozen98']:
  cols=COLUMNS[group];pred=np.zeros((10,len(source)))
  for repeat in range(10):
   folds,_=valid_group_splits(source,5,TRAIN_SEED+repeat*1000)
   for tr,va in folds:
    model,c,_=build_model(source.iloc[tr].copy(),cols,4);pred[repeat,va]=model.predict_proba(source.iloc[va][cols])[:,1]
   print('LODO_OOF',d,group,repeat+1,flush=True)
  mean=pred.mean(axis=0);m=fit_platt(mean,source.target.to_numpy());prior=float(source.target.mean());raw=old['probability_'+group].to_numpy();cal=apply(m,raw)
  csv(OUT/f'lodo_{d}_{group}_source_oof.csv',pd.DataFrame({'dataset':source.dataset,'image':source.image,'patient_id':source.patient_id,'target':source.target,'probability':mean}))
  platt_records.append({'scope':'lodo','held_out':d,'group':group,'intercept':float(m.intercept_[0]),'slope':float(m.coef_[0,0]),'reference_image_prior':prior,'OOF':'10x5 grouped outer; 4-fold inner, original 5-repeat C tuning; repeat mean'})
  evaluate('lodo',d,f,group,raw,cal,prior,prior)
csv(OUT/'calibration_metrics.csv',pd.DataFrame(rows));csv(OUT/'platt_parameters.csv',pd.DataFrame(platt_records));csv(OUT/'em_parameters.csv',pd.DataFrame(em_records))
finish(OUT,__file__,{'status':'DONE','seconds':time.perf_counter()-START,'parameters':{'primary':'d (source-OOF logit Platt then label-shift EM)','platt':'image-level unweighted LR C1e6; no sign constraint; source-specific before external averaging','LODO_OOF':'10 repetitions x5 grouped outer, original source-only 5-repeat inner C tuning with4folds; no heldout-dataset data','EM':'tolerance1e-6; max1000; empirical image-level expectations; prior clipped1e-6 for arithmetic','ECE':'Original S12 equal-width ten bins; patient-equal weighting'},'metrics':rows})
