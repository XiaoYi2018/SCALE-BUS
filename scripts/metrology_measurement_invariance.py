from metrology_common import (
    COLUMNS,
    SEED,
    csv,
    finish,
    frames,
    np,
    pd,
    resultdir,
    roc_auc_score,
    time,
)
import itertools
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
OUT=resultdir('measurement_invariance');START=time.perf_counter()
D=frames();allcols=COLUMNS['advanced76'];blocks={'GF':[c for c in allcols if 'advanced_mf_' in c],'W':[c for c in allcols if 'advanced_wavelet_' in c],'S':[c for c in allcols if 'advanced_shape_' in c],**{k:[c for c in allcols if '_'+zone+'_' in c] for k,zone in [('L','lesion'),('I','inner'),('O','outer')]},'all76':allcols,'FRE98':COLUMNS['frozen98']}
cols=COLUMNS['frozen98']+allcols
for d,f in D.items():f['dataset']=d;f['domain_group']=d+':'+f.patient_id.astype(str)
source=pd.concat([D['busuclm'],D['busi']],ignore_index=True)
rows=[]
for label in [0,1]:
 sd=source.loc[source.target==label,cols].std(ddof=1).replace(0,np.nan)
 for a,b in itertools.combinations(D,2):
  ma=D[a].loc[D[a].target==label,cols].mean();mb=D[b].loc[D[b].target==label,cols].mean()
  for c in cols:rows.append({'scope':'dataset','class':label,'a':a,'b':b,'variable':c,'SMD':float((ma[c]-mb[c])/sd[c]),'source_sd':sd[c]})
 bra=D['busbra'];counts=bra.groupby('device').size();devices=sorted(counts[counts>=100].index)
 for a,b in itertools.combinations(devices,2):
  fa=bra[(bra.target==label)&(bra.device==a)];fb=bra[(bra.target==label)&(bra.device==b)]
  for c in cols:rows.append({'scope':'scanner','class':label,'a':a,'b':b,'variable':c,'SMD':float((fa[c].mean()-fb[c].mean())/sd[c]),'source_sd':sd[c]})
smd=pd.DataFrame(rows);csv(OUT/'smd_per_variable.csv',smd)
summary=[]
for (scope,label),g in smd.groupby(['scope','class']):
 for block,cs in blocks.items():
  v=g.loc[g.variable.isin(cs),'SMD'].abs();summary.append({'scope':scope,'class':label,'block':block,'median_abs_SMD':v.median(),'q25_abs_SMD':v.quantile(.25),'q75_abs_SMD':v.quantile(.75),'comparisons':len(v)})
csv(OUT/'smd_block_summary.csv',pd.DataFrame(summary))
pool=pd.concat(list(D.values()),ignore_index=True);codes={d:i for i,d in enumerate(D)};domain=pool.dataset.map(codes).to_numpy();groups=pool.domain_group.to_numpy();aucrows=[]
for disease in ['all',0,1]:
 mask=np.ones(len(pool),bool) if disease=='all' else pool.target.to_numpy()==disease
 f=pool.loc[mask].reset_index(drop=True);y=domain[mask];gr=groups[mask];cv=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(f,y,gr))
 for block,cs in blocks.items():
  pp=np.zeros((len(f),4))
  for tr,va in cv:
   model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=3000,random_state=SEED));model.fit(f.iloc[tr][cs],y[tr]);pp[va]=model.predict_proba(f.iloc[va][cs])
  weights=1/f.groupby('domain_group').domain_group.transform('size').to_numpy()
  auc=roc_auc_score(y,pp,multi_class='ovr',average='macro',sample_weight=weights)
  aucrows.append({'class':disease,'block':block,'domain_auc_macro_ovr_patient_balanced':auc,'domain_auc_macro_ovr_image':roc_auc_score(y,pp,multi_class='ovr',average='macro'),'n_images':len(f),'groups':len(np.unique(gr))})
  print('DOMAIN',disease,block,auc,flush=True)
csv(OUT/'domain_separability.csv',pd.DataFrame(aucrows))
known={'GF':.7363,'W':.8576,'S':.9070,'L':.7099,'I':.8765,'O':.8497,'all76':.9164,'FRE98':.8851}
assoc=pd.DataFrame([{'block':b,'lodo_mean_auc':v,'median_abs_SMD':smd.loc[smd.variable.isin(blocks[b]) & (smd.scope=='dataset'),'SMD'].abs().median(),'domain_auc':next(x['domain_auc_macro_ovr_patient_balanced'] for x in aucrows if x['class']=='all' and x['block']==b)} for b,v in known.items()]);csv(OUT/'invariance_discrimination_association.csv',assoc)
finish(OUT,__file__,{'status':'DONE','seconds':time.perf_counter()-START,'parameters':{'SMD':'class-specific image means; denominator=SD of concatenated BUS-UCLM/BUSI images within that class, ddof1','domain_classifier':'5-fold StratifiedGroupKFold, dataset-prefixed patient/visual groups; scaler inside folds; balanced LR C1; OVR macro AUC','note':'Domain classifier is descriptive only; locked disease models unchanged'},'association':assoc.to_dict('records'),'scanner_devices':devices})
