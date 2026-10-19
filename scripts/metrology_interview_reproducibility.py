from metrology_common import (
    COLUMNS,
    Image,
    OUTPUT_ROOT,
    Path,
    ROOT,
    SEED,
    THRESHOLD,
    csv,
    ensemble,
    external_frame,
    finish,
    json,
    logit,
    newjson,
    np,
    os,
    pd,
    resultdir,
    sha,
    time,
)
from reproducibility_statistics import paired_statistics,decision_agreement
from scipy.stats import pearsonr,spearmanr
OUT=resultdir('interview_reproducibility')
START=time.perf_counter()
META=Path(os.environ.get('SCALE_BUS_BUSBRA_METADATA',str(ROOT/'data/raw/BUS-BRA_zenodo_8231412/BUSBRA/BUSBRA/bus_data.csv')))
meta=pd.read_csv(META);f=external_frame('busbra');f['meta_id']=f.image.map(lambda x:Path(str(x)).stem)
assert f.meta_id.is_unique and meta.ID.is_unique
f=f.drop(columns=[c for c in ['ID','Case','Pathology','Histology','BIRADS','Device','Side'] if c in f])
f=f.merge(meta[['ID','Case','Pathology','Histology','BIRADS','Device','Side']],left_on='meta_id',right_on='ID',validate='one_to_one',how='left');assert f.Case.notna().all() and len(f)==1875
assert meta.Case.nunique()==1064
sizes=f.groupby('Case').size();assert set(sizes).issubset({1,2});assert int((sizes==2).sum())==811
checks=[];candidates=[];radii=[]
for case,g in f.groupby('Case',sort=True):
 if len(g)!=2:continue
 g=g.sort_values('ID');a,b=g.to_dict('records');pathology_ok=g.Pathology.nunique(dropna=False)==1;birads_ok=g.BIRADS.nunique(dropna=False)==1;scanner_ok=g.Device.nunique(dropna=False)==1;histology_ok=g.Histology.nunique(dropna=False)==1
 checks.append({'Case':int(case),'patient_id_first':a['patient_id'],'patient_id_second':b['patient_id'],'ID_first':a['ID'],'ID_second':b['ID'],'pathology_first':a['Pathology'],'pathology_second':b['Pathology'],'BIRADS_first':a['BIRADS'],'BIRADS_second':b['BIRADS'],'scanner_first':a['Device'],'scanner_second':b['Device'],'pathology_consistent':pathology_ok,'BIRADS_consistent':birads_ok,'scanner_consistent':scanner_ok,'histology_consistent':histology_ok,'included':pathology_ok and birads_ok and scanner_ok})
 pairr=[]
 for rec in (a,b):
  with Image.open(rec['mask_path']) as h:m=np.asarray(h.convert('L'))
  mx=float(m.max());threshold=0 if mx<=1 else max(8.,.1*mx);pairr.append(float(np.sqrt((m>threshold).sum()/np.pi)))
 radii.append({'Case':int(case),'radius_first_native_pixels':pairr[0],'radius_second_native_pixels':pairr[1],'ratio_second_to_first':pairr[1]/pairr[0],'included_by_metadata':pathology_ok and birads_ok and scanner_ok})
 if pathology_ok and birads_ok and scanner_ok:
  assert a['patient_id']==b['patient_id'];assert int(a['target'])==int(b['target']);candidates.append((a,b))
check=pd.DataFrame(checks);rad=pd.DataFrame(radii);csv(OUT/'pair_consistency_all_811.csv',check);csv(OUT/'excluded_pairs.csv',check[~check.included]);csv(OUT/'paired_native_mask_radii_all_811.csv',rad)
bad=(~check.pathology_consistent)|(~check.BIRADS_consistent)
precheck={'candidate_pairs':811,'pathology_mismatches':int((~check.pathology_consistent).sum()),'BIRADS_mismatches':int((~check.BIRADS_consistent).sum()),'scanner_mismatches':int((~check.scanner_consistent).sum()),'histology_mismatches':int((~check.histology_consistent).sum()),'pathology_or_BIRADS_mismatch_fraction':float(bad.mean()),'excluded_pairs':int((~check.included).sum()),'retained_pairs':len(candidates),'radius_pearson_all_candidates':float(pearsonr(rad.radius_first_native_pixels,rad.radius_second_native_pixels).statistic),'radius_spearman_all_candidates':float(spearmanr(rad.radius_first_native_pixels,rad.radius_second_native_pixels).statistic),'radius_ratio_quantiles':{str(q):float(rad.ratio_second_to_first.quantile(q)) for q in [.05,.25,.5,.75,.95]},'radius_ratio_rule':'Descriptive only, never used for selection','ordering':'lexicographic image ID; no anatomical meaning assigned to l/r','mask_radius':'sqrt(native mask area/pi), production threshold criterion before normalization'}
newjson(OUT/'consistency_checks.json',precheck);print('E2_CONSISTENCY',json.dumps(precheck),flush=True)
if bad.mean()>.05:
 finish(OUT,__file__,{'status':'STOPPED','reason':'Pathology or BI-RADS inconsistency exceeds5%','consistency':precheck});raise RuntimeError('Lesion-pair consistency threshold exceeded')
first=pd.DataFrame([a for a,b in candidates]);second=pd.DataFrame([b for a,b in candidates]);assert first.Case.is_unique and first.patient_id.is_unique
csv(OUT/'included_lesion_pairs.csv',pd.DataFrame([{'Case':int(a['Case']),'patient_id':a['patient_id'],'first_image':a['image'],'second_image':b['image'],'target':int(a['target'])} for a,b in candidates]))
cols=COLUMNS['advanced76']+COLUMNS['frozen98'];assert len(cols)==174 and len(set(cols))==174
values_a=[first[cols].to_numpy()];values_b=[second[cols].to_numpy()];scorepairs={};keys=cols.copy()
for group in ['advanced76','frozen98']:
 pa=ensemble(first[COLUMNS[group]].to_numpy(),group);pb=ensemble(second[COLUMNS[group]].to_numpy(),group);scorepairs[group]=(pa,pb)
 for output,aa,bb in [('probability',pa,pb),('logit',logit(np.clip(pa,1e-6,1-1e-6)),logit(np.clip(pb,1e-6,1-1e-6)))]:
  keys.append(group+'_'+output);values_a.append(aa[:,None]);values_b.append(bb[:,None])
a=np.concatenate(values_a,axis=1);b=np.concatenate(values_b,axis=1);stats=paired_statistics(a,b,bootstrap=2000,seed=SEED);rows=[]
for j,name in enumerate(keys):
 row={'variable':name,'kind':'feature' if j<174 else 'score','feature_group':'advanced76' if name in COLUMNS['advanced76'] or name.startswith('advanced76_') else 'frozen98'}
 for k,v in stats.items():row[k]=v if np.isscalar(v) else v[j]
 rows.append(row)
table=pd.DataFrame(rows);csv(OUT/'feature_icc_sw_rc_cv.csv',table[table.kind=='feature']);csv(OUT/'score_icc_and_bland_altman.csv',table[table.kind=='score'])
blocks={'GF':[c for c in COLUMNS['advanced76'] if c.startswith('advanced_mf_')],'W':[c for c in COLUMNS['advanced76'] if c.startswith('advanced_wavelet_')],'S':[c for c in COLUMNS['advanced76'] if c.startswith('advanced_shape_')],**{z:[c for c in COLUMNS['advanced76'] if '_'+name+'_' in c] for z,name in [('L','lesion'),('I','inner'),('O','outer')]},'all76':COLUMNS['advanced76'],'FRE98':COLUMNS['frozen98']}
summary=[]
for block,names in blocks.items():
 values=table[table.variable.isin(names)].ICC_1_1;summary.append({'block':block,'n_variables':len(values),'median_ICC':float(values.median()),'q25_ICC':float(values.quantile(.25)),'q75_ICC':float(values.quantile(.75)),'undefined_ICC':int(values.isna().sum()),'negative_ICC_count':int((values<0).sum())})
csv(OUT/'icc_block_zone_summary.csv',pd.DataFrame(summary))
agreements=[];scorepred=pd.DataFrame({'Case':first.Case.to_numpy(),'first_image':first.image.to_numpy(),'second_image':second.image.to_numpy(),'target':first.target.to_numpy()})
for group,(pa,pb) in scorepairs.items():
 agreements.append({'feature_group':group,**decision_agreement(pa,pb,THRESHOLD[group])});scorepred[group+'_first']=pa;scorepred[group+'_second']=pb
csv(OUT/'threshold_agreement.csv',pd.DataFrame(agreements));csv(OUT/'paired_locked_probabilities.csv',scorepred)
smd=pd.read_csv(OUTPUT_ROOT/'v07_measurement_invariance/smd_per_variable.csv');assoc=smd.assign(abs_SMD=smd.SMD.abs()).groupby(['scope','variable']).abs_SMD.median().reset_index();assoc=assoc.merge(table[table.kind=='feature'][['variable','ICC_1_1','ICC_ci_low','ICC_ci_high']],on='variable',validate='many_to_one');csv(OUT/'feature_icc_vs_domain_SMD.csv',assoc)
finish(OUT,__file__,{'status':'DONE','seconds':time.perf_counter()-START,'metadata_sha256':sha(META),'statistics_helper':Path(__file__).with_name('reproducibility_statistics.py').name,'statistics_helper_sha256':sha(Path(__file__).with_name('reproducibility_statistics.py')),'parameters':{'seed':SEED,'bootstrap':2000,'ICC':'one-way random single measurement ICC(1,1); percentile patient bootstrap; negative ICC retained','Sw':'sqrt(mean((first-second)^2)/2)','RC':'2.77*Sw','CV':'100*sqrt(mean(((first-second)/(sqrt2*pairmean))^2)); only variables positive in every view','Bland_Altman':'mean(second-first) +/-1.96 sample SD; lexicographic ordering does not imply anatomical view','threshold_source':'results/v04_locked_external/locked_external_summary.json','description':'two views of the same biopsied lesion; inter-view reproducibility, not same-view test-retest','UCLM':'excluded: patient ID is not lesion ID'},'consistency':precheck,'block_summary':summary,'score_summary':table[table.kind=='score'].to_dict('records'),'decision_agreement':agreements})
