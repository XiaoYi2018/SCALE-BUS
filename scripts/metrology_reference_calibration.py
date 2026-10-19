from metrology_common import (
    Image,
    SEED,
    csv,
    finish,
    frames,
    json,
    newjson,
    np,
    pd,
    resultdir,
    rng_for,
    time,
)
from fractal_extrema.advanced_features import _zone_masks,_generalized_dimensions,_multifractal_zone_features,_wavelet_zone_features,_boundary_features,_block_masses
from fractal_extrema.features import _prepare_roi
from skimage.measure import regionprops,label
from scipy.special import ellipe
from matplotlib.path import Path as PolygonPath
import argparse
OUT=resultdir('reference_calibration');N=200;RADII=(16,24,32,48,64,96);WEIGHTS=((.25,.25,.25,.25),(.4,.3,.2,.1),(.55,.25,.15,.05))
YY,XX=np.mgrid[:256,:256]

def support_set(rng):
 out=[('full',0,np.ones((256,256),bool))]
 for r in RADII:
  cx,cy=rng.uniform(r+13,255-r-13,2);m=(XX-cx)**2+(YY-cy)**2<=r*r
  for name,z in _zone_masks(m).items():out.append((name,r,z))
 return out

def cascade(weights,rng):
 a=np.ones((1,1))
 for _ in range(8):
  h=a.shape[0];perms=np.argsort(rng.random((h,h,4)),axis=2);p=np.asarray(weights)[perms];b=np.empty((h*2,h*2))
  b[0::2,0::2]=a*p[:,:,0];b[0::2,1::2]=a*p[:,:,1];b[1::2,0::2]=a*p[:,:,2];b[1::2,1::2]=a*p[:,:,3];a=b
 assert abs(a.sum()-1)<1e-12 and a.min()>0
 return a

def variants(a,rng):
 im=(a-a.min())/(a.max()-a.min()) if a.max()>a.min() else np.ones_like(a)
 yield 'baseline',0.,im
 for sig in (.05,.10):yield 'speckle',sig,np.clip(im+im*rng.normal(0,sig,im.shape),0,1)
 for g in (.8,1.2):yield 'gamma',g,im**g
 # Existing perturbation path uses PIL bilinear; float mode avoids quantization of reference measure.
 p=Image.fromarray(im.astype(np.float32));yield 'downsample',.5,np.asarray(p.resize((128,128),Image.Resampling.BILINEAR).resize((256,256),Image.Resampling.BILINEAR),float)

def gf():
 start=time.perf_counter();rows=[];sens=[];unmeas=[]
 for wi,weights in enumerate(WEIGHTS):
  theory={'d0':2.,'d1':float(-np.sum(np.asarray(weights)*np.log2(weights))),'d2':float(-np.log2(np.sum(np.square(weights))))}
  for rep in range(N):
   rng=rng_for('E1a',wi,rep);a=cascade(weights,rng);supports=support_set(rng);vv=list(variants(a,rng))
   for zone,r,z in supports:
    sizes,_=_block_masses(np.where(z,a,0),z)
    if z.sum()<32 or len(sizes)<3:unmeas.append({'weight_set':wi,'rep':rep,'zone':zone,'mask_radius':r,'pixels':int(z.sum()),'valid_sizes':len(sizes)});continue
    raw=_generalized_dimensions(np.where(z,a,0),z);production=_multifractal_zone_features(vv[0][2],z,zone)
    for metric in ['d0','d1','d2','spread','r2_mean']:
     ref=theory.get(metric,np.nan)
     if metric=='d0' and zone in ('inner','outer'):ref=np.nan
     for path,value in [('estimator',raw[metric]),('production_intensity',production[f'advanced_mf_{zone}_intensity_{metric}'])]:rows.append({'weight_set':wi,'weights':str(weights),'rep':rep,'zone':zone,'mask_radius':r,'path':path,'variable':metric,'value':value,'reference':ref,'error':value-ref,'additional_deviation_from_estimator':value-raw[metric] if path=='production_intensity' else 0.})
    for condition,param,img in vv[1:]:
     f=_multifractal_zone_features(img,z,zone);rd=_generalized_dimensions(np.where(z,img,0),z);base_rd=_generalized_dimensions(np.where(z,vv[0][2],0),z)
     for signal in ['intensity','gradient']:
      for metric in ['d0','d1','d2','spread','r2_mean']:
       key=f'advanced_mf_{zone}_{signal}_{metric}';delta=f[key]-production[key];step=param-1 if condition in ('gamma','downsample') else param
       sens.append({'weight_set':wi,'rep':rep,'zone':zone,'mask_radius':r,'path':'production_'+signal,'condition':condition,'parameter':param,'variable':metric,'delta':delta,'sensitivity_coefficient':delta/step})
     for metric in ['d0','d1','d2','spread','r2_mean']:
      # Reference perturbation is intensity-mapped, paired with mapped baseline rather than raw cascade.
      sens.append({'weight_set':wi,'rep':rep,'zone':zone,'mask_radius':r,'path':'estimator_mapped_intensity','condition':condition,'parameter':param,'variable':metric,'delta':rd[metric]-base_rd[metric],'sensitivity_coefficient':(rd[metric]-base_rd[metric])/step})
   if (rep+1)%25==0:print('E1a',wi,rep+1,'seconds',round(time.perf_counter()-start,1),flush=True)
 f=pd.DataFrame(rows);ss=pd.DataFrame(sens);csv(OUT/'multifractal_realizations.csv',f);csv(OUT/'multifractal_sensitivity_realizations.csv',ss);csv(OUT/'unmeasurable_supports.csv',pd.DataFrame(unmeas,columns=['weight_set','rep','zone','mask_radius','pixels','valid_sizes']))
 summary=f.groupby(['weight_set','weights','zone','mask_radius','path','variable'],dropna=False).agg(n=('value','size'),mean=('value','mean'),SD=('value','std'),reference=('reference','first'),bias=('error','mean'),RMSE=('error',lambda x:np.sqrt(np.mean(x*x))),additional_deviation_from_estimator=('additional_deviation_from_estimator','mean')).reset_index();csv(OUT/'multifractal_summary.csv',summary)
 csv(OUT/'multifractal_sensitivity_summary.csv',ss.groupby(['weight_set','zone','mask_radius','path','condition','parameter','variable']).agg(n=('delta','size'),mean_delta=('delta','mean'),SD_delta=('delta','std'),mean_sensitivity=('sensitivity_coefficient','mean')).reset_index())
 newjson(OUT/'gf_complete.json',{'seconds':time.perf_counter()-start,'unmeasurable_count':len(unmeas),'rows':len(f),'sensitivity_rows':len(ss)})

def wavelet():
 start=time.perf_counter();rows=[]
 # Phase varies over 200 independent paired realizations; noise uses N(0,1), affine mapped to [0,1].
 for rep in range(N):
  rng=rng_for('E1b',rep);zones=support_set(rng);phase=rng.uniform(0,2*np.pi);signals=[]
  for angle in (0,45,90):
   t=np.deg2rad(angle)
   for period in (4,8,16):signals.append(('grating',angle,period,.5+.5*np.cos(2*np.pi*(XX*np.cos(t)+YY*np.sin(t))/period+phase)))
  noise=rng.normal(size=(256,256));noise=(noise-noise.min())/(noise.max()-noise.min());signals.append(('white_noise',-1,0,noise))
  for kind,angle,period,img in signals:
   full_reference=_wavelet_zone_features(img,np.ones_like(img,dtype=bool),'full')
   for zone,r,z in zones:
    if zone=='full':continue
    fs=_wavelet_zone_features(img,z,zone)
    for key,val in fs.items():
     suffix=key.removeprefix('advanced_wavelet_'+zone+'_');level,metric=suffix.split('_',1);expected_sign=-1 if angle==0 else 1 if angle==90 else np.nan
     rows.append({'rep':rep,'kind':kind,'angle':angle,'period':period,'zone':zone,'mask_radius':r,'level':level,'variable':metric,'value':val,'expected_hv_sign':expected_sign if kind=='grating' and metric=='hv_logratio' else np.nan,'full_support_reference':full_reference['advanced_wavelet_full_'+suffix],'deviation_from_full_support':val-full_reference['advanced_wavelet_full_'+suffix],'HV_deviation_from_zero':val if metric=='hv_logratio' and (angle==45 or kind=='white_noise') else np.nan,'diagonal_deviation_from_zero':val if metric=='diagonal_fraction' and angle in (0,90) else np.nan})
  if (rep+1)%25==0:print('E1b',rep+1,'seconds',round(time.perf_counter()-start,1),flush=True)
 f=pd.DataFrame(rows);csv(OUT/'wavelet_realizations.csv',f);summary=f.groupby(['kind','angle','period','zone','mask_radius','level','variable']).agg(n=('value','size'),mean=('value','mean'),SD=('value','std'),expected_hv_sign=('expected_hv_sign','first'),mean_full_support_reference=('full_support_reference','mean'),mean_deviation_from_full_support=('deviation_from_full_support','mean'),SD_deviation_from_full_support=('deviation_from_full_support','std'),mean_HV_deviation_from_zero=('HV_deviation_from_zero','mean'),mean_diagonal_deviation_from_zero=('diagonal_deviation_from_zero','mean')).reset_index();summary['sign_agrees']=np.where(summary.expected_hv_sign.notna(),np.sign(summary['mean'])==summary.expected_hv_sign,np.nan);csv(OUT/'wavelet_summary.csv',summary)
 newjson(OUT/'wavelet_complete.json',{'seconds':time.perf_counter()-start,'rows':len(f),'noise_expectation':'ensemble HV log-ratio near zero, no analytic finite-mask entropy or diagonal fraction specified','angle_convention':'cos(theta)*x + sin(theta)*y; 0deg vertical structures => HV negative, 90deg horizontal => positive; db2 symmetric level3 unchanged'})

def shape():
 start=time.perf_counter();rows=[]
 for shape,q,n in [('circle',1,0),('ellipse_1.5',1.5,0),('ellipse_2',2,0),('ellipse_3',3,0),('pentagon',1,5),('octagon',1,8),('star5',1,10)]:
  for r in (8,12,16,24,32,48,64,96):
   if n==0:
    a=r*np.sqrt(q);b=r/np.sqrt(q);P=4*a*ellipe(1-b*b/(a*a));A=np.pi*r*r;hull=A;outer=a;vertices=None
   else:
    angles=np.arange(n)*2*np.pi/n+np.pi/2;radii=np.ones(n)
    if shape=='star5':radii[1::2]=.5
    vertices=np.column_stack([radii*np.cos(angles),radii*np.sin(angles)]);v=vertices;area0=abs(np.sum(v[:,0]*np.roll(v[:,1],-1)-v[:,1]*np.roll(v[:,0],-1)))/2;scale=np.sqrt(np.pi*r*r/area0);vertices*=scale;outer=scale
    A=np.pi*r*r;P=np.linalg.norm(vertices-np.roll(vertices,-1,axis=0),axis=1).sum();hull=(5/2*np.sin(2*np.pi/5)*scale*scale) if shape=='star5' else A
   refs={'compactness':4*np.pi*A/P**2,'solidity':A/hull,'eccentricity':np.sqrt(1-1/q**2),'axis_ratio':q,'convex_area_ratio':hull/A,'perimeter_normalized':P/np.sqrt(A)}
   side=int(2*np.ceil(outer+16));yy,xx=np.mgrid[:side,:side];xx=xx-(side-1)/2;yy=yy-(side-1)/2
   for angle in np.linspace(0,90,10):
    t=np.deg2rad(angle);xr=xx*np.cos(t)+yy*np.sin(t);yr=-xx*np.sin(t)+yy*np.cos(t)
    if n==0:m=(xr/a)**2+(yr/b)**2<=1
    else:m=PolygonPath(vertices).contains_points(np.column_stack([xr.ravel(),yr.ravel()])).reshape(side,side)
    fs=_boundary_features(m)
    for key,val in fs.items():
     metric=key.removeprefix('advanced_shape_');ref=refs.get(metric,np.nan);rows.append({'shape':shape,'mask_radius':r,'rotation_deg':angle,'canvas_side':side,'variable':metric,'value':val,'reference':ref,'error':val-ref})
 f=pd.DataFrame(rows);csv(OUT/'shape_realizations.csv',f);csv(OUT/'shape_summary.csv',f.groupby(['shape','mask_radius','variable']).agg(n=('value','size'),mean=('value','mean'),SD=('value','std'),reference=('reference','first'),bias=('error','mean'),RMSE=('error',lambda x:np.sqrt(np.mean(x*x)))).reset_index())
 real=[]
 for dataset,frame in frames().items():
  for rec in frame.to_dict('records'):
   with Image.open(rec['image_path']) as h:im=np.asarray(h.convert('L'))
   with Image.open(rec['mask_path']) as h:mask=np.asarray(h.convert('L'))
   _,m=_prepare_roi(im,mask);rp=max(regionprops(label(m,connectivity=2)),key=lambda p:p.area);real.append({'dataset':dataset,'image':rec['image'],'mask_equivalent_radius':np.sqrt(m.sum()/np.pi),'axis_ratio':rp.major_axis_length/max(rp.minor_axis_length,np.finfo(float).eps)})
  print('E1c_REAL',dataset,len(frame),flush=True)
 real=pd.DataFrame(real);csv(OUT/'real_lesion_geometry.csv',real);rs=real.groupby('dataset')[['mask_equivalent_radius','axis_ratio']].quantile([.05,.5,.95]).unstack();rs.columns=[c+'_'+str(p) for c,p in rs.columns];csv(OUT/'real_lesion_geometry_summary.csv',rs.reset_index())
 bounds=real.mask_equivalent_radius.quantile([.05,.95]).to_numpy();csv(OUT/'calibration_real_range_map.csv',pd.DataFrame([{'experiment':e,'mask_radius':r,'within_pooled_real_5_95_percentile':bool(bounds[0]<=r<=bounds[1]),'pooled_real_q05':bounds[0],'pooled_real_q95':bounds[1]} for e,rr in [('E1a_E1b',RADII),('E1c',(8,12,16,24,32,48,64,96))] for r in rr]))
 newjson(OUT/'shape_complete.json',{'seconds':time.perf_counter()-start,'synthetic_rows':len(f),'real_images':len(real)})

def summary():
 finish(OUT,__file__,{'status':'DONE','parameters':{'seed':SEED,'E1a_realizations_per_condition':N,'weights':WEIGHTS,'radii':RADII,'cascade':'8-level four-quadrant positive weights, independent random permutation at each node; generator-referenced finite-support bias','raw_measure':'original unshifted unit-mass cascade restricted to support (zero outside)','D0_reference':'full and lesion support2; inner/outer no analytic reference','production':'cascade affine min-max to [0,1], uniform cascade mapped to constant1; unchanged production min subtraction +1/255 and Sobel +1/255','sensitivity':'paired mapped images, clip after speckle; floating PIL bilinear .5 resampling avoids quantizing reference; coefficient delta/(parameter-baseline), baseline speckle0,gamma1,resolution1','R_squared':'production r2_mean is mean across D0/D1/D2 fits; no separate per-D R2 recorded','E1b':'200 phases/noise realizations; db2 symmetric3; quantitative tables, qualitative expectations only','E1c':'native sufficiently padded synthetic canvas; no prepare_roi; star inner/outer radius.5;10angles0..90deg','real_geometry':'unchanged production prepare_roi 256grid; radius from all lesion pixels; major/minor from largest connected component'},'stages':{x:json.loads((OUT/(x+'_complete.json')).read_text()) for x in ['gf','wavelet','shape']}})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['gf','wavelet','shape','summary']);a=ap.parse_args();globals()[a.phase]()
if __name__=='__main__':main()
