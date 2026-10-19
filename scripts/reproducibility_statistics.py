"""One-way random single-measure ICC and paired measurement statistics.

Pure numerical helper; does not read or write any files or choose lesion pairs.
"""
import numpy as np

def paired_statistics(first,second,bootstrap=2000,seed=20261018):
 a=np.asarray(first,dtype=float);b=np.asarray(second,dtype=float)
 if a.ndim==1:a=a[:,None]
 if b.ndim==1:b=b[:,None]
 assert a.shape==b.shape and a.ndim==2 and len(a)>1
 assert np.isfinite(a).all() and np.isfinite(b).all()
 n,k=a.shape;mu=(a+b)/2;mu=mu-mu.mean(axis=0);q=(a-b)**2/2
 msb=2*np.var(mu,axis=0,ddof=1);msw=q.mean(axis=0);den=msb+msw
 icc=np.divide(msb-msw,den,out=np.full(k,np.nan),where=den>0)
 rng=np.random.default_rng(seed);counts=rng.multinomial(n,np.full(n,1/n),size=bootstrap).astype(float)
 boot_mu=counts@mu/n;boot_var=np.maximum((counts@(mu*mu)-n*boot_mu**2)/(n-1),0);boot_b=2*boot_var;boot_w=counts@q/n;boot_den=boot_b+boot_w
 boot_icc=np.divide(boot_b-boot_w,boot_den,out=np.full_like(boot_den,np.nan),where=boot_den>0)
 lo=np.full(k,np.nan);hi=lo.copy();valid=np.isfinite(boot_icc).any(axis=0)
 if valid.any():lo[valid],hi[valid]=np.nanquantile(boot_icc[:,valid],[.025,.975],axis=0)
 sw=np.sqrt(msw);positive=np.all(a>0,axis=0)&np.all(b>0,axis=0);cv=np.full(k,np.nan)
 cv[positive]=100*np.sqrt(np.mean(((a[:,positive]-b[:,positive])/(np.sqrt(2)*(a[:,positive]+b[:,positive])/2))**2,axis=0))
 diff=b-a;bias=diff.mean(axis=0);sd_diff=diff.std(axis=0,ddof=1)
 return {'n_pairs':n,'ICC_1_1':icc,'ICC_ci_low':lo,'ICC_ci_high':hi,'Sw':sw,'RC_2_77_Sw':2.77*sw,'within_subject_CV_percent':cv,'all_values_positive':positive,'difference_mean_second_minus_first':bias,'difference_SD':sd_diff,'LoA_low':bias-1.96*sd_diff,'LoA_high':bias+1.96*sd_diff,'valid_bootstrap_count':np.isfinite(boot_icc).sum(axis=0),'zero_total_variance':den==0}

def decision_agreement(first,second,threshold):
 a=np.asarray(first)>=threshold;b=np.asarray(second)>=threshold;n=len(a)
 agreement=float(np.mean(a==b));pa=float(a.mean());pb=float(b.mean());expected=pa*pb+(1-pa)*(1-pb)
 return {'n_pairs':n,'threshold':threshold,'agreement':agreement,'Cohen_kappa':(agreement-expected)/(1-expected) if expected<1 else np.nan,'first_positive':int(a.sum()),'second_positive':int(b.sum()),'both_positive':int(np.sum(a&b)),'both_negative':int(np.sum(~a&~b))}
