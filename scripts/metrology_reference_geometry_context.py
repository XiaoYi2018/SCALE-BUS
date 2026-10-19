from metrology_common import (
    Path,
    csv,
    newjson,
    pd,
    resultdir,
    sha,
)
OUT=resultdir('reference_calibration')
real=pd.read_csv(OUT/'real_lesion_geometry.csv');rows=[]
for dataset,f in real.groupby('dataset'):
 rb=f.mask_equivalent_radius.quantile([.05,.5,.95]);ab=f.axis_ratio.quantile([.05,.5,.95])
 for shape,ratio in [('circle',1.),('ellipse_1.5',1.5),('ellipse_2',2.),('ellipse_3',3.),('pentagon',1.),('octagon',1.),('star5',1.)]:
  for radius in (8,12,16,24,32,48,64,96):rows.append({'dataset':dataset,'experiment':'E1c','shape':shape,'nominal_mask_equivalent_radius':radius,'analytic_axis_ratio':ratio,'within_dataset_radius_5_95':bool(rb.loc[.05]<=radius<=rb.loc[.95]),'within_dataset_axis_ratio_5_95':bool(ab.loc[.05]<=ratio<=ab.loc[.95]),'real_radius_q05':rb.loc[.05],'real_radius_median':rb.loc[.5],'real_radius_q95':rb.loc[.95],'real_axis_ratio_q05':ab.loc[.05],'real_axis_ratio_median':ab.loc[.5],'real_axis_ratio_q95':ab.loc[.95]})
 for radius in (16,24,32,48,64,96):rows.append({'dataset':dataset,'experiment':'E1a_E1b','shape':'circle','nominal_mask_equivalent_radius':radius,'analytic_axis_ratio':1.,'within_dataset_radius_5_95':bool(rb.loc[.05]<=radius<=rb.loc[.95]),'within_dataset_axis_ratio_5_95':bool(ab.loc[.05]<=1.<=ab.loc[.95]),'real_radius_q05':rb.loc[.05],'real_radius_median':rb.loc[.5],'real_radius_q95':rb.loc[.95],'real_axis_ratio_q05':ab.loc[.05],'real_axis_ratio_median':ab.loc[.5],'real_axis_ratio_q95':ab.loc[.95]})
csv(OUT/'calibration_per_dataset_context.csv',pd.DataFrame(rows))
newjson(OUT/'geometry_context_provenance.json',{'script':Path(__file__).name,'script_sha256':sha(__file__),'input':'real_lesion_geometry.csv','input_sha256':sha(OUT/'real_lesion_geometry.csv'),'note':'Nominal radius and analytic axis ratio contextualized against each dataset separately; digital-mask dimensions need not equal nominal values.'})
