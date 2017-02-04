#!/usr/bin/env python

import numpy as np
from astropy.table import Table,join

unitab_dtype = [
  ('objId','i4'),
  ('telescope','S8'),
  ('instrument','S11'),
  ('mjd','f8'),
  ('expTime','f4'),
  ('airmass','f4'),
  ('filter','S1'),
  ('aperMag','f4'), ('aperMagErr','f4'),
  ('psfMag','f4'), ('psfMagErr','f4'),
  ('modelMag','f4'), ('modelMagErr','f4'),
  ('sdss_aperMag','f4'), ('sdss_psfMag','f4'), ('sdss_modelMag','f4'), 
]

def merge_qsos():
	'''Merge the RM target file (target_fibermap.fits) with the master
	   quasar catalog for the field (targets_all.fits), which is the superset
	   of quasars the RM targets were selected from. The main purpose is to
	   assign a unique identifier to each quasar ('objId'), where the RM
	   targets are numbered 000-849 (as in target_fibermap.fits), and the
	   non-RM targets are numbered starting at 10000.
	'''
	rmtarg = Table.read('target_fibermap.fits')
	rmqso = rmtarg[:849]
	allqso = Table.read('targets_final.fits')
	rmqso['objId'] = np.arange(len(rmqso),dtype=np.int32)
	# for non-RM targets construct a unique identifier starting at i=10000
	allqso['objId'] = 10000 + np.arange(len(allqso),dtype=np.int32)
	allqso['objId'][rmqso['INDX']] = rmqso['objId']
	# delete duplicate/unnecessary columns
	allqso = allqso['objId','RA_DR10','DEC_DR10','Z','Z_ERR',
	                'PROGRAMNAME','RELEASE']
	allqso['RA_DR10'].name = 'ra'
	allqso['DEC_DR10'].name = 'dec'
	rmqso = rmqso['objId','ZFINAL']
	mtab = join(allqso,rmqso,'objId',join_type='outer')
	mtab.write('allqsos_rmfield.fits',overwrite=True)

def get_SDSS():
	phot = Table.read('allqsos_rmfield_phot.fits')
	tab = Table(dtype=unitab_dtype)
	for obs in phot:
		for j,b in enumerate('ugriz'):
			mjd = obs['TAI_'+b] / (24*3600)
			row = (obs['rmObjId'],'SDSS2.5m','SDSS-Imager',mjd,
			       54.0,obs['airmass_'+b],b,
			       obs['fiberMag_'+b],obs['psfMag_'+b],obs['modelMag_'+b],
			       obs['fiberMagErr_'+b],obs['psfMagErr_'+b],
			         obs['modelMagErr_'+b],
			       obs['fiberMag_'+b],obs['psfMag_'+b],obs['modelMag_'+b])
			tab.add_row(row)
	return tab

