#!/usr/bin/env python

import os
import glob
import numpy as np
from astropy.table import Table
import fitsio
from bokpipe.bokdm import SimpleFileNameMap

cfhtImgDir = os.environ.get('CFHTRMDATA',
                         os.path.join(os.environ['HOME'],'data','public',
                                      'CFHT','archive','SDSSRM'))

cfhtCatDir = os.environ.get('CFHTRMOUTDIR',
                         os.path.join(os.environ['BOK90PRIMEOUTDIR'],'CFHT'))

def set_obs_db_item(t,i,imFile):
	f = fitsio.FITS(imFile)
	h = f[0].read_header()
	t['objName'][i] = h['OBJECT']
	t['utDate'][i] = h['DATE-OBS'].replace('-','')
	#t['date_obs'] = h['DATE-OBS']
	t['utObs'][i] = h['UTC-OBS']
	t['lstObs'][i] = h['LST-OBS']
	t['mjdStart'][i] = h['MJD-OBS']
	t['airmass'][i] = h['AIRMASS']
	t['humidity'][i] = h['RELHUMID']
	t['ccdbin1'][i] = h['CCDBIN1']
	t['ccdbin2'][i] = h['CCDBIN2']
	try:
		t['elixirZp'][i] = h['PHOT_C']
	except:
		pass
	for j,hdu in enumerate(f[1:]):
		h = hdu.read_header()
		t['ccdGain'][i,j] = h['GAIN']
		t['ampGain'][i,j] = [ h['GAINA'], h['GAINB'] ]

def make_obs_db():
	arxDat = Table.read(os.path.join(cfhtImgDir,
	                                 'result_g5s2i0ftxwta9jbn.csv'))
	t = Table()
	t['expNum'] = arxDat['Sequence Number']
	t['fileName'] = np.char.strip(arxDat['Product ID'])
	t['utDate'] = 'YYYYMMDD'
	t['imType'] = 'object'
	t['filter'] = 'g'
	t['longFilt'] = arxDat['Filter']
	t['objName'] = ' '*35
	t['expTime'] = arxDat['Int. Time']
	t['ccdbin1'] = np.int32(0)
	t['ccdbin2'] = np.int32(0)
	t['airmass'] = np.float32(0.0)
	t['utObs'] = 'HH:MM:SS.SSS'
	t['lstObs'] = 'HH:MM:SS.SS'
	t['targetRa'] = arxDat['RA (J2000.0)']
	t['targetDec'] = arxDat['Dec. (J2000.0)']
	t['mjdStart'] = arxDat['Start Date'].astype(np.float64)
	t['humidity'] = np.float32(0.0)
	t['ccdGain'] = np.zeros((1,36),dtype=np.float32)
	t['ampGain'] = np.zeros((1,36,2),dtype=np.float32)
	t['elixirZp'] = np.float32(0.0)
	nfiles = 0
	for i,fn in enumerate(t['fileName']):
		try:
			set_obs_db_item(t,i,os.path.join(cfhtImgDir,fn+'.fits.fz'))
			nfiles += 1
		except IOError:
			print fn,' not found'
			continue
	t.sort('expNum')
	t.write(os.path.join('.','config','sdssrm-cfht.fits.gz'),
	        overwrite=True)
	print 'found %d/%d files' % (nfiles,len(t))

class CfhtObsDb(object):
	def __init__(self):
		cfhtDir = '.' # XXX
		self.obsDb = Table.read(os.path.join(cfhtDir,'config',
		                                     'sdssrm-cfht.fits.gz'))
		if not os.path.exists(cfhtCatDir):
			os.makedirs(cfhtCatDir)
		self.fmap = { ftype:SimpleFileNameMap(cfhtImgDir,cfhtCatDir,'.'+ftype)
		                for ftype in ['cat','wcscat','psf'] }
		self.fmap['img'] = SimpleFileNameMap(cfhtImgDir,cfhtCatDir,
		                                     fromRaw=True)
	def getFiles(self,filt=None):
		return self.obsDb['fileName']
	def __call__(self,ftype):
		return self.fmap[ftype]

