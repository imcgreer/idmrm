#!/usr/bin/env python

import os,sys
import glob
import numpy as np
from astropy.table import Table
from astropy.io import fits
from bokpipe.bokdm import SimpleFileNameMap

cfhtImgDir = os.environ.get('CFHTRMDATA',
                         os.path.join(os.environ['HOME'],'data','public',
                                      'CFHT','archive','SDSSRM'))

cfhtCatDir = os.environ.get('CFHTRMOUTDIR',
                         os.path.join(os.environ['BOK90PRIMEOUTDIR'],'CFHT'))

def set_obs_db_item(t,i,imFile):
	f = fits.open(imFile)
	h = f[0].header
	t['objName'][i] = h['OBJECT']
	t['utDate'][i] = h['DATE-OBS'].replace('-','')
	t['utDir'][i] = t['utDate'][i][:6]
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
		h = hdu.header
		t['ccdGain'][i,j] = h['GAIN']
		t['ampGain'][i,j] = [ h['GAINA'], h['GAINB'] ]

def make_obs_db():
	arxDat = Table.read(os.path.join(cfhtImgDir,
	                                 'result_g5s2i0ftxwta9jbn.csv'))
	t = Table()
	t['frameIndex'] = arxDat['Sequence Number']
	t['expNum'] = arxDat['Sequence Number']
	t['fileName'] = np.char.strip(arxDat['Product ID'])
	t['utDate'] = 'YYYYMMDD'
	t['utDir'] = 'YYYYMM'
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
	t['ccdGain'] = np.zeros((1,40),dtype=np.float32)
	t['ampGain'] = np.zeros((1,40,2),dtype=np.float32)
	t['elixirZp'] = np.float32(0.0)
	t['good'] = True
	nfiles = 0
	errlog = open('cfhtdataerrs.log','w')
	for i,fn in enumerate(t['fileName']):
		sys.stdout.write("\r%d/%d" % (i+1,len(t)))
		sys.stdout.flush()
		try:
			set_obs_db_item(t,i,os.path.join(cfhtImgDir,fn+'.fits.fz'))
			nfiles += 1
		except IOError:
			errlog.write('%s not found\n' % fn)
			continue
	errlog.close()
	t.sort('expNum')
	t.write(os.path.join('.','config','sdssrm-cfht.fits.gz'),
	        overwrite=True)
	print 'found %d/%d files' % (nfiles,len(t))

def update_obs_db():
	t = Table.read(os.path.join('.','config','sdssrm-cfht.fits.gz'))
	ii = np.where((t['utDate']=='YYYYMMDD')&(t['expNum']<1e6))[0]
	for i,fn in zip(ii,t['fileName'][ii]):
		set_obs_db_item(t,i,os.path.join(cfhtImgDir,fn+'.fits.fz'))
	t.write(os.path.join('.','config','sdssrm-cfht.fits.gz'),
	        overwrite=True)

def move_files_to_subdirs(obsDb):
	for row in obsDb:
		rootf = os.path.join(cfhtImgDir,row['fileName']+'.fits.fz')
		subdir = os.path.join(cfhtImgDir,row['utDir'])
		if not os.path.exists(subdir):
			os.mkdir(subdir)
		destf = os.path.join(subdir,row['fileName']+'.fits.fz')
		if not os.path.exists(destf) and os.path.exists(rootf):
			print rootf,' --> ',destf
			os.rename(rootf,destf)

class CfhtDataMap(object):
	def __init__(self):
		cfhtDir = '.' # XXX
		self.obsDb = Table.read(os.path.join(cfhtDir,'config',
		                                     'sdssrm-cfht.fits.gz'))
		self.obsDb['frameIndex'] = self.obsDb['expNum'] # XXX
		if not os.path.exists(cfhtCatDir):
			os.makedirs(cfhtCatDir)
		for utDir in self.obsDb['utDir']:
			d = os.path.join(cfhtCatDir,utDir)
			if not os.path.exists(d):
				os.mkdir(d)
		self.fmap = { ftype:SimpleFileNameMap(cfhtImgDir,cfhtCatDir,'.'+ftype)
		                for ftype in ['cat','wcscat','psf'] }
		self.fmap['img'] = SimpleFileNameMap(cfhtImgDir,cfhtCatDir,
		                                     fromRaw=True)
	def getFiles(self,filt=None,with_frames=False):
		s = self.obsDb['good'].copy()
		f = np.char.add(np.char.add(self.obsDb['utDir'][s],'/'),
		                self.obsDb['fileName'][s])
		if with_frames:
			return f,np.where(s)[0]
		else:
			return f
	def __call__(self,ftype):
		return self.fmap[ftype]

if __name__=='__main__':
	#make_obs_db()
	dm = CfhtDataMap()
	move_files_to_subdirs(dm.obsDb)
	#update_obs_db()

