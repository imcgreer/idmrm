#!/usr/bin/env python

import os
import numpy as np
import subprocess
import multiprocessing
from functools import partial
from astropy.io import fits
from astropy.table import Table,vstack,hstack,join
from astropy.stats import sigma_clip
from astropy.wcs import InconsistentAxisTypesError

from bokpipe import bokphot,bokpl,bokproc,bokutil,bokastrom
from bokpipe.bokdm import SimpleFileNameMap
import bokrmpipe,bokrmphot
import cfhtrm
import idmrmphot

nom_pixscl = 0.18555

cfhtrm_aperRad = np.array([0.75,1.5,2.275,3.4,4.55,6.67,10.]) / nom_pixscl

def _cat_worker(dataMap,imFile,**kwargs):
	clobber = kwargs.pop('redo',False)
	verbose = kwargs.pop('verbose',0)
	bokutil.mplog('extracting catalogs for '+imFile)
	imgFile = dataMap('img')(imFile)
	psfFile = dataMap('psf')(imFile)
	aheadFile = imgFile.replace('.fits','.ahead')
	tmpFile = imgFile.replace('.fz','')
	catFile = dataMap('wcscat')(imFile)
	print '-->',imgFile
	kwargs.setdefault('SEEING_FWHM','1.0')
	kwargs.setdefault('PIXEL_SCALE','0.18555')
	kwargs.setdefault('SATUR_KEY','SATURATE')
	kwargs.setdefault('GAIN_KEY','GAIN')
	if not os.path.exists(catFile):
		if not os.path.exists(tmpFile):
			subprocess.call(['funpack',imgFile])
		bokphot.sextract(tmpFile,catFile,full=False,**kwargs)
	if not os.path.exists(psfFile):
		if not os.path.exists(tmpFile):
			subprocess.call(['funpack',imgFile])
		bokphot.run_psfex(catFile,psfFile,instrument='cfhtmegacam',**kwargs)
	if not os.path.exists(aheadFile):
		bokastrom.scamp_solve(tmpFile,catFile,filt='r',
		                      clobber=clobber,verbose=verbose)
	if False:
		os.remove(catFile)
	catFile = dataMap('cat')(imFile)
	# XXX while using these as primary
	apers = ','.join(['%.2f'%a for a in cfhtrm_aperRad])
	kwargs.setdefault('DETECT_MINAREA','10.0')
	kwargs.setdefault('DETECT_THRESH','2.0') 
	kwargs.setdefault('ANALYSIS_THRESH','2.0')
	kwargs.setdefault('PHOT_APERTURES',apers)
	kwargs.setdefault('PARAMETERS_NAME',
	                  os.path.join(bokphot.configDir,'cfht_catalog_tmp.par'))
	if not os.path.exists(catFile):
		if not os.path.exists(tmpFile):
			subprocess.call(['funpack',imgFile])
		bokphot.sextract(tmpFile,catFile,psfFile,full=True,**kwargs)
	if os.path.exists(tmpFile):
		os.remove(tmpFile)

def make_sextractor_catalogs(dataMap,procMap,**kwargs):
	files = dataMap.getFiles()
	p_cat_worker = partial(_cat_worker,dataMap,**kwargs)
	status = procMap(p_cat_worker,files)

def _phot_worker(dataMap,photCat,inp,matchRad=2.0,redo=False,verbose=0):
	imFile,frame = inp
	refCat = photCat.refCat
	catPfx = photCat.name
	fmap = SimpleFileNameMap(None,cfhtrm.cfhtCatDir,
	                         '.'.join(['',catPfx,'phot']))
	catFile = dataMap('cat')(imFile)
	aperFile = fmap(imFile)
	if verbose:
		print '--> ',imFile
	if os.path.exists(aperFile) and not redo:
		return
	tabs = []
	f = fits.open(catFile)
	for ccdNum,hdu in enumerate(f[1:]):
		c = hdu.data
		m1,m2,sep = bokrmphot.srcor(refCat['ra'],refCat['dec'],
		                            c['ALPHA_J2000'],c['DELTA_J2000'],matchRad)
		if len(m1)==0:
			continue
		expTime = dataMap.obsDb['expTime'][frame]
		t = Table()
		t['x'] = c['X_IMAGE'][m2]
		t['y'] = c['Y_IMAGE'][m2]
		t['objId'] = refCat['objId'][m1]
		t['counts'] = c['FLUX_APER'][m2] / expTime
		t['countsErr'] = c['FLUXERR_APER'][m2] / expTime
		t['flags'] = np.tile(c['FLAGS'][m2],(len(cfhtrm_aperRad),1)).T
		t['psfCounts'] = c['FLUX_PSF'][m2] / expTime
		t['psfCountsErr'] = c['FLUXERR_PSF'][m2] / expTime
		t['ccdNum'] = ccdNum
		t['frameIndex'] = dataMap.obsDb['frameIndex'][frame]
		t['__nmatch'] = len(m1)
		t['__sep'] = sep
		tabs.append(t)
	if len(tabs)==0:
		if verbose:
			print 'no objects!'
		return
	vstack(tabs).write(aperFile,overwrite=True)

def make_phot_catalogs(dataMap,procMap,photCat,**kwargs):
	files = zip(*dataMap.getFiles(with_frames=True))
	p_phot_worker = partial(_phot_worker,dataMap,photCat,**kwargs)
	status = procMap(p_phot_worker,files)

def _zp_worker(dataMap,photCat,instrCfg,inp):
	imFile,filt = inp
	catPfx = photCat.name
	fmap = SimpleFileNameMap(None,cfhtrm.cfhtCatDir,
	                         '.'.join(['',catPfx,'phot']))
	try:
		print imFile
		imCat = idmrmphot.load_catalog(fmap(imFile),filt,
		                               photCat.refCat,instrCfg,verbose=True)
	except IOError:
		return idmrmphot.generate_zptab_entry(instrCfg) # null entry
	return idmrmphot.image_zeropoint(imCat,instrCfg)

class cfhtCfg(object):
	name = 'cfht'
	nCCD = 40
	nAper = 7
	aperNum = -2
	ccd0 = 0
	minStar = 10
	magRange = {'g':(17.0,21.5),'i':(17.0,21.0)}
	def colorXform(self,mag,clr,band):
		return mag

def calc_zeropoints(dataMap,procMap,photCat,zpFile):
	files,frames = dataMap.getFiles(with_frames=True)
#	if True:
#		foo = np.where(dataMap.obsDb['frameIndex']==30)[0]
#		files = files[foo]
#		frames = frames[foo]
	filt = dataMap.obsDb['filter'][frames]
	p_zp_worker = partial(_zp_worker,dataMap,photCat,cfhtCfg())
	zps = procMap(p_zp_worker,zip(files,filt))
	zptab = vstack(zps)
	zptab['frameIndex'] = dataMap.obsDb['frameIndex'][frames]
	zptab.write(zpFile,overwrite=True)

def calibrate_lightcurves(dataMap,photCat,zpFile):
	zpTab = Table.read(zpFile)
	if True:
		# these are hacks to fill the zeropoints table for CCDs with no
		# measurements... this may be necessary as sometimes too few reference
		# stars will land on a given CCD. but need to understand it better.
		for row in zpTab:
			iszero = row['aperZp'] == 0
			if np.sum(~iszero) > 10:
				row['aperZp'][iszero] = np.median(row['aperZp'][~iszero])
				row['aperNstar'][iszero] = 999
			for j in range(7):
				iszero = row['aperCorr'][:,j] == 0
				if np.sum(~iszero) > 5:
					row['aperCorr'][iszero,j] = np.median(row['aperCorr'][~iszero,j])
	outFile = 'cfhtrm_%s.fits' % photCat.name
	catPfx = photCat.name
	fmap = SimpleFileNameMap(None,cfhtrm.cfhtCatDir,
	                         '.'.join(['',catPfx,'phot']))
	files_and_frames = dataMap.getFiles(with_frames=True)
	t = []
	for f,frameIdx in zip(*files_and_frames):
		try:
			tab = Table.read(fmap(f))
			tab['filter'] = dataMap.obsDb['filter'][frameIdx]
			t.append(tab)
		except IOError:
			pass
	photTab = vstack(t)
	idmrmphot.calibrate_lightcurves(photTab,zpTab,dataMap,outFile)

if __name__=='__main__':
	import sys
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--catalogs',action='store_true',
	                help='make source extractor catalogs and PSF models')
	parser.add_argument('--dophot',action='store_true',
	                help='make source extractor catalogs and PSF models')
	parser.add_argument('--zeropoint',action='store_true',
	                help='do zero point calculation')
	parser.add_argument('--lightcurves',action='store_true',
	                help='construct lightcurves')
	parser.add_argument('--aggregate',action='store_true',
	                help='construct aggregate photometry')
	parser.add_argument('--catalog',type=str,default='sdssrm',
	                help='reference catalog ([sdssrm]|sdss|cfht)')
	parser.add_argument('-p','--processes',type=int,default=1,
	                help='number of processes to use [default=single]')
	parser.add_argument('-R','--redo',action='store_true',
	                help='redo (overwrite existing files)')
	parser.add_argument('-u','--utdate',type=str,default=None,
	                help='UT date(s) to process [default=all]')
	parser.add_argument('--lctable',type=str,
	                help='lightcurve table')
	parser.add_argument('--zptable',type=str,default='cfhtrm_zeropoints.fits',
	                help='zeropoints table')
	parser.add_argument('--catdir',type=str,
	                help='directory containing photometry catalogs')
	parser.add_argument('-v','--verbose',action='count',
	                help='increase output verbosity')
	args = parser.parse_args()
	#
	if args.processes > 1:
		pool = multiprocessing.Pool(args.processes)
		procMap = pool.map
	else:
		procMap = map
	dataMap = cfhtrm.CfhtDataMap()
	photCat = bokrmphot.load_target_catalog(args.catalog,args.catdir,
	                                        args.lctable)
	photCat.load_ref_catalog()
	timerLog = bokutil.TimerLog()
	kwargs = dict(redo=args.redo,verbose=args.verbose)
	if args.utdate:
		utDate = args.utdate.split(',')
		dataMap.setUtDate(utDate)
	if args.catalogs:
		make_sextractor_catalogs(dataMap,procMap,**kwargs)
		timerLog('sextractor catalogs')
	if args.dophot:
		make_phot_catalogs(dataMap,procMap,photCat,**kwargs)
		timerLog('photometry catalogs')
	if args.zeropoint:
		calc_zeropoints(dataMap,procMap,photCat,args.zptable)
		timerLog('zeropoints')
	if args.lightcurves:
		calibrate_lightcurves(dataMap,photCat,args.zptable)
		timerLog('lightcurves')
	if args.aggregate:
		photCat.load_bok_phot(nogroup=True)
#		which = 'nightly' if args.nightly else 'all'
		which = 'all'
		bokrmphot.aggregate_phot(photCat,which)#,**kwargs)
		timerLog('aggregate phot')
	timerLog.dump()
	if args.processes > 1:
		pool.close()

