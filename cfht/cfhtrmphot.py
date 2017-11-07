#!/usr/bin/env python

import os,sys
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

def get_phot_file(photCat,inFile):
	if inFile is None:
		return '{0}_{1}.fits'.format('cfhtrmphot',photCat.name)
	else:
		return inFile

class CfhtConfig(object):
	name = 'cfht'
	nCCD = 40
	nAper = 7
	nAmp = 80
	ccd0 = 0
	zpAperNum = -2
	zpMinSnr = 10.
	zpMinNobs = 10
	zpMaxSeeing = 1.7/nom_pixscl
	zpMaxChiVal = 5.
	zpMagRange = {'g':(17.0,20.5),'i':(17.0,21.0)}
	zpFitKwargs = {'minContig':1}
	apCorrMaxRmsFrac = 0.5
	apCorrMinSnr = 20.
	apCorrMinNstar = 20
	maxFrameOutlierFrac = 0.02
	#colorxfun = Sdss2BokTransform()
	def colorXform(self,mag,clr,filt):
		return mag + -0.0761*clr + 0.08775 # XXX a crude fit to g

def _cat_worker(dataMap,imFile,**kwargs):
	clobber = kwargs.pop('redo',False)
	verbose = kwargs.pop('verbose',0)
	bokutil.mplog('extracting catalogs for '+imFile)
	imgFile = dataMap('img')(imFile)
	psfFile = dataMap('psf')(imFile)
	aheadFile = imgFile.replace('.fits.fz','.ahead')
	tmpFile = imgFile.replace('.fz','')
	catFile = dataMap('wcscat')(imFile)
	print '-->',imgFile
	kwargs.setdefault('SEEING_FWHM','1.0')
	kwargs.setdefault('PIXEL_SCALE','0.18555')
	kwargs.setdefault('SATUR_KEY','SATURATE')
	kwargs.setdefault('GAIN_KEY','GAIN')
	if not os.path.exists(aheadFile):
		print aheadFile,' not found!'
		return
	if not os.path.exists(imgFile):
		print imgFile,' not found!'
		return
	if True:
		# a few widely spaced ccds
		pix = np.array([ fits.getdata(imgFile,ccdNum)[::8] 
		                           for ccdNum in [10,16,21,33] ])
		sky = sigma_clip(pix).mean()
		if verbose > 0:
			print 'sky level is %.2f' % sky
		kwargs.setdefault('BACK_TYPE','MANUAL')
		kwargs.setdefault('BACK_VALUE','%.1f'%sky)
	if not os.path.exists(catFile):
		if not os.path.exists(tmpFile):
			subprocess.call(['funpack',imgFile])
		bokphot.sextract(tmpFile,catFile,full=False,
		                 clobber=clobber,verbose=verbose,**kwargs)
	if not os.path.exists(psfFile):
		if not os.path.exists(tmpFile):
			subprocess.call(['funpack',imgFile])
		bokphot.run_psfex(catFile,psfFile,instrument='cfhtmegacam',
		                  clobber=clobber,verbose=verbose,**kwargs)
	if not os.path.exists(aheadFile):
		bokastrom.scamp_solve(tmpFile,catFile,filt='r',
		                      clobber=clobber,verbose=verbose)
		if not os.path.exists(aheadFile):
			print imgFile,' WCS failed!'
			return
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
	#kwargs.setdefault('BACK_SIZE','64,128')
	#kwargs.setdefault('BACK_FILTERSIZE','1')
	kwargs.setdefault('BACKPHOTO_TYPE','LOCAL')
	#kwargs.setdefault('CHECKIMAGE_TYPE','BACKGROUND')
	#kwargs.setdefault('CHECKIMAGE_NAME',imgFile.replace('.fits.fz','.back.fits'))
	if not os.path.exists(catFile):
		if not os.path.exists(tmpFile):
			subprocess.call(['funpack',imgFile])
		bokphot.sextract(tmpFile,catFile,psfFile,full=True,
		                 clobber=clobber,verbose=verbose,**kwargs)
	if os.path.exists(tmpFile):
		os.remove(tmpFile)

def _exc_cat_worker(*args,**kwargs):
	try:
		_cat_worker(*args,**kwargs)
	except:
		pass

def make_sextractor_catalogs(dataMap,procMap,**kwargs):
	files = dataMap.getFiles()
	p_cat_worker = partial(_exc_cat_worker,dataMap,**kwargs)
	status = procMap(p_cat_worker,files)

def get_phot_fn(dataMap,imFile,catPfx):
	fmap = SimpleFileNameMap(None,cfhtrm.cfhtCatDir,
	                         '.'.join(['',catPfx,'phot']))
	catFile = dataMap('cat')(imFile)
	return fmap(imFile)

def _phot_worker(dataMap,photCat,inp,matchRad=2.0,redo=False,verbose=0):
	imFile,frame = inp
	refCat = photCat.refCat
	aperFile = get_phot_fn(dataMap,imFile,photCat.name)
	if verbose:
		print '--> ',imFile
	if os.path.exists(aperFile) and not redo:
		return
	tabs = []
	try:
		f = fits.open(catFile)
	except:
		print catFile,' not found!'
		return
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
		t['__number'] = c['NUMBER'][m2]
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

def load_raw_cfht_aperphot(dataMap,photCat):
	photTabs = []
	for imFile in dataMap.getFiles():
		aperFile = get_phot_fn(dataMap,imFile,photCat.name)
		try:
			photTabs.append(Table.read(aperFile))
			print "loaded catalog {}".format(aperFile)
		except IOError:
			print "WARNING: catalog {} missing, skipped!".format(aperFile)
	return vstack(photTabs)

def calc_zeropoints(dataMap,refCat,cfhtCfg,debug=False):
	#
	fields = ['frameIndex','utDate','filter','mjdStart','mjdMid','airmass']
	frameList = dataMap.obsDb[fields]
	frameList.sort('frameIndex')
	# zero point trends are fit over a season
	if 'season' not in frameList.colnames:
		frameList['season'] = idmrmphot.get_season(frameList['mjdStart'])
	# select the zeropoint aperture
	cfhtPhot = load_raw_cfht_aperphot(dataMap,refCat)
	# XXX temporary hack
	cfhtPhot['nMasked'] = np.int32(0)
	cfhtPhot['peakCounts'] = np.float32(1)
	phot = idmrmphot.extract_aperture(cfhtPhot,cfhtCfg.zpAperNum)
	# calculate zeropoints and aperture corrections
	zpdat = idmrmphot.iter_selfcal(phot,frameList,refCat,cfhtCfg,
	                               mode='focalplane')
	frameList = idmrmphot.calc_apercorrs(cfhtPhot,frameList,cfhtCfg,
	                                     mode='focalplane')
	#
	if True:
		zpdat.zptrend.write('cfht_zptrend.dat',overwrite=True,format='ascii')
	if debug:
		zpdat.sePhot.write('zp_sephot.fits',overwrite=True)
		zpdat.coaddPhot.write('zp_coaddphot.fits',overwrite=True)
	return frameList

def calibrate_lightcurves(dataMap,photCat,zpFile,cfhtCfg):
	zpTab = Table.read(zpFile)
	if False:
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
	phot = load_raw_cfht_aperphot(dataMap,photCat)
	phot = idmrmphot.calibrate_lightcurves(phot,zpTab,cfhtCfg,
	                                       zpmode='focalplane',
	                                       apcmode='focalplane')
	return phot

def check_status(dataMap):
	from collections import defaultdict
	from bokpipe.bokastrom import read_headers
	missing = defaultdict(list)
	incomplete = defaultdict(list)
	files = dataMap.getFiles()
	for i,f in enumerate(files):
		imgFile = dataMap('img')(f)
		if not os.path.exists(imgFile):
			missing['img'].append(f)
			continue
		nCCD = fits.getheader(imgFile,0)['NEXTEND']
		aheadFile = imgFile.replace('.fits.fz','.ahead')
		if not os.path.exists(aheadFile):
			missing['ahead'].append(f)
		else:
			hdrs = read_headers(aheadFile)
			if len(hdrs) < nCCD:
				incomplete['ahead'].append(f)
		for k in ['wcscat','psf','cat']:
			outFile = dataMap(k)(f)
			if not os.path.exists(outFile):
				missing[k].append(f)
			else:
				try:
					ff = fits.open(outFile)
				except IOError:
					incomplete[k].append(f)
					continue
				n = len(ff)-1
				if k == 'wcscat':
					n //= 2 # ldac
				if n < nCCD:
					incomplete[k].append(f)
		sys.stdout.write("\r%d/%d" % (i+1,len(files)))
		sys.stdout.flush()
	print
	print 'total images: ',len(files)
	for k in ['img','ahead','wcscat','psf','cat']:
		n = len(files) - len(missing[k]) - len(incomplete[k])
		print '%10s %5d %5d %5d' % (k,n,len(missing[k]),len(incomplete[k]))
	d = { f for l in missing.values() for f in l }
	if len(d)>0:
		logfile = open('missing.log','w')
		for f in d:
			logfile.write(f+'\n')
		logfile.close()
	d = { f for l in incomplete.values() for f in l }
	if len(d)>0:
		logfile = open('incomplete.log','w')
		for f in d:
			logfile.write(f+'\n')
		logfile.close()

if __name__=='__main__':
	import sys
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--catalogs',action='store_true',
	                help='make source extractor catalogs and PSF models')
	parser.add_argument('--dophot',action='store_true',
	                help='do photometry on images')
	parser.add_argument('--zeropoint',action='store_true',
	                help='do zero point calculation')
	parser.add_argument('--lightcurves',action='store_true',
	                help='construct lightcurves')
	parser.add_argument('--aggregate',action='store_true',
	                help='construct aggregate photometry')
	parser.add_argument('--binnedstats',action='store_true',
	                help='compute phot stats in mag bins')
	parser.add_argument('--status',action='store_true',
	                help='check processing status')
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
	parser.add_argument('--aper',type=int,default=-2,
	                help='index of aperture to select [-2]')
	parser.add_argument('--zptable',type=str,default='cfhtrm_zeropoints.fits',
	                help='zeropoints table')
	parser.add_argument('--outfile',type=str,default='',
	                help='output file')
	parser.add_argument('--photo',action='store_true',
	                help='use only photometric frames')
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
	photCat = idmrmphot.load_target_catalog(args.catalog)
	timerLog = bokutil.TimerLog()
	kwargs = dict(redo=args.redo,verbose=args.verbose)
	cfhtCfg = CfhtConfig()
	phot = None
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
		zps = calc_zeropoints(dataMap,photCat,cfhtCfg,debug=True)
		zps.write(args.zptable,overwrite=True)
		timerLog('zeropoints')
	if args.lightcurves:
		phot = calibrate_lightcurves(dataMap,photCat,args.zptable,cfhtCfg)
		photFile = get_phot_file(photCat,args.lctable)
		phot.write('lcs.fits',overwrite=True)
		timerLog('lightcurves')
	if args.aggregate:
		photCat.load_bok_phot(nogroup=True)
#		which = 'nightly' if args.nightly else 'all'
		which = 'all'
		bokrmphot.aggregate_phot(photCat,which,aperNum=-2)#,**kwargs)
		timerLog('aggregate phot')
	if args.binnedstats:
		if phot is None:
			photFile = get_phot_file(photCat,args.lctable)
			print 'loaded lightcurve catalog {}'.format(photFile)
			phot = Table.read(photFile)
		apPhot = idmrmphot.extract_aperture(phot,args.aper,
		                                    maskBits=(2**8-1),
		                                    lightcurve=True)
		if args.photo:
			if frameList is None:
				print 'loading zeropoints table {0}'.format(frameListFile)
				frameList = Table.read(frameListFile)
			photoFrames = frameList['frameIndex'][frameList['isPhoto']]
			nbefore = len(apPhot)
			apPhot = apPhot[np.in1d(apPhot['frameIndex'],photoFrames)]
			print 'restricting to {0} photo frames yields {1}/{2}'.format(
			          len(photoFrames),nbefore,len(apPhot))
		if True:
			# there's too little 2009 data for useful statistics
			apPhot['season'] = idmrmphot.get_season(apPhot['mjd'])
			apPhot = apPhot[apPhot['season']!='2009']
		bs = idmrmphot.get_binned_stats(apPhot,photCat.refCat,cfhtCfg,
		                                binEdges=np.arange(17.5,20.11,0.2))
		outfile = args.outfile if args.outfile else 'phot_stats_cfht.fits'
		bs.write(outfile,overwrite=True)
		timerLog('binned stats')
	if args.status:
		check_status(dataMap)
	timerLog.dump()
	if args.processes > 1:
		pool.close()

