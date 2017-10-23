#!/usr/bin/env python

import os
import numpy as np
import multiprocessing
from functools import partial
from astropy.io import fits
from astropy.table import Table,vstack,hstack,join
from astropy.stats import sigma_clip
from astropy.wcs import InconsistentAxisTypesError
from astropy.time import Time

k_ext = {'g':0.17,'i':0.06}

def match_by_id(tab1,tab2,idKey):
	# assumes the idKey field is sorted in tab2
	ii = np.searchsorted(tab2[idKey],tab1[idKey])
	assert np.all(tab1[idKey]==tab2[idKey][ii])
	return ii

def join_by_id(tab1,tab2,idKey):
	ii = match_by_id(tab1,tab2,idKey)
	# remove duplicate column names
	c2 = filter(lambda c: c not in tab1.colnames, tab2.colnames)
	return hstack([tab1,tab2[c2][ii]])

def in_range(arr,_range):
	return np.logical_and(arr >= _range[0], arr <= _range[1])

def map_group_to_items(tab):
	'''Given an astropy.table.Table that has been grouped, return the
	   set of indices that maps the groups back to the original table.
	   Useful for comparing aggregate values to individual measurements.
	   Example:
	     g = tab.group_by('object')
	     mean_mag = g['mag'].groups.aggregate(np.mean)
	     ii = map_group_to_items(g)
	     tab['delta_mag'] = g['mag'] - mean_mag[ii]
	'''
	return np.concatenate( 
	          [ np.repeat(i,np.diff(ii))
	              for i,ii in enumerate(zip(tab.groups.indices[:-1],
	                                        tab.groups.indices[1:])) ] )

##############################################################################
#
# target catalogs 
#
##############################################################################

# XXX make this one directory
sdssRmDataDir = os.path.join(os.environ['SDSSRMDIR'],'data')
bokRmDataDir = os.path.join(os.environ['BOKRMDIR'],'data')

class RmTargetCatalog(object):
	def __init__(self):
		self.refCat = Table.read(self.refCatFile)
		if 'objId' not in self.refCat.colnames:
			self.refCat['objId']  = np.arange(len(self.refCat))
		self.refCatOrig = self.refCat
	def select_targets(self,ii):
		self.refCat = self.refCat[ii]
	def ingest_rmphot(self,rmPhotName,rmPhotData):
		pass

class RmQsoCatalog(RmTargetCatalog):
	name = 'rmqso'
	refCatFile = os.path.join(sdssRmDataDir,'target_fibermap.fits')
	def __init__(self,**kwargs):
		super(RmQsoCatalog,self).__init__(**kwargs)
		if 'RA' in self.refCat.colnames:
			self.refCat.rename_column('RA','ra')
			self.refCat.rename_column('DEC','dec')

class AllQsoCatalog(RmTargetCatalog):
	name = 'allqso'
	refCatFile = os.path.join(sdssRmDataDir,'allqsos_rmfield.fits')
	def __init__(self,**kwargs):
		super(AllQsoCatalog,self).__init__(**kwargs)

class SdssStarCatalog(RmTargetCatalog):
	name = 'sdssstars'
	refCatFile = os.path.join(bokRmDataDir,
	                          'sdssRefStars_DR13.fits.gz')
	def __init__(self,**kwargs):
		super(SdssStarCatalog,self).__init__(**kwargs)
	def bin_stats_by_ref_mag(self,band='g',aperNum=3,binEdges=None):
		if binEdges is None:
			binEdges = np.arange(16.9,19.11,0.2)
		mbins = binEdges[:-1] + np.diff(binEdges)/2
		nbins = len(mbins)
		# group the reference objects into magnitude bins using ref mag
		self.load_ref_catalog()
		binNum = np.digitize(self.refCat[band],binEdges)
		ii = np.where((binNum>=1) & (binNum<=nbins))[0]
		refCat = self.refCat['objId',band][ii].copy()
		refCat['binNum'] = binNum[ii] - 1
		#
		phot = self.get_aperture_table(aperNum)
		phot = phot[(phot['filter']==band) & 
		            np.in1d(phot['objId'],refCat['objId']) ]
		objGroup = phot.group_by('objId')
		#
		minNobs = 10
		def min_nobs(table,key_colnames):
			return np.sum(~table['aperMag'].mask) > minNobs
		objGroup = objGroup.groups.filter(min_nobs)
		#
		ii = map_group_to_items(objGroup)
		nClipIter = 3
		sigThresh = 5.0
		mask = objGroup['aperMag'].mask.copy()
		objGroup['n'] = ~mask
		print 'before iter: ',np.sum(~mask)
		for iterNum in range(nClipIter):
			mean_mag,rms_mag = calc_mean_mags(objGroup,median=True,
			                                  extMask=mask)
			dmag = objGroup['aperMag'] - mean_mag[ii]
			nsig = dmag / rms_mag[ii]
			print 'max deviation: ',np.ma.abs(nsig[~mask]).max()
			if iterNum < nClipIter-1:
				mask[:] |= np.ma.greater(np.ma.abs(nsig),sigThresh)
				print 'after iter: ',iterNum+1,np.sum(~mask)
		objGroup['nClip'] = ~mask
		#
		aggPhot = join_by_objid(objGroup.groups.keys,refCat)
		aggPhot['meanMag'] = mean_mag
		aggPhot['rmsInt'] = rms_mag
		objGroup['dmagExt'] = objGroup['aperMag'] - aggPhot[band][ii]
		objGroup['dmagInt'] = dmag
		outies = objGroup['n','nClip'].groups.aggregate(np.ma.sum)
		aggPhot['outlierFrac'] = 1-outies['nClip']/outies['n'].astype(float)
		binAggPhot = aggPhot.group_by('binNum')
		binGroup = Table(binAggPhot.groups.keys)
		binGroup['mbins'] = mbins
		perc = lambda x,p: np.percentile(x.compressed(),p)
		for p in [25,50,75]:
			pc = binAggPhot['rmsInt'].groups.aggregate(lambda x: perc(x,p))
			binGroup['sig%d'%(p)] = pc
		binGroup['outlierFrac'] = binAggPhot['outlierFrac'].groups.aggregate(
		                                                        np.ma.mean)
		# external accuracy in mag bins
#		allAggPhot = objGroup.group_by('binNum')
#		binExtOff = allAggPhot['dmagExt'].groups.aggregate(np.ma.median)
#		binExtRms = allAggPhot['dmagExt'].groups.aggregate(np.ma.std)
#		binGroup['median_dmagExt'] = binExtOff
#		binGroup['rmsExt'] = binExtRms
		return binGroup

class SdssStarCatalogOld(SdssStarCatalog):
	name = 'sdssstarsold'
	refCatFile = os.path.join(sdssRmDataDir,'sdss.fits')
	def __init__(self,**kwargs):
		super(SdssStarCatalogOld,self).__init__(**kwargs)

class CleanSdssStarCatalog(SdssStarCatalog):
	name = 'sdssrefstars'
	refCatFile = os.path.join(bokRmDataDir,
	                          'sdssRefStars_DR13_clean.fits.gz')
	def __init__(self,**kwargs):
		super(CleanSdssStarCatalog,self).__init__(**kwargs)
		if not os.path.exists(self.refCatFile):
			self.__generate()
	def __generate(self):
		qsos = Table.read(os.path.join(sdssRmDataDir,'targets_final.fits'))
		refCat = Table.read(os.path.join(bokRmDataDir,
		                                 'sdssRefStars_DR13.fits.gz'))
		print 'starting with ',len(refCat)
		# remove objects in the quasar catalog
		m1,m2,s = srcor(refCat['ra'],refCat['dec'],qsos['RA'],qsos['DEC'],2.0)
		refCat.remove_rows(m1)
		print 'removed ',len(m1),' quasars from star catalog ',len(refCat)
		# remove objects with bad photometry; bad list comes 
		# from calc_group_stats
		f = os.path.join(bokRmDataDir,'sdssPhotSummary.fits')
		objSum = Table.read(f)
		for b in 'gi':
			refCat[b+'_orig'] = refCat[b].copy()
			ii = np.where((objSum['filter']==b)&(objSum['rchi2']>10))[0]
			bad = np.where(np.in1d(refCat['objId'],objSum['objId'][ii]))[0]
			refCat[b][bad] = 99.99
			print 'flagged ',len(bad),' noisy objects in band ',b
		# restrict mag range
		ii = np.where(((refCat['g']>17)&(refCat['g']<20.5)) |
		              ((refCat['i']>17)&(refCat['i']<20.5)))[0]
		refCat = refCat[ii]
		print 'have ',len(refCat),' clean stars after mag cut'
		# save the clean catalog
		refCat.write(self.refCatFile)

class CfhtStarCatalog(RmTargetCatalog):
	name = 'cfhtstars'
	refCatFile = os.path.join(sdssRmDataDir,'CFHTLSW3_starcat.fits')
	def __init__(self,**kwargs):
		super(CfhtStarCatalog,self).__init__(**kwargs)

def load_target_catalog(target,catdir,photfile):
	if catdir is None: catdir='.'
	targets = {
	  'sdssrm':RmQsoCatalog,
	  'allqsos':AllQsoCatalog,
	  'sdssall':SdssStarCatalog,
	  'sdss':CleanSdssStarCatalog,
	  'sdssold':SdssStarCatalogOld,
	  'cfht':CfhtStarCatalog,
	}
	return targets[target]()#catdir=catdir,photfile=photfile)


##############################################################################
#
# aggregate photometry
#
##############################################################################

def clipped_mean_phot(phot,magKey,ivarKey,sigma=3.0,iters=3,minNobs=10):
	ii = map_group_to_items(phot)
	phot['_mag'] = phot[magKey].copy()
	phot['_ivar'] = phot[ivarKey].copy()
	for iterNum in range(iters):
		nObs = np.sum(~phot['_mag'].mask)
		print 'starting iter {0} with {1} points'.format(iterNum+1,
		                                                 nObs)
		                                                 #phot['nObs'].sum())
		medMag = phot['_mag'].groups.aggregate(np.ma.median)
		medMag.mask[:] |= np.isnan(medMag)
		phot['dMag'] = np.ma.subtract(phot['_mag'],medMag[ii])
		rms_mag = phot['dMag'].groups.aggregate(np.ma.std)
		rms_mag.mask[:] |= rms_mag == 0
		dev = np.ma.abs(phot['dMag']/rms_mag[ii])
		reject = np.ma.greater(dev,sigma)
		print '... max deviation {0:.2f}, rejecting {1}'.format(dev.max(),
		                                                     reject.sum())
		phot['_mag'].mask[:] |= reject.filled(True)
		phot['dMag'].mask[:] |= reject.filled(True)
	phot['_ivar'].mask[:] |= phot['_mag'].mask
	phot['_wtmag'] = phot['_mag'] * phot['_ivar']
	phot['nObs'] = ~phot['_mag'].mask
	meanPhot = phot['_wtmag','_ivar','nObs'].groups.aggregate(np.ma.sum)
	meanPhot['meanMag'] = np.ma.divide(meanPhot['_wtmag'],meanPhot['_ivar'])
	meanPhot['rmsMag'] = phot['_mag'].groups.aggregate(np.ma.std)
#	import pdb; pdb.set_trace()
	phot['dMag'] = np.ma.subtract(phot[magKey],meanPhot['meanMag'][ii])
	del phot['_wtmag','_mag','_ivar','nObs']
	del meanPhot['_wtmag']
	meanPhot.rename_column('_ivar','magIvar')
	meanPhot['magIvar'].mask[:] |= meanPhot['meanMag'].mask
	meanPhot['rmsMag'].mask[:] |= meanPhot['meanMag'].mask
	for k in ['meanMag','magIvar','rmsMag']:
		meanPhot[k].fill_value = 0
		meanPhot[k] = meanPhot[k].astype(np.float32)
	meanPhot['nObs'] = meanPhot['nObs'].astype(np.int32)
	meanPhot = hstack([phot.groups.keys,meanPhot])
	return phot,meanPhot


##############################################################################
#
# zero points
#
##############################################################################

# XXX break this up into steps

def selfcal(rawPhot,frameList,refCat,mode='ccd',
            magRange=None,maxSeeing=None,minSnr=10,minNobs=10,
            maxChiVal=5.0):
	''' mode is 'focalplane','ccd','amp' '''
	frameList.sort('frameIndex')
	zp0 = np.float32(25)
	#
	if mode == 'focalplane':
		zpGroups = []
	elif mode == 'ccd':
		zpGroups = ['ccdNum']
	elif mode == 'amp':
		zpGroups = ['ampNum']
	else:
		raise ValueError
	#
	if magRange is None:
		magRange = {'g':(0,30),'i':(0,30)}
	if maxSeeing is None:
		maxSeeing = np.inf
	#
	try:
		isPhoto = frameList['isPhoto']
	except KeyError:
		isPhoto = np.ones(len(frameList),dtype=bool)
	try:
		fwhm = frameList['fwhmPix']
		if fwhm.ndim == 2:
			fwhm = fwhm.mean(axis=-1)
		isSee = fwhm < maxSeeing
	except KeyError:
		isSee = True
	goodFrames = frameList['frameIndex'][np.logical_and(isPhoto,isSee)]
	# match to frame database to get observing particulars
	frFields = ['frameIndex','filter','airmass','mjdStart']
	rawPhot = join_by_id(rawPhot,frameList[frFields],'frameIndex')
	# match to reference catalog to select only stars in desired magnitude
	# range, and also cut to photometric frames
	calPhot = join_by_id(rawPhot,refCat.refCat['objId','g','i'],'objId')
	isG = in_range(calPhot['g'],magRange['g'])
	isI = in_range(calPhot['i'],magRange['i'])
	isMag = np.choose(calPhot['filter']=='g',[isI,isG])
	goodFrame = np.in1d(calPhot['frameIndex'],goodFrames)
	# convert from raw counts to magnitudes
	calPhot['snr'] = np.ma.divide(calPhot['counts'],calPhot['countsErr'])
	calPhot['counts'].mask[:] |= calPhot['snr'] <= 5.0
	#
	k = np.choose(calPhot['filter']=='g',[ k_ext['i'], k_ext['g'] ])
	x = calPhot['airmass']
	calPhot['rawMag'] = zp0 - 2.5*np.ma.log10(calPhot['counts']) - k*x
	calPhot['magErr'] = 1.0856*np.ma.divide(calPhot['countsErr'],
	                                        calPhot['counts'])
	calPhot['magIvar'] = np.ma.power(calPhot['magErr'],-2)
	#
	ii = np.where(np.logical_and(isMag,goodFrame))[0]
	phot = calPhot[ii].group_by(['objId','filter'])
	phot,objPhot = clipped_mean_phot(phot,'rawMag','magIvar')
	#
	jj = np.where(np.logical_and(~objPhot['meanMag'].mask,
	                             objPhot['nObs'] > minNobs))[0]
	ii = np.where(np.logical_and(isMag,
	                     np.in1d(calPhot['objId'],objPhot['objId'][jj])))[0]
	#
	zpPhot = join_by_id(calPhot[ii],objPhot['objId','meanMag'][jj],'objId')
	#
	zpPhot['dMag'] = np.ma.subtract(zpPhot['rawMag'],zpPhot['meanMag'])
	zpPhot['dMag'].mask[:] |= zpPhot['snr'] <= minSnr
	zpPhot = zpPhot.group_by(['frameIndex']+zpGroups)
	zpPhot,framePhot = clipped_mean_phot(zpPhot,'dMag','magIvar')
	if mode == 'focalplane':
		frameList['aperZp'] = zp0 - framePhot['meanMag'].filled(zp0)
		frameList['aperZpRms'] = framePhot['rmsMag'].filled(0)
		frameList['aperNstar'] = framePhot['nObs'].filled(0)
	elif mode == 'ccd':
		nCcd,ccd0 = 4,1 # XXX
		frameList['aperZp'] = np.zeros((1,nCcd),dtype=np.float32)
		frameList['aperZpRms'] = np.zeros((1,nCcd),dtype=np.float32)
		frameList['aperNstar'] = np.zeros((1,nCcd),dtype=np.int32)
		framePhot = framePhot.group_by('ccdNum')
		for ccdNum,ccdzp in zip(framePhot.groups.keys['ccdNum'],
		                        framePhot.groups):
			ccdj = ccdNum - ccd0
			ii = match_by_id(ccdzp,frameList,'frameIndex')
			frameList['aperZp'][ii,ccdj] = zp0 - ccdzp['meanMag'].filled(zp0)
			frameList['aperZpRms'][ii,ccdj] = ccdzp['rmsMag'].filled(0)
			frameList['aperNstar'][ii,ccdj] = ccdzp['nObs'].filled(0)
	else:
		raise NotImplementedError
	# summary statistics
	zpPhot['chiVal'] = zpPhot['dMag']*np.ma.sqrt(zpPhot['magIvar'])
	zpPhot['chiSqr'] = np.ma.power(zpPhot['chiVal'],2)
	zpPhot['chiSqr'].mask[:] |= np.ma.greater(np.ma.abs(zpPhot['chiVal']),
	                                          maxChiVal)
	objByFrames = zpPhot.group_by('frameIndex')
	objByFrames['nStar'] = (~zpPhot['chiSqr'].mask).astype(np.int32)
	objByFrames['_ntot'] = (~zpPhot['chiVal'].mask).astype(np.float32)
	frameStats = objByFrames['nStar','chiSqr','_ntot'].groups.aggregate(
	                                                             np.ma.sum)
	for c in frameStats.colnames:
		frameStats[c].mask[:] |= frameStats['nStar'] == 0
	frameStats['outlierFrac'] = 1 - np.ma.divide(frameStats['nStar'],
	                                             frameStats['_ntot'])
	frameStats['rchi2'] = np.ma.divide(frameStats['chiSqr'],
	                                   frameStats['nStar']-1)
	del frameStats['_ntot']
	frameStats = hstack([objByFrames.groups.keys,frameStats])
	#
	ii = match_by_id(frameStats,frameList,'frameIndex')
	frameList['nStar'] = np.int32(0)
	frameList['nStar'][ii] = frameStats['nStar'].filled(0)
	for c in ['chiSqr','outlierFrac','rchi2']:
		frameList[c] = np.float32(0)
		frameList[c][ii] = frameStats[c].filled(0)
	#
	return zpPhot,objPhot,frameList

def add_photometric_flag(frameList,nIter=3,minContig=10):
	# resort the table by mjd (frameIndex may not be monotonic with time)
	zpTab = frameList[frameList['mjdStart'].argsort()]
	#
	zp = np.ma.array(zpTab['aperZp'],mask=zpTab['aperZp']==0)
	if zpTab['aperZp'].ndim > 1:
		zp = zp.mean(axis=-1)
	zpTab['_meanzp'] = zp.filled(0)
	# first group by season
	zpTab = zpTab.group_by(['season','filter'])
	zpTrend = Table(names=('season','filter','mjd0','dzpdt','zpt0'),
	                dtype=('S4','S1','f8','f4','f4'))
	#
	for (season,filt),sdat in zip(zpTab.groups.keys,zpTab.groups):
		meanZp = np.ma.array(sdat['_meanzp'],mask=sdat['_meanzp']==0)
		# and set the starting point to be day 0 of the season
		day0 = Time(season+'-01-01T00:00:00',format='isot',scale='utc')
		dMjd = sdat['mjdStart'] - day0.mjd
		# guess that the "true" zeropoint is around the 90th percentile
		# of the measured zeropoints
		zp90 = np.percentile(meanZp.compressed(),90)
		# and mask obviously non-photometric data
		msk = np.ma.greater(zp90 - meanZp, 0.25)
		# then iteratively fit a line to the zeropoints allow variation 
		# with time, rejecting low outliers at each iteration
		for iterNum in range(nIter):
			zpFit = np.polyfit(dMjd[~msk]/100,meanZp[~msk],1)
			resid = np.polyval(zpFit,dMjd/100) - meanZp
			msk |= resid > 0.1
		# now search for continguous images within 10% of fitted zeropoint
		ncontig = np.zeros(len(msk),dtype=int)
		for utd in np.unique(sdat['utDate']):
			jj = np.where(sdat['utDate']==utd)[0]
			if not np.any(msk[jj]):
				# the whole night was photometric
				ncontig[jj] = len(jj)
				continue
			i1 = 0
			for i2 in np.where(msk[jj])[0]:
				ncontig[jj[i1:i2]] = i2 - i1
				i1 = i2 + 1
			ncontig[jj[i1:]] = len(jj) - i1
		cmsk = (ncontig < minContig) & ~msk
		msk |= ncontig < minContig
		print '%s %s %d/%d photometric' % (season,filt,np.sum(~msk),len(msk))
		sdat['isPhoto'][:] &= np.logical_not(msk.filled())
		zpTrend.add_row( (season,filt,day0.mjd,zpFit[1],zpFit[0]) )
	return zpTab,zpTrend

def _get_sdss_offsets(coaddPhot,refCat,colorXform):
	phot2ref = join_by_id(coaddPhot,refCat,'objId')
	zpoff = {}
	for b in 'gi':
		bphot = phot2ref[phot2ref['filter']==b]
		mags = colorXform(bphot[b],
		                  bphot['g']-bphot['i'],
		                  bphot['filter'])
		dmag = bphot['meanMag'] - mags
		dmag = sigma_clip(dmag,sigma=2.0)
		zpoff[b] = dmag.mean()
		print '{0}: {1:5.2f} +/- {2:4.2f}'.format(b,dmag.mean(),dmag.std())
	return zpoff

def iter_selfcal(phot,frameList,refCat,**kwargs):
	nIter = kwargs.pop('calNiter',2)
	minNobs = kwargs.pop('calMinObs',10)
	maxRms = kwargs.pop('calMaxRms',0.2)
	colorXform = kwargs.pop('calColorXform')
	# this routine updates the frameList with zeropoint data
	frameList['isPhoto'] = True
	zpts = frameList
	for iterNum in range(nIter):
		sePhot,coaddPhot,zpts = selfcal(phot,zpts,refCat,**kwargs)
		#
		zpts,zptrend = add_photometric_flag(zpts)
	# get the zeropoint offset to SDSS to put Bok photometry on SDSS system
	ii = np.where( (coaddPhot['nObs'] > minNobs) &
	               (coaddPhot['rmsMag'] < maxRms) )[0] 
	zp0 = _get_sdss_offsets(coaddPhot[ii],refCat.refCat,colorXform)
	# back out the extinction correction
	for b in 'gi':
		ii = np.where(zpts['filter']==b)[0]
		kx = k_ext[b]*zpts['airmass'][ii][:,np.newaxis]
		zp = np.ma.array(zpts['aperZp'][ii],mask=zpts['aperZp']==0)
		zp = np.ma.subtract(zp, zp0[b] + kx)
		zpts['aperZp'][ii] = zp.filled(0)
	del sePhot['meanMag']
	return sePhot,coaddPhot,zpts,zptrend


##############################################################################
#
# helper routines for working with photometry tables
#
##############################################################################

def extract_aperture(phot,aperNum,maskBits=None,badFrames=None):
	# aperture-independent columns
	gCols = ['frameIndex','objId','ccdNum','nMasked','peakCounts']
	aphot = Table(phot[gCols].copy(),masked=True)
	# aperture-dependent columns
	aCols = ['counts','countsErr','flags']
	for c in aCols:
		aphot[c] = phot[c][:,aperNum]
	# apply masks
	if not maskBits:
		maskBits = 2**32 - 1
	mask = (aphot['flags'] & maskBits) > 0
	if badFrames is not None:
		mask[:] |= np.in1d(aphot['frameIndex'],badFrames)
	for c in ['counts','countsErr']:
		aphot[c].mask[:] |= mask
	return aphot

def calibrate_lightcurves(phot,zpts,zpmode='ccd'):
	ii = match_by_id(phot,zpts,'frameIndex')
	#
	if zpmode == 'focalplane':
		jj = None
	elif zpmode == 'ccd':
		ccd0 = 1 # XXX
		jj = phot['ccdNum'] - ccd0
	elif zpmode == 'amp':
		jj = phot['ampNum'] - amp0
	zp = zpts['aperZp'][ii,jj][:,np.newaxis]
	#
	apCorr = 1
	#
	corrCps = phot['counts'] * apCorr 
	poscounts = np.ma.array(corrCps,mask=phot['counts']<=0)
	magAB = zp - 2.5*np.ma.log10(poscounts)
	magErr = 1.0856*np.ma.divide(phot['countsErr'],
	                             np.ma.array(phot['counts'],
	                                         mask=poscounts.mask))
	phot['aperMag'] = magAB.filled(99.99)
	phot['aperMagErr'] = magErr.filled(0)
	# convert AB mag to nanomaggie
	fluxConv = 10**(-0.4*(zp-22.5))
	flux = corrCps * fluxConv
	fluxErr = phot['countsErr'] * apCorr * fluxConv
	phot['aperFlux'] = flux.filled(0)
	phot['aperFluxErr'] = fluxErr.filled(0)
	#
	obsdat = zpts['frameIndex','airmass','mjdMid'].copy()
	obsdat.rename_column('mjdMid','mjd')
	phot = join_by_id(phot,obsdat,'frameIndex')
	return phot

