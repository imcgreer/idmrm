#!/usr/bin/env python

import os
import itertools
from collections import namedtuple
import multiprocessing
from functools import partial
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.table import Table,vstack,hstack,join
from astropy.stats import sigma_clip
from astropy.wcs import InconsistentAxisTypesError
from astropy.time import Time

k_ext = {'g':0.17,'i':0.06}

idmPhotFlags = {
  'FRAME_MANY_OUTLIERS':(10,"frame containing object has many outliers"),
   'FRAME_LARGE_CHISQR':(11,"frame containing object has large chi-sqr"),
}

def get_season(mjd):
	'''splits the seasons on Dec. 1 of previous year'''
	seasons = np.arange(2000,2030)
	mjdstart = np.array([ Time(str(season-1)+'-12-01',format='iso').mjd
	                        for season in seasons ])
	ii = np.searchsorted(mjdstart,mjd) - 1
	return seasons.astype('S4')[ii]

##############################################################################
#
# table helpers
#
##############################################################################

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

def group_median_rms(tabGroup,withcount=False):
	gStat = Table()
	gStat = tabGroup.groups.aggregate(np.ma.median)
	gRms = tabGroup.groups.aggregate(np.ma.std)
	for k in tabGroup.colnames:
		gStat[k].mask[:] |= np.isnan(gStat[k])
		gRms[k].mask[:] |= np.isnan(gRms[k]) | (gRms[k]==0)
		gStat[k+'_rms'] = gRms[k]
	if withcount:
		n = tabGroup.groups.aggregate(lambda x: np.sum(~x.mask))
		for k in tabGroup.colnames:
			gStat[k+'_n'] = n[k]
	return gStat

def group_mean_rms(tabGroup,names=None,clean=True):
	if names is None:
		names = tabGroup.colnames + []
	for k in names:
		if k+'__sqr' not in tabGroup.colnames:
			tabGroup[k+'__sqr'] = np.ma.power(tabGroup[k],2)
			tabGroup[k+'__sqr'].mask = tabGroup[k].mask
			tabGroup[k+'_n'] = ~tabGroup[k].mask
	gStat = tabGroup.groups.aggregate(np.ma.sum)
	for k in names:
		gStat[k].mask[:] |= np.isnan(gStat[k])
		gStat[k+'__sqr'].mask[:] |= np.isnan(gStat[k+'__sqr'])
		gStat[k+'_n'].mask[:] |= np.isnan(gStat[k+'_n']) | (gStat[k+'_n']==0)
		gStat[k] /= gStat[k+'_n']
		gVar = ( np.ma.divide(gStat[k+'__sqr'],gStat[k+'_n']) - 
		          np.ma.power(gStat[k],2) )
		gStat[k+'_rms'] = np.ma.sqrt(gVar)
		if clean:
			del tabGroup[k+'__sqr',k+'_n']
		del gStat[k+'__sqr']
	return gStat

def clipped_group_mean_rms(tabGroup,iters=2,sigma=3.0,nmedian=1):
	origNames = tabGroup.colnames + []
	ii = map_group_to_items(tabGroup)
	for iterNum in range(iters):
		print 'iteration ',iterNum+1
		if iterNum < nmedian:
			gStat = group_median_rms(tabGroup,withcount=True)
		else:
			gStat = group_mean_rms(tabGroup,names=origNames,clean=False)
		for k in origNames:
			dev = np.ma.abs((tabGroup[k]-gStat[k][ii])/gStat[k+'_rms'][ii])
			rej = np.ma.greater(dev,sigma)
			tabGroup[k].mask[:] |= rej
			try:
				tabGroup[k+'__sqr'].mask[:] |= rej
				tabGroup[k+'_n'][rej] = False
			except KeyError:
				pass
	return group_mean_rms(tabGroup,names=origNames)

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
		# XXX
		raise NotImplementedError("this file no longer exists")
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

def load_target_catalog(target):
	targets = {
	  'sdssrm':RmQsoCatalog,
	  'allqsos':AllQsoCatalog,
	  'sdssall':SdssStarCatalog,
	  'sdss':CleanSdssStarCatalog,
	  'sdssold':SdssStarCatalogOld,
	  'cfht':CfhtStarCatalog,
	}
	return targets[target]()

class ColorTransform(object):
	colorMin = 0.3
	colorMax = 2.8
	def __init__(self,photsys,refsys,clearZero=False):
		_cfgdir = os.path.join(os.environ['BOKRMDIR'],'..') # XXX
		self.data = Table.read(os.path.join(_cfgdir,'colorterms.fits'))
		self.photSys = photsys
		self.refSys = refsys
		ii = np.where( (self.data['photsys']==photsys) &
		               (self.data['refsys']==refsys) )[0]
		self.data = self.data[ii]
		if clearZero:
			self.data['cterms'][:,-1] = 0
	def get_epoch(self,filt,mjd):
		jj = np.where(self.data['filter']==filt)[0]
		return np.digitize(mjd,self.data['mjdmax'][jj])
	def __call__(self,mags,colors,filt,mjd,masked=True):
		corrMags = mags.copy()
		for b in np.unique(filt):
			epochs = self.get_epoch(b,mjd)
			for epoch in np.unique(epochs):
				ii = np.where((filt==b) & (epoch==epochs))[0]
				j = np.where( (self.data['filter']==b) &
				              (self.data['epoch']==epoch) )[0][0]
				corrMags[ii] += np.polyval(self.data['cterms'][j],
				                           colors[ii])
		if masked:
			corrMags = np.ma.array(corrMags,
			                       mask=(colors<self.colorMin) |
			                            (colors>self.colorMax))
		return corrMags


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
		gStats = group_median_rms(phot['_mag',])
		phot['dMag'] = np.ma.subtract(phot['_mag'],gStats['_mag'][ii])
		dev = np.ma.abs(phot['dMag']/gStats['_mag_rms'][ii])
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

def get_frame_groups(mode,instrCfg):
	if mode == 'focalplane':
		groupCols = ['frameIndex',]
		nGroup = 1
		indxfun = lambda t: 0
	elif mode == 'ccd':
		groupCols = ['frameIndex','ccdNum']
		nGroup = instrCfg.nCcd
		indxfun = lambda t: t['ccdNum'] - instrCfg.ccd0
	elif mode == 'amp':
		groupCols = ['frameIndex','ampNum']
		nGroup = instrCfg.nAmp
		indxfun = lambda t: t['ampNum'] - instrCfg.amp0
	else:
		raise ValueError
	fg = namedtuple('frameGroups',['colnames','ngroup','groupindex'])
	return fg(groupCols,nGroup,indxfun)

# XXX break this up into steps

def selfcal(rawPhot,frameList,refCat,instrCfg,
            flatField=None,mode='ccd',initzps=True):
	''' mode is 'focalplane','ccd','amp' '''
	frameList.sort('frameIndex')
	zp0 = np.float32(25)
	zpGroups = get_frame_groups(mode,instrCfg)
	#
	try:
		isPhoto = frameList['isPhoto']
	except KeyError:
		isPhoto = np.ones(len(frameList),dtype=bool)
	try:
		fwhm = frameList['fwhmPix']
		if fwhm.ndim == 2:
			fwhm = fwhm.mean(axis=-1)
		isSee = fwhm < instrCfg.zpMaxSeeing
	except KeyError:
		isSee = True
	# initialize the zeropoint columns
	if initzps:
		frameList['aperZp'] = np.zeros((1,zpGroups.ngroup),dtype=np.float32)
		frameList['aperZpRms'] = np.zeros((1,zpGroups.ngroup),dtype=np.float32)
		frameList['aperZpNstar'] = np.zeros((1,zpGroups.ngroup),dtype=np.int32)
	#
	goodFrames = frameList['frameIndex'][np.logical_and(isPhoto,isSee)]
	# match to frame database to get observing particulars
	frFields = ['frameIndex','filter','airmass','mjdStart']
	rawPhot = join_by_id(rawPhot,frameList[frFields],'frameIndex')
	# match to reference catalog to select stars in desired magnitude range 
	calPhot = join_by_id(rawPhot,refCat.refCat['objId','g','i'],'objId')
	isG = in_range(calPhot['g'],instrCfg.zpMagRange['g'])
	isI = in_range(calPhot['i'],instrCfg.zpMagRange['i'])
	isMag = np.choose(calPhot['filter']=='g',[isI,isG])
	calPhot = calPhot[isMag]
	# identify photometric frames
	goodFrame = np.in1d(calPhot['frameIndex'],goodFrames)
	# convert from raw counts to magnitudes, mask low S/N measurements
	calPhot['snr'] = np.ma.divide(calPhot['counts'],calPhot['countsErr'])
	calPhot['counts'].mask[:] |= calPhot['snr'] <= 5.0
	# apply the extinction and flatfield corrections to get raw internal mags
	k = np.choose(calPhot['filter']=='g',[ k_ext['i'], k_ext['g'] ])
	x = calPhot['airmass']
	if flatField is None:
		ff = 0.
	else:
		ff = flatField(calPhot['filter'],calPhot['mjdStart'],
		               calPhot['ccdNum'],calPhot['x'],calPhot['y'])
	rawMag = -2.5*np.ma.log10(calPhot['counts'])
	calPhot['rawMag'] = zp0 + (rawMag - k*x + ff)
	calPhot['magErr'] = 1.0856*np.ma.divide(calPhot['countsErr'],
	                                        calPhot['counts'])
	calPhot['magIvar'] = np.ma.power(calPhot['magErr'],-2)
	# now group by object and filter to get mean magnitudes across only
	# the *photometric* frames
	sePhot = calPhot.group_by(['objId','filter'])
	sePhot['_rawMag'] = sePhot['rawMag'].copy()
	sePhot['_rawMag'].mask[:] |= ~goodFrame
	sePhot,coaddPhot = clipped_mean_phot(sePhot,'_rawMag','magIvar')
	# calculate dm = mag_i - <mag>
	ii = map_group_to_items(sePhot)
	sePhot['dMag'] = np.ma.subtract(sePhot['rawMag'],coaddPhot['meanMag'][ii])
	sePhot['dMag'].mask[:] |= sePhot['snr'] < instrCfg.zpMinSnr
	sePhot['dMag'].mask[:] |= coaddPhot['nObs'][ii] < instrCfg.zpMinNobs
	# now regroup by the zeropoint groups (image,[ccd|amp]) and determine
	# the relative zeropoint from the average mean mag difference on each img
	# this includes the non-photometric frames (goodFrame mask not used)
	sePhot = sePhot.group_by(zpGroups.colnames)
	sePhot,framePhot = clipped_mean_phot(sePhot,'dMag','magIvar')
	# fill the zeropoint columns in the frame table
	ii = match_by_id(framePhot,frameList,'frameIndex')
	jj = zpGroups.groupindex(framePhot)
	frameList['aperZp'][ii,jj] = zp0 - framePhot['meanMag'].filled(zp0)
	frameList['aperZpRms'][ii,jj] = framePhot['rmsMag'].filled(0)
	frameList['aperZpNstar'][ii,jj] = framePhot['nObs'].filled(0)
	# summary statistics for each frame calculated by regrouping the
	# object photometry and getting chi-sqr values summed over the frame
	# first apply the relative zeropoints back to the mag differences
	ii = map_group_to_items(sePhot)
	sePhot['dMag'] += framePhot['meanMag'][ii]
	sePhot['chiVal'] = sePhot['dMag']*np.ma.sqrt(sePhot['magIvar'])
	sePhot['chiSqr'] = np.ma.power(sePhot['chiVal'],2)
	sePhot['chiSqr'].mask[:] |= np.ma.greater(np.ma.abs(sePhot['chiVal']),
	                                          instrCfg.zpMaxChiVal)
	objsByFrame = sePhot.group_by('frameIndex')
	objsByFrame['nStar'] = (~objsByFrame['chiSqr'].mask).astype(np.int32)
	objsByFrame['_ntot'] = (~objsByFrame['chiVal'].mask).astype(np.float32)
	frameStats = objsByFrame['nStar','chiSqr','_ntot'].groups.aggregate(
	                                                             np.ma.sum)
	for c in frameStats.colnames:
		frameStats[c].mask[:] |= frameStats['nStar'] == 0
	frameStats['outlierFrac'] = 1 - np.ma.divide(frameStats['nStar'],
	                                             frameStats['_ntot'])
	frameStats['rchi2'] = np.ma.divide(frameStats['chiSqr'],
	                                   frameStats['nStar']-1)
	del frameStats['_ntot']
	frameStats = hstack([objsByFrame.groups.keys,frameStats])
	# populate the frame table with the summary statistic columns
	ii = match_by_id(frameStats,frameList,'frameIndex')
	frameList['nStar'] = np.int32(0)
	frameList['nStar'][ii] = frameStats['nStar'].filled(0)
	for c in ['chiSqr','outlierFrac','rchi2']:
		frameList[c] = np.float32(0)
		frameList[c][ii] = frameStats[c].filled(0)
	#
	del sePhot['_rawMag']
	return sePhot,coaddPhot,frameList

def add_photometric_flag(frameList,nIter=3,minContig=10,
                         maxTrendDev=0.05,maxRms=0.02):
	# resort the table by mjd (frameIndex may not be monotonic with time)
	keys = ['frameIndex','utDate','season','filter','mjdStart',
	        'aperZp','isPhoto']
	mjd_ii = frameList['mjdStart'].argsort()
	zpTab = Table(frameList[keys][mjd_ii],masked=True)
	#
	zp = np.ma.array(zpTab['aperZp'],mask=zpTab['aperZp']==0)
	if zpTab['aperZp'].ndim > 1:
		zp = zp.mean(axis=-1)
	zpTab['_meanzp'] = zp
	# first group by season
	zpTab = zpTab.group_by(['season','filter'])
	zpTrend = Table(names=('season','filter','mjd0','dzpdt','zpt0'),
	                dtype=('S4','S1','f8','f4','f4'))
	# fit a linear trend to the zeropoints across an observing season
	# to account for slow decline (likely due to mirror reflectivity)
	for (season,filt),sdat in zip(zpTab.groups.keys,zpTab.groups):
		# and set the starting point to be day 0 of the season
		day0 = Time(season+'-01-01T00:00:00',format='isot',scale='utc')
		dMjd = sdat['mjdStart'] - day0.mjd
		# not enough data to determine a mean zeropoint
		if np.sum(~sdat['_meanzp'].mask) < 5:
			sdat['_meanzp'].mask[:] = True
			continue
		# guess that the "true" zeropoint is around the 75th percentile
		# of the measured zeropoints
		zpGuess = np.percentile(sdat['_meanzp'].compressed(),75)
		# and mask obviously non-photometric data
		msk = np.ma.greater(zpGuess - sdat['_meanzp'], 0.25)
		# then iteratively fit a line to the zeropoints allow variation 
		# with time, rejecting low outliers at each iteration
		for iterNum in range(nIter):
			zpFit = np.ma.polyfit(dMjd[~msk]/100,sdat['_meanzp'][~msk],1)
			resid = np.polyval(zpFit,dMjd/100) - sdat['_meanzp']
			msk |= np.ma.greater(resid,maxTrendDev)
		# now search for continguous images within 10% of fitted zeropoint
		if minContig > 1:
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
		zpTrend.add_row( (season,filt,day0.mjd,zpFit[0],zpFit[1]) )
	# now reject nights with too much variation in the zeropoint
	zpTab['_meanzp_p'] = zpTab['_meanzp'].copy()
	zpTab['_meanzp_p'].mask |= ~zpTab['isPhoto']
	zpTab = zpTab.group_by(['utDate','filter'])
	ii = map_group_to_items(zpTab)
	nightMean = zpTab['_meanzp_p'].groups.aggregate(np.ma.mean)
	nightMean.mask[np.isnan(nightMean)] = True
	toolow = zpTab['_meanzp_p'] - nightMean[ii] < -maxRms
	zpTab['_meanzp_p'].mask |= toolow
	zpTab['isPhoto'][toolow] = False
	nightRms = zpTab['_meanzp_p'].groups.aggregate(np.ma.std)
	nightRms.mask[np.isnan(nightRms)] = True
	badNights = np.where(np.ma.greater(nightRms,maxRms))[0]
	for night in badNights:
		zpTab.groups[night]['isPhoto'] = False
	isPhoto = zpTab['isPhoto'][mjd_ii.argsort()].filled()
	return isPhoto,zpTrend

class FlatField(object):
	def __init__(self,zpTrend=None):
		self.zpTrend = None
		self.imgMjds = {'g':[],'i':[]}
		self.images = {'g':[],'i':[]}
		if zpTrend:
			self.setPedestal(zpTrend)
	def setPedestal(self,zpTrend):
		self.zpTrend = zpTrend.copy()
		self.zpTrend['_dmag0'] = 0.0
		for b in 'gi':
			ii = np.where(self.zpTrend['filter']==b)[0]
			if len(ii)==0:
				continue
			dmag0 = self.zpTrend['zpt0'][ii] - self.zpTrend['zpt0'][ii].max()
			self.zpTrend['_dmag0'][ii] = dmag0
	def getPedestal(self,filt,mjd):
		epoch = np.zeros(len(mjd),dtype=np.int32)
		for b in 'gi':
			ii = np.where(filt==b)[0]
			if len(ii)==0:
				continue
			jj = np.where(self.zpTrend['filter']==b)[0]
			j = np.searchsorted(self.zpTrend['mjd0'][jj],mjd[ii]) - 1
			j[j<0] = 0 # in case some come before first mjd0
			epoch[ii] = jj[j]
		dMjd = mjd - self.zpTrend['mjd0'][epoch]
		# the zpfit trends only apply to a single season 
		#    (which may start in Dec)
		assert np.all( (dMjd >= -30) & (dMjd < 330) )
		zpslope = self.zpTrend['dzpdt'][epoch]*dMjd/100
		return self.zpTrend['_dmag0'][epoch] + zpslope
	def addImage(self,mjd,filt,img):
		self.imgMjds[filt].append(mjd)
		self.images[filt].append(img)
	def __call__(self,filt,mjd,ccdNum,x,y):
		ff = 0.
		if self.zpTrend:
			ff += self.getPedestal(filt,mjd)
		return ff

def _get_sdss_offsets(coaddPhot,refCat,colorXform,meanMjd):
	phot2ref = join_by_id(coaddPhot,refCat,'objId')
	# this assumes the color transformation at the mean epoch can be
	# applied to all the data
	phot2ref['mjd'] = meanMjd
	zpoff = {}
	for b in 'gi':
		bphot = phot2ref[phot2ref['filter']==b]
		if len(bphot)==0:
			continue
		mags = colorXform(bphot[b],
		                  bphot['g']-bphot['i'],
		                  bphot['filter'],bphot['mjd'])
		dmag = bphot['meanMag'] - mags
		dmag = sigma_clip(dmag,sigma=2.0)
		zpoff[b] = dmag.mean()
		print '{0}[{3:.1f}]: {1:5.2f} +/- {2:4.2f}'.format(b,
		                                     dmag.mean(),dmag.std(),meanMjd)
	return zpoff

def iter_selfcal(phot,frameList,refCat,instrCfg,**kwargs):
	nIter = kwargs.pop('calNiter',2)
	minNobs = kwargs.pop('calMinObs',10)
	maxRms = kwargs.pop('calMaxRms',0.2)
	# this routine updates the frameList with zeropoint data
	frameList['isPhoto'] = True
	zpts = frameList
	flatField = None
	for iterNum in range(nIter):
		sePhot,coaddPhot,zpts = selfcal(phot,zpts,refCat,instrCfg,
		                                flatField=flatField,**kwargs)
		if iterNum < nIter-1:
			isPhoto,zpTrend = add_photometric_flag(zpts,
			                                       **instrCfg.zpFitKwargs)
			zpts['isPhoto'] = isPhoto
			flatField = FlatField(zpTrend)
	# get the zeropoint offset to SDSS to put photometry on SDSS system
	ii = np.where( (coaddPhot['nObs'] > minNobs) &
	               (coaddPhot['rmsMag'] < maxRms) )[0] 
	zp0 = _get_sdss_offsets(coaddPhot[ii],refCat.refCat,instrCfg.colorXform,
	                        zpts['mjdStart'].mean())
	# back out the extinction and flatfield corrections
	for b in 'gi':
		ii = np.where(zpts['filter']==b)[0]
		if len(ii)==0:
			continue
		kx = k_ext[b]*zpts['airmass'][ii][:,np.newaxis]
		ff = flatField.getPedestal(zpts['filter'][ii],
		                           zpts['mjdStart'][ii])[:,np.newaxis]
		zp = np.ma.array(zpts['aperZp'][ii],mask=zpts['aperZp'][ii]==0)
		zp = zp - zp0[b] - kx + ff
		zpts['aperZp'][ii] = zp.filled(0)
	rv = namedtuple("zpdat",["sePhot","coaddPhot","zpts","zptrend"])
	return rv(sePhot,coaddPhot,zpts,zpTrend)


##############################################################################
#
# aperture corrections
#
##############################################################################

def calc_apercorrs(rawPhot,frameList,instrCfg,mode='ccd',
                   iters=2,sigma=2.0,interpbad=True):
	''' mode is 'focalplane','ccd','amp' 
	    assumes input list only contains stars'''
	frameList.sort('frameIndex')
	nAper = rawPhot['counts'].shape[-1] # XXX
	apGroups = get_frame_groups(mode,instrCfg)
	#
	refAper = instrCfg.zpAperNum
	counts = np.ma.array(rawPhot['counts'],mask=rawPhot['flags']>0)
	cerr = np.ma.array(rawPhot['countsErr'],mask=rawPhot['countsErr']==0)
	snr = np.ma.divide(counts[:,refAper],cerr[:,refAper])
	ii = np.where(snr.filled(0) > instrCfg.apCorrMinSnr)[0]
	phot = Table(rawPhot[apGroups.colnames][ii],masked=True)
	phot['counts'] = counts[ii]
	phot['snr'] = snr[ii]
	# have to split the multidim column into singledim columns for aggregate
	apNames = []
	for j in range(nAper):
		k = '_apc_%02d' % j
		phot[k] = np.ma.divide(phot['counts'][:,refAper],
		                       phot['counts'][:,j])
		apNames.append(k)
	# compute the sigma-clipped mean aperture correction in each group
	phot = phot.group_by(apGroups.colnames)
	gStat = clipped_group_mean_rms(phot[apNames],iters=4,sigma=2.5,nmedian=2)
	ii = match_by_id(phot.groups.keys,frameList,'frameIndex')
	# back to multidim arrays
	frameList['aperCorr'] = np.zeros((1,apGroups.ngroup,nAper),dtype='f4')
	frameList['aperCorrRms'] = np.zeros((1,apGroups.ngroup,nAper),dtype='f4')
	frameList['aperCorrNstar'] = np.zeros((1,apGroups.ngroup,nAper),dtype='i4')
	jj = apGroups.groupindex(phot.groups.keys)
	def _restack_arr(a,sfx):
		return np.dstack([a[k+sfx].filled(0) for k in apNames]).squeeze()
	frameList['aperCorr'][ii,jj] = _restack_arr(gStat,'')
	frameList['aperCorrRms'][ii,jj] = _restack_arr(gStat,'_rms')
	frameList['aperCorrNstar'][ii,jj] = _restack_arr(gStat,'_n')
	# fill bad / missing values
	if interpbad and mode != 'focalplane':
		apCorr = np.ma.array(frameList['aperCorr'],
		                     mask=frameList['aperCorr']==0)
		fracRms = np.ma.divide(frameList['aperCorrRms'],apCorr)
		apCorr.mask[:] |= ( (frameList['aperCorrNstar'] 
		                       < instrCfg.apCorrMinNstar) |
		                    (frameList['aperCorrRms'] == 0) |
		                    (fracRms > instrCfg.apCorrMaxRmsFrac) )
		meanCorr = apCorr.mean(axis=1)
		nGood = (~apCorr.mask).sum(axis=1)
		meanCorr.mask[:] |= nGood < apGroups.ngroup//2
		for j in range(nAper):
			if j==range(nAper)[refAper]:
				meanCorr[:,j] = 1.0
				continue
			ii = ~meanCorr.mask[:,j] # good values
			fillfun = interp1d(frameList['mjdStart'][ii],
			                   meanCorr[ii,j].filled(),
			                   fill_value='extrapolate')
			meanCorr[~ii,j] = fillfun(frameList['mjdStart'][~ii])
		meanCorr = np.tile(meanCorr[:,np.newaxis,:],(1,4,1)) 
		# now replace the masked entries
		frameList['aperCorr'][apCorr.mask] = meanCorr[apCorr.mask].filled()
	return frameList

##############################################################################
#
# helper routines for working with photometry tables
#
##############################################################################

def extract_aperture(phot,aperNum,maskBits=None,lightcurve=False):
	# aperture-independent columns
	gCols = ['frameIndex','objId','ccdNum','x','y']
	if 'peakCounts' in phot.colnames: # XXX
		gCols += ['nMasked','peakCounts']
	if lightcurve:
		gCols += ['airmass','mjd','filter']
	aphot = Table(phot[gCols].copy(),masked=True)
	# aperture-dependent columns
	apCols = ['counts','countsErr']
	if 'aperMag' in phot.colnames:
		apCols += ['aper'+s1+s2 for s1 in ['Mag','Flux'] for s2 in ['','Err']]
	for c in apCols+['flags']:
		aphot[c] = phot[c][:,aperNum]
	# apply masks
	if not maskBits:
		maskBits = 2**32 - 1
	mask = (aphot['flags'] & maskBits) > 0
	for c in apCols:
		aphot[c].mask[:] |= mask
	return aphot

def calibrate_lightcurves(phot,frameList,instrCfg,zpmode='ccd',apcmode='ccd'):
	ii = match_by_id(phot,frameList,'frameIndex')
	# get the zeropoints according to their grouping (frame,[ccd|amp])
	zpGroups = get_frame_groups(zpmode,instrCfg)
	jj = zpGroups.groupindex(phot)
	if frameList['aperZp'].ndim==1: # XXX hack
		zp = frameList['aperZp'][ii][:,np.newaxis]
	else:
		zp = frameList['aperZp'][ii,jj][:,np.newaxis]
	zp = np.ma.array(zp,mask=(zp==0))
	# get the aperture corrs according to their grouping (frame,[ccd|amp])
	apGroups = get_frame_groups(apcmode,instrCfg)
	jj = apGroups.groupindex(phot)
	apCorr = frameList['aperCorr'][ii,jj]
	apCorr = np.ma.array(apCorr,mask=(apCorr==0))
	# apply the zeropoints and aperture corrections to get calibrated
	# fluxes and magnitudes
	corrCps = phot['counts'] * apCorr 
	poscounts = np.ma.array(corrCps,mask=phot['counts']<=0)
	magAB = zp - 2.5*np.ma.log10(poscounts)
	magErr = 1.0856*np.ma.divide(phot['countsErr'],
	                             np.ma.array(phot['counts'],
	                                         mask=poscounts.mask))
	phot['aperMag'] = magAB.filled(99.99)
	phot['aperMagErr'] = magErr.filled(0) # convert AB mag to nanomaggie
	fluxConv = 10**(-0.4*(zp-22.5))
	flux = corrCps * fluxConv
	fluxErr = phot['countsErr'] * apCorr * fluxConv
	phot['aperFlux'] = flux.filled(0)
	phot['aperFluxErr'] = fluxErr.filled(0)
	# add flags based on frame statistics
	try:
		badFrame = np.where(frameList['outlierFrac'] > 
		                      instrCfg.maxFrameOutlierFrac)[0]
		ii = np.in1d(phot['frameIndex'],frameList['frameIndex'][badFrame])
		phot['flags'][ii] |= 1 << idmPhotFlags['FRAME_MANY_OUTLIERS'][0]
	except AttributeError:
		pass
	try:
		badFrame = np.where(frameList['rchi2'] > 
		                      instrCfg.maxFrameChiSqrNu)[0]
		ii = np.in1d(phot['frameIndex'],frameList['frameIndex'][badFrame])
		phot['flags'][ii] |= 1 << idmPhotFlags['FRAME_LARGE_CHISQR'][0]
	except AttributeError:
		pass
	# include some basic observing particulars useful for lightcurves
	obsdat = frameList['frameIndex','airmass','mjdMid','filter'].copy()
	obsdat.rename_column('mjdMid','mjd')
	phot = join_by_id(phot,obsdat,'frameIndex')
	return phot

def get_binned_stats(apPhot,refCat,instrCfg,binEdges=None,minNobs=10,
                     **kwargs):
	if binEdges is None:
		binEdges = np.arange(16.9,19.51,0.2)
	pctiles = [25,50,75,90,99]
	mbins = binEdges[:-1] + np.diff(binEdges)/2
	nbins = len(mbins)
	#
	if 'season' not in apPhot.colnames:
		apPhot['season'] = get_season(apPhot['mjd'])
	#
	print '[{0:5d}] initial photometry table'.format(len(apPhot))
	#
	print '...matching to reference catalog'
	ii = match_by_id(apPhot,refCat,'objId')
	print '...color transformation to get external reference mags'
	refMag = np.choose(apPhot['filter']=='g',
	                   [refCat['i'][ii],refCat['g'][ii]])
	refClr = refCat['g'][ii] - refCat['i'][ii]
	refMag = instrCfg.colorXform(refMag,refClr,
	                             apPhot['filter'],apPhot['mjd'])
	jj = np.where(np.logical_and(refMag>binEdges[0],refMag<binEdges[-1]))[0]
	print '[{0:5d}] select within reference mag range'.format(len(jj))
	apPhot = apPhot[jj]
	apPhot['refMag'] = refMag[jj]
	apPhot['dMagExt'] = apPhot['aperMag'] - apPhot['refMag']
	# calc mean mags, internal calibration mag differences
	print '...calculate internal mean magnitudes'
	apPhot = apPhot.group_by(['season','filter','objId'])
	objPhot = clipped_group_mean_rms(apPhot['aperMag','dMagExt'])
	objPhot = hstack([apPhot.groups.keys,objPhot])
	# associate each object with a magnitude bin
	print '...bin by reference mag'
	objPhot['refMag'] = apPhot['refMag'][apPhot.groups.indices[:-1]]
	binNum = np.digitize(objPhot['refMag'],binEdges)
	objPhot['binNum'] = binNum - 1
	assert np.all((objPhot['binNum']>=0)&(objPhot['binNum']<len(mbins)))
	# regroup into magnitude bins
	objPhot = objPhot[objPhot['aperMag_n']>=minNobs]
	binPhot = objPhot.group_by(['season','filter','binNum'])
	binGroup = Table(binPhot.groups.keys)
	# average magnitude offsets
	meanVals = binPhot['dMagExt',].groups.aggregate(np.ma.median)
	binGroup['dMagExt'] = meanVals['dMagExt']
	# percentile distributions of per-object scatter in mag bins
	perc = lambda x,p: np.percentile(x.compressed(),p)
	rmskeys = ['aperMag_rms','dMagExt_rms']
	for p in pctiles:
		pc = binPhot[rmskeys].groups.aggregate(lambda x: perc(x,p))
		binGroup['sig%d'%(p)] = pc['aperMag_rms']
		binGroup['sigExt%d'%(p)] = pc['dMagExt_rms']
	# fill the bin statistics into a multidimensional array
	binGroup = binGroup.group_by(['season','filter'])
	binStats = Table(binGroup.groups.keys)
	binStats['dMagExt'] = np.zeros((1,len(mbins)),dtype=np.float32)
	for p in pctiles:
		binStats['sig%d'%p] = np.zeros((1,len(mbins)),dtype=np.float32)
		binStats['sigExt%d'%p] = np.zeros((1,len(mbins)),dtype=np.float32)
	for i in range(len(binStats)):
		jj = binGroup.groups[i]['binNum']
		binStats['dMagExt'][i,jj] = binGroup.groups[i]['dMagExt']
		for p in pctiles:
			binStats['sig%d'%p][i,jj] = binGroup.groups[i]['sig%d'%p]
			binStats['sigExt%d'%p][i,jj] = binGroup.groups[i]['sigExt%d'%p]
	binStats.meta['mbins'] = ','.join(['%.2f'%m for m in mbins])
	return binStats

