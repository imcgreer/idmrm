#!/usr/bin/env python

import os
import numpy as np
import multiprocessing
from functools import partial
from astropy.io import fits
from astropy.table import Table,vstack,hstack,join
from astropy.stats import sigma_clip
from astropy.wcs import InconsistentAxisTypesError

##############################################################################
#
# utilities for joining indexed tables (more efficient than table.join())
#
##############################################################################

def map_ids(ids1,ids2):
	ii = -np.ones(ids2.max()+1,dtype=int)
	ii[ids2] = np.arange(len(ids2))
	return ii[ids1]

def join_by_id(tab1,tab2,idkey):
	ii = map_ids(tab1[idkey],tab2[idkey])
	# avoid duplication
	tab2 = tab2.copy()
	for c in tab1.colnames:
		if c in tab2.colnames:
			del tab2[c]
	return hstack([tab1,tab2[ii]])

def join_by_frameid(tab1,tab2):
	return join_by_id(tab1,tab2,'frameIndex')

def join_by_objid(tab1,tab2):
	return join_by_id(tab1,tab2,'objId')

##############################################################################
#
# zero points
#
##############################################################################

def srcor(ra1,dec1,ra2,dec2,sep):
	from astropy.coordinates import SkyCoord,match_coordinates_sky
	from astropy import units as u
	c1 = SkyCoord(ra1,dec1,unit=(u.degree,u.degree))
	c2 = SkyCoord(ra2,dec2,unit=(u.degree,u.degree))
	idx,d2d,d3c = match_coordinates_sky(c1,c2)
	ii = np.where(d2d.arcsec < sep)[0]
	return ii,idx[ii],d2d.arcsec[ii]

# XXX for bok do by table groups [frameIndex] since catalogs are nightly

def load_catalog(catFile,filt,refCat,instrCfg,verbose=False):
	imCat = Table.read(catFile)
	imCat = Table(imCat,masked=True)
	jj = imCat['objId'] # XXX
	imCat['refMag'] = instrCfg.colorXform(refCat[filt][jj],
	                                   refCat['g'][jj]-refCat['i'][jj],filt)
	isMag = ( (imCat['refMag'] > instrCfg.magRange[filt][0]) &
	          (imCat['refMag'] < instrCfg.magRange[filt][1]) )
	mask = ( (imCat['counts'][:,instrCfg.aperNum] < 0) |
	         (imCat['countsErr'][:,instrCfg.aperNum] <= 0) |
	         (imCat['flags'][:,instrCfg.aperNum] > 0) |
	         ~isMag )
	if verbose:
		print '%d total objs, %d in magRange, %d selected' % \
		          (len(imCat),isMag.sum(),(~mask).sum())
	counts = np.ma.array(imCat['counts'][:,instrCfg.aperNum],mask=mask)
	imCat['snr'] = np.ma.divide(counts,imCat['countsErr'][:,instrCfg.aperNum])
	counts.mask |= imCat['snr'] < 15
	mag = -2.5*np.ma.log10(counts)
	imCat['dmag'] = imCat['refMag'] - mag
	return imCat

def generate_zptab_entry(instrCfg,byccd=False):
	t = Table()
	for k,dt in [('aperZp','f4'),('aperZpRms','f4'),
	             ('aperNstar','i4'),('aperCorr','f4')]:
		s = (1,)
		if byccd:
			s += (instrCfg.nCCD,)
		if k == 'aperCorr':
			s += (instrCfg.nAper,)
		t[k] = np.zeros(s,dtype=dt)
	return t

def _calc_zeropoint(objs,instrCfg):
	apNum = instrCfg.aperNum
	dMag = sigma_clip(objs['dmag'],sigma=2.2,iters=3)
	nstar = np.ma.sum(~dMag.mask)
	zp,zprms,aperCorr = 0.0,0.0,0.0
	if nstar > instrCfg.minStar:
		#
		zp,ivar = np.ma.average(dMag,weights=objs['snr']**2,returned=True)
		M = np.float((~dMag.mask).sum())
		wsd = np.sqrt(np.sum((dMag-zp)**2*objs['snr']**2) /
		                   ((M-1)/M*np.sum(objs['snr']**-2)))
		nstar = len(dMag.compressed())
		zprms = 1.0856 / wsd #* ivar**-0.5
		#
		mask = dMag.mask | (objs['snr'].filled(0) < 50)
		counts = np.ma.array(objs['counts'])
		counts[mask,:] = np.ma.masked
		fratio = np.ma.divide(counts,counts[:,apNum][:,np.newaxis])
		fratio = np.ma.masked_outside(fratio,0,1.5)
		fratio = sigma_clip(fratio,axis=0)
		invfratio = np.ma.power(fratio,-1)
		aperCorr = invfratio.mean(axis=0).filled(0)
	return nstar,zp,zprms,aperCorr

def zeropoint_focalplane(imCat,instrCfg,verbose=0):
	t = generate_zptab_entry(instrCfg)
	nstar,zp,zprms,aperCorr = _calc_zeropoint(imCat,instrCfg)
	t['aperNstar'][:] = nstar
	t['aperZp'][:] = zp
	t['aperZpRms'][:] = zprms
	t['aperCorr'][:] = aperCorr
	return t

def zeropoint_byccd(imCat,instrCfg,verbose=0):
	t = generate_zptab_entry(instrCfg,True)
	ccdObjs = imCat.group_by('ccdNum')
	for ccdNum,objs in zip(ccdObjs.groups.keys['ccdNum'],ccdObjs.groups):
		ccd = ccdNum - instrCfg.ccd0
		nstar,zp,zprms,aperCorr = _calc_zeropoint(objs,instrCfg)
		t['aperNstar'][0,ccd] = nstar
		t['aperZp'][0,ccd] = zp
		t['aperZpRms'][0,ccd] = zprms
		t['aperCorr'][0,ccd] = aperCorr
	return t

def image_zeropoint(imCat,instrCfg,verbose=0):
	t = generate_zptab_entry(instrCfg)
	ccdObjs = imCat.group_by('ccdNum')
	apNum = instrCfg.aperNum
	for ccdNum,objs in zip(ccdObjs.groups.keys['ccdNum'],ccdObjs.groups):
		ccd = ccdNum - instrCfg.ccd0
		dMag = sigma_clip(objs['dmag'],sigma=2.2,iters=3)
		nstar = np.ma.sum(~dMag.mask)
		if verbose > 1:
			print 'have %d/%d stars on ccd%d' % (nstar,len(dMag),ccdNum)
		if nstar > instrCfg.minStar:
			zp,ivar = np.ma.average(dMag,weights=objs['snr']**2,returned=True)
			M = np.float((~dMag.mask).sum())
			wsd = np.sqrt(np.sum((dMag-zp)**2*objs['snr']**2) /
			                   ((M-1)/M*np.sum(objs['snr']**-2)))
			t['aperNstar'][0,ccd] = len(dMag.compressed())
			t['aperZp'][0,ccd] = zp
			t['aperZpRms'][0,ccd] = 1.0856 / wsd #* ivar**-0.5
			#
#			ii = np.where(objs['snr'] > 30)[0]
#			mask = ( (imCat['counts'] <= 0) |
#			         (imCat['countsErr'] <= 0) |
#			         (imCat['flags'] > 0) )
			mask = dMag.mask | (objs['snr'] < 30)
			counts = np.ma.array(objs['counts'],mask=mask)
			fratio = np.ma.divide(counts,counts[:,apNum][:,np.newaxis])
			fratio = np.ma.masked_outside(fratio,0,1.5)
			fratio = sigma_clip(fratio,axis=0)
			invfratio = np.ma.power(fratio,-1)
			t['aperCorr'][0,ccd] = invfratio.mean(axis=0).filled(0)
	#
	return t

def calibrate_lightcurves(photTab,zpTab,dataMap,outFile,
                          zptmode='focalplane',apermode='focalplane',
                          fill=True,minNstar=1):
	isbok = False
	if isbok:
		ccdj = photTab['ccdNum'] - 1
	else:
		ccdj = photTab['ccdNum']
	ii = map_ids(photTab['frameIndex'],zpTab['frameIndex'])
	print '--> ',len(photTab),len(ii)
	#
	obsdat = dataMap.obsDb['frameIndex','airmass','mjdMid','expTime'].copy()
	obsdat.rename_column('mjdMid','mjd')
	photTab = join_by_frameid(photTab,obsdat)
	#
	nAper = photTab['counts'].shape[-1]
	apCorr = np.ma.array(zpTab['aperCorr'][ii],
	                     mask=zpTab['aperCorr'][ii]==0)
	if apermode == 'focalplane':
		pass #apCorr = apCorr.mean(axis=1)
	elif apermode == 'ccd':
		raise NotImplementedError # started here in case it is needed
		if fill:
			_apCorr = apCorr.mean(axis=1)
			apCorr[apCorr.mask] = _apCorr[apCorr.mask]
		apCorr = np.zeros((len(ii),nAper),dtype=np.float32)
		# cannot for the life of me figure out how to do this with indexing
		for apNum in range(nAper):
			apCorr[np.arange(len(ii)),apNum] = \
			            zpTab['aperCorr'][ii,ccdj,apNum]
	else:
		raise ValueError
	#
	zp = np.ma.array(zpTab['aperZp'][ii],
	                 mask=zpTab['aperNstar'][ii]<minNstar)
	if zptmode == 'focalplane':
		pass #zp = zp.mean(axis=1)
	elif zptmode == 'ccd':
		zp = zp[np.arange(len(ii)),ccdj]
	else:
		raise ValueError
	zp = zp[:,np.newaxis]
	#
	corrCps = photTab['counts'] * apCorr 
	poscounts = np.ma.array(corrCps,mask=photTab['counts']<=0)
	magAB = zp - 2.5*np.ma.log10(poscounts)
	magErr = 1.0856*np.ma.divide(photTab['countsErr'],
	                             np.ma.array(photTab['counts'],
	                                         mask=poscounts.mask))
	photTab['aperMag'] = magAB.filled(99.99)
	photTab['aperMagErr'] = magErr.filled(0)
	# convert AB mag to nanomaggie
	fluxConv = 10**(-0.4*(zp-22.5))
	flux = corrCps * fluxConv
	fluxErr = photTab['countsErr'] * apCorr * fluxConv
	photTab['aperFlux'] = flux.filled(0)
	photTab['aperFluxErr'] = fluxErr.filled(0)
	print 'writing to ',outFile
	photTab.write(outFile,overwrite=True)

