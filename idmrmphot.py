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

def load_catalog(catFile,filt,expTime,refCat,instrCfg):
	imCat = Table.read(catFile)
	imCat = Table(imCat,masked=True)
	jj = imCat['objId'] # XXX
	# XXX hack for sextractor
	if instrCfg.name == 'cfht':
		imCat['flags'] = np.tile(imCat['flags'],(instrCfg.nAper,1)).T
	imCat['refMag'] = instrCfg.colorXform(refCat[filt][jj],
	                                   refCat['g'][jj]-refCat['i'][jj],filt)
	isMag = ( (imCat['refMag'] > instrCfg.magRange[filt][0]) &
	          (imCat['refMag'] < instrCfg.magRange[filt][1]) )
	mask = ( (imCat['counts'][:,instrCfg.aperNum] < 0) |
	         (imCat['countsErr'][:,instrCfg.aperNum] <= 0) |
	         (imCat['flags'][:,instrCfg.aperNum] > 0) |
	         ~isMag )
	counts = np.ma.array(imCat['counts'][:,instrCfg.aperNum],mask=mask)
	imCat['snr'] = np.ma.divide(counts,imCat['countsErr'][:,instrCfg.aperNum])
	counts.mask |= imCat['snr'] < 15
	mag = -2.5*np.ma.log10(counts/expTime)
	imCat['dmag'] = imCat['refMag'] - mag
	return imCat

def generate_zptab_entry(instrCfg):
	t = Table()
	for k,dt in [('aperZp','f4'),('aperZpRms','f4'),
	             ('aperNstar','i4'),('aperCorr','f4')]:
		s = (1,instrCfg.nCCD)
		if k == 'aperCorr':
			s = s + (instrCfg.nAper,)
		t[k] = np.zeros(s,dtype=dt)
	return t

def image_zeropoint(imCat,instrCfg):
	t = generate_zptab_entry(instrCfg)
	ccdObjs = imCat.group_by('ccdNum')
	for ccdNum,objs in zip(ccdObjs.groups.keys['ccdNum'],ccdObjs.groups):
		ccd = ccdNum - instrCfg.ccd0
		dMag = sigma_clip(objs['dmag'],sigma=2.2,iters=3)
		nstar = np.ma.sum(~dMag.mask)
		if nstar > instrCfg.minStar:
			zp,ivar = np.ma.average(dMag,weights=objs['snr']**2,returned=True)
			M = np.float((~dMag.mask).sum())
			wsd = np.sqrt(np.sum((dMag-zp)**2*objs['snr']**2) /
			                   ((M-1)/M*np.sum(objs['snr']**-2)))
			t['aperNstar'][0,ccd] = len(dMag.compressed())
			t['aperZp'][0,ccd] = zp
			t['aperZpRms'][0,ccd] = 1.0856 / wsd #* ivar**-0.5
			#
			ii = np.where(objs['snr'] > 30)[0]
			mask = ( (imCat['counts'] <= 0) |
			         (imCat['countsErr'] <= 0) |
			         (imCat['flags'] > 0) )
			counts = np.ma.array(objs['counts'][ii],mask=mask[ii])
			fratio = np.ma.divide(counts,counts[:,-1][:,np.newaxis])
			fratio = np.ma.masked_outside(fratio,0,1.5)
			fratio = sigma_clip(fratio,axis=0)
			invfratio = np.ma.power(fratio,-1)
			t['aperCorr'][0,ccd] = invfratio.mean(axis=0).filled(0)
	#
	return t

def calibrate_lightcurves(photTab,zpTab,dataMap,outFile,minNstar=1):
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
	apCorr = np.zeros((len(ii),nAper),dtype=np.float32)
	# cannot for the life of me figure out how to do this with indexing
	for apNum in range(nAper):
		apCorr[np.arange(len(ii)),apNum] = \
		            zpTab['aperCorr'][ii,ccdj,apNum]
	zp = np.ma.array(zpTab['aperZp'][ii],
	                 mask=zpTab['aperNstar'][ii]<minNstar)
	zp = zp[np.arange(len(ii)),ccdj][:,np.newaxis]
	corrCps = photTab['counts'] * apCorr 
	if not isbok:
		corrCps /= photTab['expTime'][:,np.newaxis]
	poscounts = np.ma.array(corrCps,mask=photTab['counts']<=0)
	magAB = zp - 2.5*np.ma.log10(poscounts)
	magErr = 1.0856*np.ma.divide(photTab['countsErr'],poscounts)
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

