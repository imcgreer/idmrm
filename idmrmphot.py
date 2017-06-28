#!/usr/bin/env python

import os
import numpy as np
import multiprocessing
from functools import partial
from astropy.io import fits
from astropy.table import Table,vstack,hstack,join
from astropy.stats import sigma_clip
from astropy.wcs import InconsistentAxisTypesError

from bokpipe import bokphot,bokpl,bokproc,bokutil
import bokrmpipe

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

