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

from bokpipe import bokphot,bokpl,bokproc,bokutil
import bokrmpipe
import cfhtrm

nom_pixscl = 0.18555

cfhtrm_aperRad = np.array([0.75,1.5,2.275,3.4,4.55,6.67,10.]) / nom_pixscl

def _cat_worker(dataMap,imFile,**kwargs):
	bokutil.mplog('extracting catalogs for '+imFile)
	import pdb; pdb.set_trace()
	imgFile = dataMap('img')(imFile)
	psfFile = dataMap('psf')(imFile)
	tmpFile = imgFile.replace('.fz','')
	if not os.path.exists(psfFile):
		catFile = dataMap('wcscat')(imFile)
		print '-->',imgFile
		subprocess.call(['funpack',imgFile])
		bokphot.sextract(tmpFile,catFile,full=False,**kwargs)
		bokphot.run_psfex(catFile,psfFile,instrument='cfhtmegacam',**kwargs)
		if False:
			os.remove(catFile)
	catFile = dataMap('cat')(imFile)
	if not os.path.exists(catFile):
		if not os.path.exists(tmpFile):
			subprocess.call('funpack',imgFile)
		bokphot.sextract(tmpFile,catFile,psfFile,full=True,**kwargs)
	if os.path.exists(tmpFile):
		os.remove(tmpFile)

def make_sextractor_catalogs(dataMap,**kwargs):
	procmap = map # XXX
	apers = ','.join(['%.2f'%a for a in cfhtrm_aperRad])
	kwargs.setdefault('DETECT_MINAREA','10.0')
	kwargs.setdefault('DETECT_THRESH','2.0')
	kwargs.setdefault('ANALYSIS_THRESH','2.0')
	kwargs.setdefault('PHOT_APERTURES',apers)
	kwargs.setdefault('SEEING_FWHM','1.0')
	kwargs.setdefault('PIXEL_SCALE','0.18555')
	kwargs.setdefault('SATUR_KEY','SATURATE')
	kwargs.setdefault('GAIN_KEY','GAIN')
	files = dataMap.getFiles()
	files = [files[0]]
	print files
	p_cat_worker = partial(_cat_worker,dataMap,**kwargs)
	status = procmap(p_cat_worker,files)

if __name__=='__main__':
	import sys
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--catalogs',action='store_true',
	                help='make source extractor catalogs and PSF models')
	args = parser.parse_args()
	#
	dataMap = cfhtrm.CfhtDataMap()
	if args.catalogs:
		make_sextractor_catalogs(dataMap)

