#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord,match_coordinates_sky
from astropy import units as u
from astropy.table import Table,join

from bokpipe import bokphot,bokpl,bokgnostic
from bokpipe.bokproc import ampOrder,BokImStat
import bokrmpipe
import bokrmphot

def plot_gain_vals(diagfile):
	g = np.load(diagfile)#,gains=gainCorV,skys=skyV,gainCor=gainCor)
	plt.figure(figsize=(12,8))
	plt.subplots_adjust(0.07,0.04,0.97,0.97,0.25,0.05)
	for amp in range(16):
		ax = plt.subplot(4,4,amp+1)
		ax.plot(g['gains'][:,0,amp],c='b')
		ax.axhline(g['gainCor'][0,amp],c='purple',ls='--')
		ax.plot(g['gains'][:,1,amp],c='r')
		ax.axhline(g['gainCor'][1,amp],c='orange',ls='--')
		ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
		ax.xaxis.set_visible(False)
		ax.text(0.05,0.99,'IM%d'%ampOrder[amp],
		        size=8,va='top',transform=ax.transAxes)
		ax.text(0.25,0.99,'%.3f'%g['gainCor'][0,amp],color='blue',
		        size=8,va='top',transform=ax.transAxes)
		ax.text(0.50,0.99,'%.3f'%g['gainCor'][1,amp],color='red',
		        size=8,va='top',transform=ax.transAxes)
		ax.set_xlim(-1,g['gains'].shape[0]+1)
		ax.set_ylim(0.96,1.04)

def srcor(ra1,dec1,ra2,dec2,sep):
	c1 = SkyCoord(ra1,dec1,unit=(u.degree,u.degree))
	c2 = SkyCoord(ra2,dec2,unit=(u.degree,u.degree))
	idx,d2d,d3c = match_coordinates_sky(c1,c2)
	ii = np.where(d2d.arcsec < sep)[0]
	return ii,idx[ii],d2d.arcsec[ii]

def calc_sky_backgrounds(dataMap,outputFile):
	extns = ['IM4']
	imstat = BokImStat(extensions=extns,quickprocess=True,
	                   stats_region='amp_central_quadrant')
	skyvals = []
	for utd in dataMap.iterUtDates():
		files,ii = dataMap.getFiles(imType='object',with_frames=True)
		if files is not None:
			rawfiles = [ dataMap('raw')(f) for f in files ]
			imstat.process_files(rawfiles)
			sky = imstat.meanVals.squeeze()
			print '%8s %4d %10.1f %10.1f %10.1f' % \
			      (utd,len(files),sky.mean(),sky.min(),sky.max())
			skyvals.extend([ (utd,f.split('/')[1],s) 
			                    for f,s in zip(files,sky) ])
		imstat.reset()
	tab = Table(rows=skyvals,names=('utDate','fileName','skyMean'))
	tab.write(outputFile)

def id_sky_frames(obsDb,skytab,utds,thresh=10000.):
	frametab = obsDb['frameIndex','utDate','fileName','objName'].copy()
	ii = np.where(np.in1d(frametab['utDate'],utds))[0]
	skytab = join(skytab,frametab[ii],'fileName')
	assert np.all(skytab['utDate_1']==skytab['utDate_2'])
	del skytab['utDate_2']
	skytab.rename_column('utDate_1','utDate')
	# first cut the repeated pointings
	ii = np.where((skytab['objName'][:-1]==skytab['objName'][1:]) & 
	              (skytab['utDate'][:-1]==skytab['utDate'][1:]))[0]
	skytab.remove_rows(1+ii)
	# then cut on the sky threshold
	ii = np.where(skytab['skyMean'] < thresh)[0]
	skytab = skytab[ii]
	return skytab

def id_sky_frames_2014():
	obsDb = bokpl._load_obsdb('config/sdssrm-bok2014.fits.gz')
	darkNights = {
	  'g':['20140126','20140128','20140312','20140424','20140426',
	       '20140427','20140518','20140630','20140702','20140718'],
	  'i':['20140123','20140126','20140129','20140425','20140428',
	       '20140701','20140717'],
	}
	for filt in 'gi':
		skytab = Table.read('data/bokrm2014sky%s.fits.gz'%filt)
		skyframes = id_sky_frames(obsDb,skytab,darkNights[filt],
		                          thresh={'g':1500.,'i':5000.}[filt])
		outf = 'config/bokrm2014_darksky_%s.txt' % filt
		skyframes['skyMean'].format = '{:8.1f}'
		skyframes['utDate','fileName','skyMean'].write(outf,format='ascii')

def check_img_astrom(imgFile,refCat,catFile=None,mlim=19.5,band='g'):
	imFits = fits.open(imgFile)
	if catFile is None:
		catFile = imgFile.replace('.fits','.cat.fits')
	catFits = fits.open(catFile)
	try:
		ahead = bokastrom.read_headers(imgFile.replace('.fits','.ahead'))
	except:
		ahead = None
	rv = []
	for ccd in range(1,5):
		ccdCat = catFits[ccd].data
		hdr = imFits[ccd].header
		if ahead is not None:
			hdr.update(ahead[ccd-1].items())
		w = WCS(hdr)
		foot = w.calc_footprint()
		ras = sorted(foot[:,0])
		decs = sorted(foot[:,1])
		ii = np.where((refCat['ra']>ras[1])&(refCat['ra']<ras[2]) &
		              (refCat['dec']>decs[1])&(refCat['dec']<decs[2]) &
		              (refCat[band]<mlim))[0]
		m1,m2,sep = srcor(ccdCat['ALPHA_J2000'],ccdCat['DELTA_J2000'],
		                  refCat['ra'][ii],refCat['dec'][ii],5.0)
		rv.append(dict(N=len(ii),nMatch=len(ii),
		               ra=ccdCat['ALPHA_J2000'][m1],
		               dec=ccdCat['DELTA_J2000'][m1],
		               raRef=refCat['ra'][ii[m2]],
		               decRef=refCat['dec'][ii[m2]],
		               sep=sep))
	return rv

def rmobs_meta_data(dataMap):
	bokgnostic.obs_meta_data(dataMap,outFile='bokrmMetaData.fits')

def dump_data_summary(dataMap,splitrm=False):
	for utd in dataMap.iterUtDates():
		files,ii = dataMap.getFiles(with_frames=True)
		ntotal = len(ii)
		nbiases = np.sum(dataMap.obsDb['imType'][ii] == 'zero')
		missing = []
		if nbiases<=3: missing.append('nobias')
		print '%8s %7d %7d ' % (utd,ntotal,nbiases),
		nflt = {}
		for filt in 'gi':
			nflats = np.sum( (dataMap.obsDb['imType'][ii] == 'flat') &
			                 (dataMap.obsDb['filter'][ii] == filt) )
			print '%7d ' % nflats,
			nflt[filt] = nflats
		isrm = np.array([n.startswith('rm') 
		                    for n in dataMap.obsDb['objName'][ii]])
		for filt in 'gi':
			nsci = np.sum( (dataMap.obsDb['imType'][ii] == 'object') &
			               (dataMap.obsDb['filter'][ii] == filt) )
			nrm = np.sum( (dataMap.obsDb['imType'][ii] == 'object') &
			              (dataMap.obsDb['filter'][ii] == filt) & isrm )
			if splitrm:
				print '%7d %7d' % (nsci,nrm),
			else:
				print '%7d ' % (nrm),
			if nrm>0 and nflt[filt]<=3: missing.append('no%sflat'%filt)
		print '  ',','.join(missing)

def check_processed_data(dataMap):
	import fitsio
	sdss = fits.getdata(os.environ['BOK90PRIMEDIR']+'/../data/sdss.fits',1)
	zeropoints = fits.getdata('zeropoints_g.fits')
	tabf = open(os.path.join('proc_diag.html'),'w')
	tabf.write(bokgnostic.html_diag_head)
	rowstr = ''
	files_and_frames = dataMap.getFiles(with_frames=True)
	for f,i in zip(*files_and_frames):
		frameId = dataMap.obsDb['frameIndex'][i]
		rowstr = ''
		procf = dataMap('proc2')(f)
		rowstr += bokgnostic.html_table_entry('%d'%frameId,'nominal')
		rowstr += bokgnostic.html_table_entry(f,'nominal')
		print procf
		try:
			hdr0 = fitsio.read_header(procf,ext=0)
			for k in ['OSCNSUB','CCDPROC','CCDJOIN','CCDPRO2','SKYSUB']:
				if k in hdr0:
					rowstr += bokgnostic.html_table_entry(r'&#10004;',
					                                      'nominal')
				else:
					rowstr += bokgnostic.html_table_entry(r'&#9747;',
					                                      'bad')
				status = 'nominal' if k in hdr0 else 'missing'
		except:
			print procf,' does not exist'
			for k in ['OSCNSUB','CCDPROC','CCDJOIN','CCDPRO2','SKYSUB']:
				rowstr += bokgnostic.html_table_entry('','missing')
		try:
			zpi = np.where(dataMap.obsDb['frameIndex'][i] ==
			                                  zeropoints['frameId'])[0][0]
		except IndexError:
			zpi = None
		for ccdi in range(4):
			if zpi is not None:
				zp = zeropoints['aperZp'][zpi,ccdi]
				if zp > 25.90:
					status = 'nominal'
				elif zp > 25.70:
					status = 'warning'
				elif zp > 25.40:
					status = 'bad'
				else:
					status = 'weird'
			else:
				zp = 0.0
				status = 'missing'
			rowstr += bokgnostic.html_table_entry('%.2f'%zp,status)
		catf = dataMap('cat')(f)
		try:
			m = check_img_astrom(procf,sdss,catFile=catf)
			for c in m:
				sep = np.median(c['sep'])
				if sep > 0.4:
					status = 'bad'
				elif sep > 0.2:
					status = 'warning'
				elif sep > 0.0:
					status = 'nominal'
				else:
					status = 'weird'
				rowstr += bokgnostic.html_table_entry('%.3f'%sep,status)
		except IOError:
			for i in range(4):
				rowstr += bokgnostic.html_table_entry('','missing')
		tabf.write(r'<tr>'+rowstr+r'</tr>'+'\n')
		tabf.flush()
	tabf.write(bokgnostic.html_diag_foot)
	tabf.close()

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser = bokpl.init_file_args(parser)
	parser.add_argument('--catalog',type=str,default='sdssrm',
	                help='reference catalog ([sdssrm]|sdss|cfht)')
	parser.add_argument('--metadata',action='store_true',
	                help='construct observations meta data table')
	parser.add_argument('--datasum',action='store_true',
	                help='output summary of available data')
	parser.add_argument('--checkproc',action='store_true',
	                help='check processing status of individual files')
	parser.add_argument('--calcsky',type=str,
	                help='calculate sky backgrounds')
	args = parser.parse_args()
	args = bokrmpipe.set_rm_defaults(args)
	dataMap = bokpl.init_data_map(args)
	dataMap = bokpl.set_master_cals(dataMap)
	refCat = bokrmphot.load_catalog(args.catalog)
	if args.datasum:
		dump_data_summary(dataMap)
	elif args.checkproc:
		check_processed_data(dataMap)
	elif args.calcsky:
		calc_sky_backgrounds(dataMap,args.calcsky)
	elif args.metadata:
		rmobs_meta_data(dataMap)

