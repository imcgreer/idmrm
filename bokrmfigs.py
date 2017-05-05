#!/usr/bin/env python

import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from astropy.table import Table
from astropy.io import fits

def zp_strip_chart(kind='aper',byFrame=False):
	plt.figure(figsize=(14,4))
	plt.subplots_adjust(0.04,0.10,0.98,0.98)
	for pNum,filt in enumerate('g',start=1):
		ax = plt.subplot(1,1,pNum)
		zpDat = Table.read('zeropoints_%s.fits'%filt)
		utds,ii = np.unique(zpDat['utDate'],return_index=True)
		xval = zpDat['frameNum'] if byFrame else np.arange(len(zpDat))
		plt.plot(xval,zpDat[kind+'Zp'])
		for i,utd in zip(ii,utds):
			plt.axvline(xval[i],c='gray')
			plt.text(xval[i]+1,25.55,utd,
			         ha='left',rotation='vertical',size=9)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
		ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
		ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))
	plt.xlim(xval[0]-20,xval[-1]+20)
	plt.ylim(25.2,26.15)

def conditions_strip_charts(byFrame=False):
	from bokrmphot import match_to # move this
	metaDat = Table.read('bokrmMetaData.fits')
	obsDb = Table.read('config/sdssrm-bok2014.fits')
	ff = match_to(metaDat['frameId'],obsDb['frameIndex'])
	plt.figure(figsize=(14,7))
	plt.subplots_adjust(0.04,0.04,0.98,0.98)
	for pNum,filt in enumerate('g',start=1):
		utds,ii = np.unique(obsDb['utDate'][ff],return_index=True)
		xval = metaDat['frameId'] if byFrame else np.arange(len(metaDat))
		ax1 = plt.subplot(2,1,1)
		ax1.plot(xval,0.455*metaDat['fwhmPix'].mean(axis=-1))
		ax2 = plt.subplot(2,1,2,sharex=ax1)
		ax2.plot(xval,26.0-2.5*np.log10(metaDat['skyElPerSec']*0.455**-2))
		for ax,ypos in zip([ax1,ax2],[2.5,20.5]):
			ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
			ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
			ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
			for i,utd in zip(ii,utds):
				ax.axvline(xval[i],c='gray')
				ax.text(xval[i]+1,ypos,utd,
				        ha='left',rotation='vertical',size=9)
	ax1.set_xlim(xval[0]-20,xval[-1]+20)
	ax1.set_ylim(0,3.0)
	ax2.set_ylim(22.6,18.5)

def dump_lc(lc,rowNum,aperNum):
	for i in rowNum:
		print '%7d %12.5f %3s %8.3f %6.3f %5d' % \
		         (lc['frameIndex'][i],lc['mjd'][i],lc['filter'][i],
		          lc['aperMag'][i,aperNum],lc['aperMagErr'][i,aperNum],
		          lc['flags'][i,aperNum])

def plot_lightcurve(lcTab,targetNum,aperNum=1,refCat=None,
                    targetSource='RM',shownightly=False):
	ymin,ymax = 1e9,0
	plt.figure(figsize=(12,5))
	plt.subplots_adjust(0.05,0.05,0.97,0.94)
	ax,pnum = None,1
	if refCat is not None:
		_j = np.where(refCat['objId']==targetNum)[0][0]
	gkeys = lcTab.groups.keys
	if gkeys.colnames == ['objId']:
		j = np.where(lcTab.groups.keys['objId']==targetNum)[0][0]
		lc = lcTab.groups[j].group_by('filter')
	elif gkeys is None or gkeys.colnames == ['objId','filter']:
		if gkeys is None:
			lcTab = lcTab.group_by(['objId','filter'])
		jj = np.where(lcTab.groups.keys['objId']==targetNum)[0]
		lc = lcTab.groups[jj]
	else:
		raise ValueError
	for band,lc,clr in zip(lc.groups.keys['filter'],lc.groups,['C2','C1']):
		ax = plt.subplot(2,1,pnum,sharex=ax)
		jj = np.where(lc['aperMag'][:,aperNum] < 30)[0]
		if len(jj)==0:
			pnum += 1
			continue
		plt.errorbar(lc['mjd'][jj],
		             lc['aperMag'][jj,aperNum],
		             lc['aperMagErr'][jj,aperNum],
		             fmt='s',mec=clr,mfc='none',ms=3,capsize=3,ecolor=clr)
		if True:
			dump_lc(lc,jj,aperNum)
		kk = np.where(lc['flags'][:,aperNum] > 0)[0]
		plt.scatter(lc['mjd'][kk],lc['aperMag'][kk,aperNum],
		            marker='x',s=50,color='k',lw=1)
		if shownightly:
			nightly = Table.read('nightly_lcs_bokrm_g.fits')
			nightly = nightly.group_by('objId')
			_j = np.where(nightly.groups.keys['objId']==targetNum)[0][0]
			_lc = nightly.groups[j]
			plt.errorbar(_lc['mean_mjd'][:],
			             _lc['aperMag'][:,aperNum],
			             _lc['aperMagErr'][:,aperNum],
			             fmt='s',mfc='none',mew=2,ecolor=clr,ms=8)
		ymin = np.percentile(lc['aperMag'][jj,aperNum],5)
		ymax = np.percentile(lc['aperMag'][jj,aperNum],95)
		dy = max(0.05,5*np.median(lc['aperMagErr'][jj,aperNum]))
		plt.ylim(ymax+dy,ymin-dy)
		bi = 1 if clr=='g' else 3
		if band=='g':
			plt.title('%s-%d' % (targetSource,targetNum))
		if False:#targetSource=='RM':
			plt.axhline(target['PSFMAG'][targetNum,bi],color=clr)
			plt.scatter(target['DR_MJD'][targetNum],
			            target['PSFMAG'][targetNum,bi],
			            c=clr)
			if clr=='g':
				plt.title('rm%03d, z=%.2f' % 
				          (targetNum,target['ZFINAL'][targetNum]))
		if refCat is not None:
			plt.axhline(refCat[band][_j],c='gray')
		ax.xaxis.set_minor_locator(ticker.MultipleLocator(20))
		ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
		pnum += 1
	plt.xlim(min(lc['mjd'].min(),np.inf)-5,max(lc['mjd'].max(),0)+5)

