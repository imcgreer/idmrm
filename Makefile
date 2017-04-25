#!/usr/bin/make

.PHONY: initproc badpix flats proc redo

TOOLS := $(HOME)/dev/rapala/bokpipe/tools

ifdef LOGFILE
	LOGARGS := --obsdb $(LOGFILE)
endif

ifdef RAWDATA
	DATAARGS := -r $(RAWDATA)
endif

ifdef UTDATE
	UTARGS := -u $(UTDATE)
endif

ifdef NPROC
	MPARGS := -p $(NPROC)
else
	MPARGS := -p 7
endif

ifndef VERBOSE
	VERBOSE := -v
endif

ifdef BANDS
	BANDARGS := -b $(BANDS)
endif

DATAINITARGS := $(LOGARGS) $(DATAARGS) $(UTARGS) $(BANDARGS) 

INITARGS := $(DATAINITARGS) $(MPARGS) $(VERBOSE) 

BOKRMPIPE := python bokrmpipe.py
BOKRMPHOT := python bokrmphot.py

all_detrend: initproc badpix proc1 makeillum flats proc2 skysub

obsdb:
	$(BOKRMPIPE) --makeobsdb $(LOGARGS) -R \
	             -r $(HOME)/data/observing/Bok/90Prime/RM

# Overscan-subtract all images, generate 2D biases and dome flats
initproc:
	$(BOKRMPIPE) $(INITARGS) $(PROCARGS) -s oscan,bias2d,flat2d $(XARGS)

# Generate the bias ramp image (<=2015 data)
biasramp:
	$(BOKRMPIPE) $(INITARGS) -s ramp $(XARGS)

# XXX copy in a master bp mask from config dir
badpix:
	cp $(BOK90PRIMEOUTDIR)/bokpipe_v0.2/cals/BadPix* $(BOK90PRIMEOUTDIR)/bokpipe_v0.3/cals/

# First-round processing: bias/domeflat correction, combine into CCD extensions
proc1:
	$(BOKRMPIPE) $(INITARGS) $(PROCARGS) -s proc1 $(XARGS)

# Make the illumination correction image
makeillum:
	$(BOKRMPIPE) $(INITARGS) -s illum $(XARGS)

#
# Sky flat generation (processing output to temp directory)
#

#  ... apply the illumination correction to the sky flat images
flats_illumcorr:
	$(BOKRMPIPE) $(INITARGS) -s proc2 --prockey TMPPRO2 \
	             --nofringecorr --noskyflatcorr \
	             --noweightmap --nodivideexptime \
	             --skyflatframes --tmpdirout \
	             $(XARGS)

#  ... make fringe masters from sky flat images
flats_makefringe:
	$(BOKRMPIPE) $(INITARGS) -s fringe \
	             $(SKYFLATARGS) --skyflatframes --tmpdirin --tmpdirout \
	             $(XARGS)

# ... apply fringe correction to sky flat images
flats_fringecorr:
	$(BOKRMPIPE) $(INITARGS) \
	             -s proc2 --prockey TMPPRO3 \
	             --noillumcorr --noskyflatcorr \
	             --noweightmap --nodivideexptime \
	             --skyflatframes --tmpdirin --tmpdirout \
	             $(XARGS)

# ... apply sky subtraction to sky flat images
flats_skysub:
	$(BOKRMPIPE) $(INITARGS) \
	             -s skysub \
	             --skymethod polynomial --skyorder 1 \
	             --skyflatframes --tmpdirin --tmpdirout \
                 --flatcorrectbeforeskymask \
	             $(XARGS)

# ... combine temp processed images to make sky flat
flats_makeskyflat:
	$(BOKRMPIPE) $(INITARGS) -s skyflat \
	             $(SKYFLATARGS) --skyflatframes --tmpdirin --tmpdirout \
	             $(XARGS)

# all the steps to generate sky flats in one target
flats: flats_illumcorr \
       flats_makefringe flats_fringecorr flats_skysub \
       flats_makeskyflat

flats_nosky: makeillum flats_illumcorr flats_makefringe

# Second-round processing: apply illumination, skyflat, and fringe corrections
#  and do sky subtraction
proc2:
	$(BOKRMPIPE) $(INITARGS) $(PROCARGS) -s proc2,skysub $(XARGS)

# Perform individual processing steps as listed in STEPS
steps:
	$(BOKRMPIPE) $(INITARGS) $(PROCARGS) -s $(STEPS) $(XARGS)

# Assuming cals already exist, perfrom all the processing steps on science ims
procall:
	$(BOKRMPIPE) $(INITARGS) $(PROCARGS) \
	             -s oscan,proc1,proc2,skysub -t object $(XARGS)


# Obtain astrometric solutions
wcs:
	$(BOKRMPIPE) $(INITARGS) -s wcs --gaia $(XARGS)


# Generate object catalogs and PSF models with sextractor+psfex
catalogs:
	$(BOKRMPIPE) $(INITARGS) -s cat $(XARGS)


# Generate aperture photometry catalogs for RM targets
aperphot_rm:
	$(BOKRMPHOT) $(INITARGS) --aperphot $(XARGS)

# Generate aperture photometry catalogs for SDSS reference stars
aperphot_sdss:
	$(BOKRMPHOT) $(INITARGS) --aperphot $(XARGS) --catalog sdss

# Generate aperture photometry catalogs for CFHT reference stars
aperphot_cfht:
	$(BOKRMPHOT) $(INITARGS) --aperphot $(XARGS) --catalog cfht

aperphot: aperphot_sdss aperphot_cfht aperphot_rm

zeropoints_sdss:
	$(BOKRMPHOT) $(INITARGS) --zeropoint $(XARGS) --catalog sdss

# Generate lightcurve tables for RM targets
lightcurves_rm:
	$(BOKRMPHOT) $(INITARGS) --lightcurves $(XARGS)

# Generate lightcurve tables for SDSS reference stars
lightcurves_sdss:
	$(BOKRMPHOT) $(INITARGS) --lightcurves $(XARGS) --catalog sdss

# Generate lightcurve tables for CFHT reference stars
lightcurves_cfht:
	$(BOKRMPHOT) $(INITARGS) --lightcurves $(XARGS) --catalog cfht

all_sdssref: aperphot_sdss zeropoints_sdss lightcurves_sdss

metadata:
	python bokrmgnostic.py $(LOGARGS) $(UTARGS) $(BANDARGS) --metadata $(XARGS)


# Redo all processing, WCS, and catalogs from scratch
redo:
	make XARGS="-R $(XARGS)" procall wcs catalogs aperphot

rebuild_caldb: makeillum flats_makefringe flats_makeskyflat

compress:
	$(BOKRMPIPE) $(DATAINITARGS) --compress $(XARGS)

cleancals:
	$(BOKRMPIPE) $(DATAINITARGS) --cleancals $(XARGS)

#
# Diagnostic stuff
#

# Make PNG images for inspection purposes
images:
	$(BOKRMPIPE) $(INITARGS) --images $(XARGS)

# Tabulate the rms of the astrometric solutions
checkwcs:
	$(BOKRMPIPE) $(INITARGS) --wcscheck $(XARGS)

# Check status of image processing
checkproc:
	python bokrmgnostic.py $(LOGARGS) $(UTARGS) $(BANDARGS) --checkproc $(XARGS)

