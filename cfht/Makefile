#!/usr/bin/make

PYTHONPATH := ../bok:${PYTHONPATH}

CFHTRMPHOT := python cfhtrmphot.py

catalogs:
	$(CFHTRMPHOT) --catalogs $(XARGS)

phot:
	$(CFHTRMPHOT) --dophot $(XARGS)

zeropoints_sdss:
	$(CFHTRMPHOT) --zeropoint --catalog sdss $(XARGS)

lightcurves_sdss:
	$(CFHTRMPHOT) --lightcurves --catalog sdss $(XARGS)

