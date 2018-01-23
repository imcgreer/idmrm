#!/usr/bin/make

PYTHONPATH := ../bok:${PYTHONPATH}

CFHTRMPHOT := python cfhtrmphot.py

catalogs:
	$(CFHTRMPHOT) --catalogs $(XARGS)

phot_sdss:
	$(CFHTRMPHOT) --dophot --catalog sdss $(XARGS)

zeropoints_sdss:
	$(CFHTRMPHOT) --zeropoint --catalog sdss $(XARGS)

lightcurves_sdss:
	$(CFHTRMPHOT) --lightcurves --catalog sdss $(XARGS)

phot_rm:
	$(CFHTRMPHOT) --dophot --catalog sdssrm $(XARGS)

phot_allqso:
	$(CFHTRMPHOT) --dophot --catalog allqso $(XARGS)

lightcurves_rm:
	$(CFHTRMPHOT) --lightcurves --catalog sdssrm $(XARGS)

lightcurves_allqso:
	$(CFHTRMPHOT) --lightcurves --catalog allqso $(XARGS)

aggregate:
	$(CFHTRMPHOT) --aggregate --catalog sdss

stats:
	$(CFHTRMPHOT) --binnedstats --catalog sdss

