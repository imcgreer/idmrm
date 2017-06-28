#!/usr/bin/make

PYTHONPATH := ../bok:${PYTHONPATH}

CFHTRMPHOT := python cfhtrmphot.py

catalogs:
	$(CFHTRMPHOT) --catalogs $(XARGS)

phot:
	$(CFHTRMPHOT) --dophot $(XARGS)

zeropoints:
	$(CFHTRMPHOT) --zeropoint --catalog sdss $(XARGS)

