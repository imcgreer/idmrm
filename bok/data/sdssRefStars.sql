-- define the set of flux and astrometric calibration stars from SDSS
SELECT
   row_number() OVER(ORDER BY objId) - 1 AS objId,
   objId as sdssObjId,
   ra,dec,
   psfMag_g AS g,
   psfMag_r AS r,
   psfMag_i AS i,
   psfMagErr_g AS err_g,
   psfMagErr_r AS err_r,
   psfmagErr_i AS err_i,
   flags_g,flags_r,flags_i
FROM Star
WHERE
   ra BETWEEN 211.1 AND 216.3
   AND dec BETWEEN 51.5 AND 54.7
   AND ( (psfMag_g BETWEEN 17 AND 22.0) OR
         (psfMag_i BETWEEN 16 AND 21.0) )
   -- BINNED1
   AND ((flags_r & 0x10000000) != 0)
   -- NOPROFILE|PEAKCENTER|NOTCHECKED|PSF_FLUX_INTERP|SATURATED|BAD_COUNTS_ERROR|EDGE
   AND ((flags_r & 0x8100000c00a4) = 0)
   -- DEBLEND_NOPEAK
   AND (((flags_r & 0x400000000000) = 0) or (psfmagerr_r <= 0.2))
   -- INTERP_CENTER
   AND (((flags_r & 0x100000000000) = 0) or (flags_r & 0x1000) = 0)
