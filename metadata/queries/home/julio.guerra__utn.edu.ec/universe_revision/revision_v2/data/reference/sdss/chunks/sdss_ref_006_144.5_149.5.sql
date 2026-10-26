SELECT
    p.objid,
    p.ra,
    p.dec,
    s.specobjid,
    s.class,
    s.z AS redshift
FROM PhotoPrimary AS p
JOIN SpecObj AS s
    ON p.objid = s.bestobjid
WHERE
    p.ra >= 144.500000
    AND p.ra < 149.500000
    AND p.dec BETWEEN -0.25 AND 60.25
    AND p.u BETWEEN 10 AND 30
    AND p.g BETWEEN 10 AND 30
    AND p.r BETWEEN 10 AND 30
    AND p.i BETWEEN 10 AND 30
    AND p.z BETWEEN 10 AND 30
    AND p.clean = 1
    AND s.zWarning = 0
    AND s.z > 0.01
    AND s.z < 5.0
    AND s.class IN ('GALAXY', 'QSO')
