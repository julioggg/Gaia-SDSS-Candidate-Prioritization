SELECT
        
    p.objid,
    p.htmid,
    p.ra,
    p.dec,
    p.u,
    p.g,
    p.r,
    p.i,
    p.z,
    p.err_u,
    p.err_g,
    p.err_r,
    p.err_i,
    p.err_z,
    p.clean,
    s.specobjid,
    s.class,
    s.z AS redshift,
    s.zErr AS redshift_error,
    s.zWarning

        FROM PhotoPrimary AS p
        JOIN SpecObj AS s
            ON p.objid = s.bestobjid
        WHERE
            p.ra >= 120.0
            AND p.ra < 125.0
            AND p.dec >= 5.0
            AND p.dec < 10.0
        
    AND p.ra IS NOT NULL
    AND p.dec IS NOT NULL
    AND p.u BETWEEN 10 AND 30
    AND p.g BETWEEN 10 AND 30
    AND p.r BETWEEN 10 AND 30
    AND p.i BETWEEN 10 AND 30
    AND p.z BETWEEN 10 AND 30
    AND p.clean = 1
    AND s.z IS NOT NULL
    AND s.zWarning = 0
    AND s.z > 0.01
    AND s.z < 5.0
    AND s.class IN ('GALAXY', 'QSO')
