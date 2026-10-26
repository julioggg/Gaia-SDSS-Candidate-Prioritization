SELECT
    p.objid,
    p.ra,
    p.dec,
    s.specobjid,
    s.class,
    s.subClass,
    s.survey,
    s.programname,
    s.sourceType,
    s.z,
    s.zErr,
    s.zWarning,
    s.class_noqso,
    s.subClass_noqso,
    s.z_noqso,
    s.zWarning_noqso,
    s.class_person,
    s.z_person,
    s.snMedian,
    s.rChi2,
    s.rChi2Diff,
    s.velDisp,
    s.velDispErr
FROM PhotoPrimary AS p
JOIN SpecObj AS s
    ON p.objid = s.bestobjid
WHERE
    p.ra >= 230.000000
    AND p.ra < 235.000000
    AND p.dec BETWEEN 0.0 AND 60.0
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
