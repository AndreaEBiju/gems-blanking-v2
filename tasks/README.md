# Task index

Generated. Gates in **bold**.

| # | task | depends on | gate |
|---|---|---|---|
| 00 | [repo-setup](00-repo-setup.md) | none |  |
| 00A | [drive-store](00A-drive-store.md) | 00 |  |
| 01 | [**nan-interop**](01-nan-interop.md) | 00 | YES |
| 02 | [**peri-r-measurement**](02-peri-r-measurement.md) | 00,01 | YES |
| 03 | [io-channel-map](03-io-channel-map.md) | 00 |  |
| 03A | [scan-conditions](03A-scan-conditions.md) | 03,00A |  |
| 03B | [stim-split](03B-stim-split.md) | 03,03A |  |
| 04 | [derivations](04-derivations.md) | 03 |  |
| 05 | [rpeaks](05-rpeaks.md) | 03 |  |
| 06 | [envelopes](06-envelopes.md) | 04,03B |  |
| 07 | [candidates](07-candidates.md) | 05,06,02 |  |
| 08 | [matlab-fixes](08-matlab-fixes.md) | 01 |  |
| 09 | [**recall-gate**](09-recall-gate.md) | 07 | YES |
| 10 | [label-conversion](10-label-conversion.md) | 07,09 |  |
| 11 | [features](11-features.md) | 07 |  |
| 12 | [classify](12-classify.md) | 10,11 |  |
| 12A | [model-registry](12A-model-registry.md) | 12 |  |
| 13 | [extent](13-extent.md) | 12 |  |
| 14 | [routing](14-routing.md) | 13 |  |
| 15 | [emit-qc](15-emit-qc.md) | 14 |  |
| 16 | [ui-recall-audit](16-ui-recall-audit.md) | 07 |  |
| 16A | [app-shell](16A-app-shell.md) | 16 |  |
| 16B | [results-dashboard](16B-results-dashboard.md) | 12,16A |  |
| 17 | [video](17-video.md) | 03 |  |
| 18 | [velocity](18-velocity.md) | 04 |  |
| 19 | [**acceptance**](19-acceptance.md) | all | YES |
