# Web data

The scoring and export scripts write the JSON the web app reads. The web data loaders parse it with their Zod schemas, and the web copy of the prediction contract matches its Python owners.

Source: scripts/score_all_movies.py, scripts/export_web_data.py, scripts/generate_web_contract.py, web/src/lib/data.ts, web/src/lib/types.ts
Inventory-id: scripts/score_all_movies.py, scripts/export_web_data.py, scripts/generate_web_contract.py

## Sub-features

- `contract` regenerates `web/src/generated/predict-contract.json` and expects no change.
- `score-export` runs `score_all_movies.py` and `export_web_data.py` on the trained fixture.
- `zod-snapshot` runs `web_snapshot.test.ts` under vitest against the exported `web/data`.

## How to get to it (user POV)

- `make web-data` after a retrain.

## Driving it with verify

Preconditions: `web/node_modules` installed.

- **Drive.** `make verify ARGS=web-data`. The per-year table the export embeds comes from the run's own CV folds. The snapshot test checks that movies and predictions both hold one entry per fixture source row, and that every movie has a prediction.
- **Proof.** `web-data.*-generate-contract.json`, `-score-all.json`, `-export.json`, `-validate-snapshot.json`, and a manifest block.

## Gotchas

- `web/node_modules` is linked into the copy. The vitest config turns off the module cache so nothing is written through the link.
- The export needs `--per-year-table`; without it `per_year` is null and the loader schema rejects the file.
