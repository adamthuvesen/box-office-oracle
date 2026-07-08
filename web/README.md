# Box Office Oracle — Web

The screening room: a local Next.js app that puts a face on the ML pipeline.
Every visualization follows one dichotomy — **warm amber light is reality**
(actual grosses), **cool cyan light is the machine** (predictions).

## Pages

- `/` — the Constellation: ~6,100 movies (1980–2026) as WebGL particles in a
  budget×gross field, walked through a five-scene scroll narrative
- `/movies` — poster explorer (search, genre/decade filters, grid ⇄ table)
- `/movies/[id]` — title card, facts, actual vs the model's blind guess
- `/stats` — box-office statistics (seasonality film strip, genre economics,
  budget vs gross, hit factory)
- `/model` — the backtest report card (model vs baseline, blind guesses vs
  reality, live model info)
- `/predict` — the Oracle: form → Lambda `/predict` → animated reveal

## Run

```bash
# from the repo root — export gitignored data snapshots to web/data/
# (reads the local 1980-2026 parquet; no Snowflake, no TMDB API)
make web-data

pnpm install
pnpm dev          # http://localhost:3000
pnpm build        # type-check + production build
```

Live predictions need `INFERENCE_API_URL` + `INFERENCE_API_KEY` in
`.env.local` (see `.env.example`); without them `/api/predict` answers in
mock mode, clearly labeled.

## Stack

Next.js 16 (App Router, Cache Components/PPR, React Compiler), React 19.2,
Tailwind v4 (tokens in `src/app/globals.css` `@theme`), shadcn/ui on Base UI,
Recharts 3 (stats/model charts), ogl (constellation WebGL), Motion 12
(oracle reveal), TanStack Table/Query, zod-validated data loaders.
