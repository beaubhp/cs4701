# Cornell Policy RAG Results Explorer

Static Next.js dashboard for exploring the reviewed benchmark results.

## Local Run

```bash
npm install
npm run build:data
npm run dev
```

Open `http://localhost:3000`.

## Validate

```bash
npm run lint
npm run typecheck
npm run build
```

## Refresh Data

Run this after updating files under `../data/benchmark` or `../data/results`:

```bash
npm run build:data
```

This rewrites `src/data/dashboard-data.json`.

## Deploy On Vercel

Use `web/` as the Vercel project root.

No environment variables are required. The app uses only committed static benchmark and review artifacts.
