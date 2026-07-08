# Book Scout

A book recommendation system built on real reader data instead of LLM
guesswork. It learns your taste from the books you enter as read and enjoyed
in `data/reading-profile.md` and maintains a top 20 list of recommendations.

How the recommendations are made:

1. **Collaborative filtering** on the [UCSD Goodreads dataset](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/)
   (~229 million real reader interactions, scraped 2017). A one-time build
   trains an ALS model; afterwards your favourites are just queries — no
   retraining when your profile changes.
2. **Goodreads reader ratings** (Bayes damped) contribute the quality signal.
3. **A local LLM** (Ollama, Gemma3 12B on AMD/ROCm) is only used when you set
   a current reading wish: it re-ranks the candidates for wish fit. Without a
   wish no GPU is needed at all.

Every recommendation is a real catalog book by construction — invented
titles are impossible. The catalog is English-language and ends at 2017
(the dataset's scrape date).

## Requirements (one time)

```sh
sudo dnf install podman podman-compose
```

The AMD driver (`/dev/kfd`, `/dev/dri`) must be present (only needed for the
wish re-ranker).

## Build the index (one time, ~6.5 GB download)

```sh
sh scripts/download_data.sh data/goodreads
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m scripts.build_index --data-dir data
```

The build streams the 4.3 GB interactions file in chunks and fits into 16 GB
of RAM; it takes roughly 30-60 minutes on 8 cores. Afterwards
`data/goodreads/` may be deleted — only `data/index/` (a few hundred MB) is
needed at runtime. `--min-book`, `--factors` and friends are documented in
`python -m scripts.build_index --help`.

Alternatively the build runs inside the container:

```sh
podman-compose build agent
podman-compose run --rm agent python -m scripts.build_index --data-dir /data
```

## Get recommendations

Enter your books in `data/reading-profile.md`, then run the agent — it
computes the list once and exits (venv from the index build step):

```sh
DATA_DIR=data .venv/bin/python -m agent.main
```

Without a reading wish this needs no GPU or containers and finishes in
about a second. With a reading wish, start Ollama first and point the agent
at it (first use downloads the model `gemma3:12b`, about 8 GB):

```sh
podman-compose up -d ollama
DATA_DIR=data OLLAMA_HOST=http://localhost:11434 .venv/bin/python -m agent.main
podman-compose down  # stops the GPU container again
```

Alternative without a host venv: `podman-compose run --rm agent` runs the
same one-shot inside a container (`OLLAMA_HOST` is preset there).

## Maintaining your reading profile

[data/reading-profile.md](data/reading-profile.md) is your input file; run
the agent again after changing it (it is deliberately not part of the repo —
personal data, see `.gitignore`). Use
[data/reading-profile.example.md](data/reading-profile.example.md) as a
template; if the file is missing, the agent creates an empty template on
startup.

- **Current reading wish:** bullet points describing what you want to read
  right now, for example "non-fiction about physics" or "hard science
  fiction". This is the only part that uses the LLM. Leave it empty for
  pure taste-based recommendations (no GPU involved).
- **Books read:** a table with title and author of books you read and
  enjoyed. They form your taste profile, are never proposed again, and
  disappear from the list. English original titles match the catalog best —
  check [data/profile-match.md](data/profile-match.md) to see which entries
  were recognized and fix the spelling of the rest.

## Results

- **Recommendations:** [data/top_books.md](data/top_books.md). Score detail
  per book: T = taste closeness (CF), Q = Goodreads reader quality,
  W = wish fit (LLM). "Because you liked" names the favourites that pulled
  the book in.
- **Profile matching:** [data/profile-match.md](data/profile-match.md)
- **Log:** `data/agent.log`

## Configuration (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATA_DIR` | `/data` | profile, lists and index location (`data` when running on the host) |
| `OLLAMA_HOST` | (empty) | Ollama URL for the wish re-ranker; empty disables it |
| `MODEL` | `gemma3:12b` | Ollama model for the re-ranker; `gemma3:4b` is a faster, slightly less book-savvy option |
| `POOL` | `200` | candidates per list before re-ranking (smaller = faster runs) |
| `SCORER_PASSES` | `2` | re-ranker passes per rating, results get averaged |

If the GPU is not detected, uncomment the line
`HSA_OVERRIDE_GFX_VERSION: "12.0.1"` in `compose.yaml`.
