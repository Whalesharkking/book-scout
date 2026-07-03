# Book Scout

An endlessly running agent that uses a local LLM (Ollama, Gemma3 12B on
AMD/ROCm) to hunt for books matching your reading profile. It maintains two
separate top 20 lists: **non-fiction** and **other books** (German or
English). It learns from your taste: the books you enter as read and enjoyed
in `data/reading-profile.md` steer all future recommendations. Every proposal is
verified against Open Library so that invented titles never reach the lists.

## Requirements (one time)

```sh
sudo dnf install podman podman-compose
```

The AMD driver (`/dev/kfd`, `/dev/dri`) must be present.

## Start

```sh
podman-compose up -d --build
```

On first start Ollama downloads the model `gemma3:12b` (about 8 GB). The
progress is shown in the log. Tip: `compose.yaml` explains how to reuse an
existing Ollama volume from another project.

### Why gemma3:12b?

For book recommendations, world knowledge about real books (German and
English) matters most. Gemma3 is stronger there and in German language
quality than Qwen3. With about 10 GB of VRAM (Q4 plus 8k context) it runs
comfortably within 16 GB and never at the limit. `ITERATION_SLEEP` controls
the search throughput: a short pause means many iterations per day, a long
pause (for example 120) keeps the GPU idle most of the time and makes 24/7
operation even cheaper. Other Ollama models work as well, simply change
`MODEL`.

## Maintaining your reading profile

[data/reading-profile.md](data/reading-profile.md) is your input file. The agent reads
it again on every iteration. It is deliberately not part of the repo
(personal data, see `.gitignore`). Use
[data/reading-profile.example.md](data/reading-profile.example.md) as a template. If
the file is missing, the agent creates an empty template on startup.

- **Current reading wish:** bullet points describing which genres, topics or
  types you want to read right now (for example "science fiction with hard
  science" or "non-fiction about software architecture"). Leave it empty for
  general recommendations matching your taste.
- **Books read:** a table with title, author and type (`nonfiction` or
  `other`). Simply all books you have read and enjoyed.
  They are never proposed again, get removed from the lists immediately and
  serve as the taste profile for generator and scorer.

Every change to the profile automatically triggers a fresh scoring of both
lists.

## Watching

- **Recommendations:** [data/top_nonfiction.md](data/top_nonfiction.md) and
  [data/top_other.md](data/top_other.md) (updated continuously)
- **Log (compact, one line per iteration):** `data/agent.log` or
  `podman logs -f book-agent`
- **State and checked books:** `data/state.json` (survives restarts and
  prevents duplicate proposals)

## Stop

```sh
podman-compose down
```

## How it works

1. **Read the profile:** reading wish plus read books from `reading-profile.md`.
   Read books are blocked and removed from the lists.
2. **Generator** (LLM, creative): proposes 10 real books for the current
   category. It receives the profile, the top 20 and the recently checked
   titles so that nothing is repeated and proposals have to beat the list.
3. **Existence check** (Open Library): title and author must be findable as
   a real book, otherwise the proposal is dropped (hallucination guard).
   Also provides the canonical spelling and the publication year.
4. **Scorer** (LLM, strict, low temperature): rates three dimensions
   separately from 0 to 100: fit to the reading wish (45 %), taste kinship
   with the read books (35 %) and quality or reputation (20 %). Every rating
   runs `SCORER_PASSES` times and gets averaged (less random noise). Real
   Open Library reader ratings contribute 15 % to the final score (Bayes
   damped so that a few single votes cannot dominate, books without reader
   ratings get a small penalty instead of being excluded). The score detail
   appears as its own column in the lists.
5. **Top 20 maintenance:** weaker entries drop out, at most 2 books per
   author and list. Each list is fully rescored immediately after a profile
   change and otherwise about every 24 iterations.

## Configuration (compose.yaml)

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL` | `gemma3:12b` | Ollama model (`qwen3:14b` works as well) |
| `ITERATION_SLEEP` | `10` | pause in seconds between iterations (higher is easier on the GPU) |
| `SCORER_PASSES` | `2` | scoring passes per rating, results get averaged |

If the GPU is not detected, uncomment the line
`HSA_OVERRIDE_GFX_VERSION: "12.0.1"` in `compose.yaml`.
