"""Builds the recommendation index from the UCSD Goodreads dataset (one time).

Reads <data-dir>/goodreads/ (see scripts/download_data.sh) and writes
<data-dir>/index/:

- item_factors.npy   L2-normalized ALS item vectors (float32, one row per book)
- books.jsonl.gz     book catalog, line N describes factor row N
- meta.json          build parameters and counts

Designed for a 16 GB machine: the 4.3 GB interactions file is processed in
two streaming passes (count, then collect), never loaded whole.
"""

import os

# implicit parallelizes itself; nested BLAS threads would only fight over cores
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import gzip
import json
import logging
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

log = logging.getLogger("build")

CHUNK_ROWS = 5_000_000
ENGLISH = {"", "eng", "en", "en-US", "en-GB", "en-CA", "en-IN"}

# Genre buckets of goodreads_book_genres_initial. A book counts as
# non-fiction when the non-fiction shelf beats all clearly fictional ones;
# the mixed "history, historical fiction, biography" bucket is ignored.
NONFICTION_KEY = "non-fiction"
FICTION_KEYS = (
    "fiction",
    "fantasy, paranormal",
    "mystery, thriller, crime",
    "romance",
    "young-adult",
    "poetry",
    "comics, graphic",
    "children",
)


def _add_counts(counts: np.ndarray, ids: np.ndarray) -> np.ndarray:
    fresh = np.bincount(ids, minlength=counts.size)
    if fresh.size > counts.size:
        fresh[: counts.size] += counts
        return fresh
    counts[: fresh.size] += fresh
    return counts


def _iter_chunks(path: str):
    return pd.read_csv(
        path,
        usecols=["user_id", "book_id", "rating"],
        dtype={"user_id": np.int32, "book_id": np.int32, "rating": np.int8},
        chunksize=CHUNK_ROWS,
    )


def count_positives(path: str, min_rating: int) -> tuple[np.ndarray, np.ndarray]:
    """Pass 1: positives (rating >= min_rating) per user and per book."""
    user_counts = np.zeros(0, dtype=np.int64)
    book_counts = np.zeros(0, dtype=np.int64)
    rows = 0
    for chunk in _iter_chunks(path):
        liked = chunk[chunk["rating"].to_numpy() >= min_rating]
        user_counts = _add_counts(user_counts, liked["user_id"].to_numpy())
        book_counts = _add_counts(book_counts, liked["book_id"].to_numpy())
        rows += len(chunk)
        log.info("pass 1: %d million rows scanned", rows // 1_000_000)
    return user_counts, book_counts


def load_book_id_map(path: str) -> dict[int, int]:
    """goodreads book id -> csv book id (the interactions file uses csv ids)."""
    df = pd.read_csv(path, dtype=np.int64)
    csv_col, gr_col = df.columns[0], df.columns[1]
    return dict(zip(df[gr_col].to_numpy(), df[csv_col].to_numpy()))


def load_nonfiction_ids(path: str) -> set[int]:
    nonfiction = set()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            genres = row.get("genres") or {}
            nf = genres.get(NONFICTION_KEY, 0)
            if nf and nf > max((genres.get(k, 0) for k in FICTION_KEYS), default=0):
                nonfiction.add(int(row["book_id"]))
    return nonfiction


def load_author_names(path: str, wanted: set[str]) -> dict[str, str]:
    names = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("author_id") in wanted:
                names[row["author_id"]] = row.get("name", "")
    return names


def select_books(
    books_path: str, gr_to_csv: dict[int, int], book_counts: np.ndarray, min_book: int
) -> list[dict]:
    """Streams the 2.3M book metadata lines and keeps English editions with
    enough positive interactions, deduplicated to one edition per work
    (the one with the most positives)."""
    best_per_work: dict[str, dict] = {}
    scanned = 0
    with gzip.open(books_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            scanned += 1
            if scanned % 500_000 == 0:
                log.info("metadata: %d lines scanned, %d works kept", scanned, len(best_per_work))
            row = json.loads(line)
            csv_id = gr_to_csv.get(int(row["book_id"]))
            if csv_id is None or csv_id >= book_counts.size:
                continue
            positives = int(book_counts[csv_id])
            if positives < min_book or row.get("language_code") not in ENGLISH:
                continue
            title = row.get("title_without_series") or row.get("title") or ""
            authors = row.get("authors") or []
            if not title or not authors:
                continue
            work = row.get("work_id") or row["book_id"]
            current = best_per_work.get(work)
            if current is None or positives > current["pos"]:
                best_per_work[work] = {
                    "id": int(row["book_id"]),
                    "csv_id": csv_id,
                    "work": work,
                    "title": title,
                    "author_id": authors[0].get("author_id", ""),
                    "year": int(row["publication_year"]) if row.get("publication_year") else None,
                    "avg": float(row["average_rating"]) if row.get("average_rating") else None,
                    "count": int(row["ratings_count"]) if row.get("ratings_count") else 0,
                    "pos": positives,
                }
    return list(best_per_work.values())


def collect_interactions(
    path: str, item_row: np.ndarray, user_ok: np.ndarray, min_rating: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pass 2: (user, item) pairs of positive ratings on kept books/users."""
    users, items = [], []
    for chunk in _iter_chunks(path):
        u = chunk["user_id"].to_numpy()
        b = chunk["book_id"].to_numpy()
        mask = (chunk["rating"].to_numpy() >= min_rating) & (b < item_row.size) & (u < user_ok.size)
        rows = item_row[b[mask]]
        u = u[mask]
        keep = (rows >= 0) & user_ok[u]
        users.append(u[keep].astype(np.int32))
        items.append(rows[keep])
    return np.concatenate(users), np.concatenate(items)


def train(users: np.ndarray, items: np.ndarray, n_items: int, args) -> np.ndarray:
    from implicit.als import AlternatingLeastSquares

    uniq, users = np.unique(users, return_inverse=True)
    matrix = csr_matrix(
        (np.full(len(items), float(args.alpha), dtype=np.float32), (users, items)),
        shape=(len(uniq), n_items),
    )
    log.info("training ALS: %d users x %d books, %d interactions", len(uniq), n_items, matrix.nnz)
    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        random_state=42,
    )
    model.fit(matrix)
    factors = model.item_factors.astype(np.float32)
    norms = np.linalg.norm(factors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return factors / norms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--min-rating", type=int, default=4, help="rating counting as 'liked'")
    parser.add_argument("--min-book", type=int, default=60, help="min positives per book")
    parser.add_argument("--min-user", type=int, default=3, help="min positives per user")
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--regularization", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=20.0, help="ALS confidence weight")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    src = os.path.join(args.data_dir, "goodreads")
    out = os.path.join(args.data_dir, "index")
    os.makedirs(out, exist_ok=True)
    interactions = os.path.join(src, "goodreads_interactions.csv")
    started = time.time()

    user_counts, book_counts = count_positives(interactions, args.min_rating)
    log.info(
        "pass 1 done: %d users / %d books with at least one positive rating",
        int((user_counts > 0).sum()),
        int((book_counts > 0).sum()),
    )

    gr_to_csv = load_book_id_map(os.path.join(src, "book_id_map.csv"))
    books = select_books(
        os.path.join(src, "goodreads_books.json.gz"), gr_to_csv, book_counts, args.min_book
    )
    books.sort(key=lambda b: -b["pos"])
    log.info("catalog: %d works kept (English, >= %d positives)", len(books), args.min_book)

    nonfiction = load_nonfiction_ids(os.path.join(src, "goodreads_book_genres_initial.json.gz"))
    names = load_author_names(
        os.path.join(src, "goodreads_book_authors.json.gz"), {b["author_id"] for b in books}
    )
    for b in books:
        b["nf"] = b["id"] in nonfiction
        b["author"] = names.get(b.pop("author_id"), "")

    # factor row N = catalog line N
    item_row = np.full(book_counts.size, -1, dtype=np.int32)
    for row, b in enumerate(books):
        item_row[b.pop("csv_id")] = row
    user_ok = user_counts >= args.min_user

    users, items = collect_interactions(interactions, item_row, user_ok, args.min_rating)
    factors = train(users, items, len(books), args)

    np.save(os.path.join(out, "item_factors.npy"), factors)
    with gzip.open(os.path.join(out, "books.jsonl.gz"), "wt", encoding="utf-8") as fh:
        for b in books:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "built": time.strftime("%Y-%m-%d %H:%M"),
                "books": len(books),
                "interactions": len(items),
                "params": vars(args),
                "minutes": round((time.time() - started) / 60, 1),
            },
            fh,
            indent=1,
        )
    log.info("index written to %s (%.1f min)", out, (time.time() - started) / 60)


if __name__ == "__main__":
    main()
