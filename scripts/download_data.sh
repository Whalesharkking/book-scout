#!/bin/sh
# Downloads the UCSD Goodreads dataset (Wan et al., scraped 2017) needed to
# build the recommendation index. About 6.5 GB in total; downloads resume
# when interrupted, files already complete are skipped.
#
# Usage: sh scripts/download_data.sh [target-dir]   (default: data/goodreads)
set -eu

TARGET="${1:-data/goodreads}"
BASE="https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"

mkdir -p "$TARGET"

for f in \
    goodreads_interactions.csv \
    book_id_map.csv \
    goodreads_books.json.gz \
    goodreads_book_authors.json.gz
do
    echo "==> $f"
    curl -fL --retry 5 --retry-all-errors -C - -o "$TARGET/$f" "$BASE/$f"
done

echo "Done. Next step: python -m scripts.build_index --data-dir data"
