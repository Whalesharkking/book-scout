# Reading Profile

Example file. Copy it to `data/reading-profile.md` (or simply run the
agent once, it creates an empty template) and enter your own data. Run the
agent again after every change to recompute both leaderboards.

## Current reading wish

Bullet points describing what you want to read right now. This is the only
part that uses the LLM re-ranker. Leave empty for pure taste-based
recommendations.

- Books about space travel
- Historical novels with a real historical background

## Books read

Books you have read and enjoyed. Type: `nonfiction` or `other`. They form
your taste profile and are never proposed again. English original titles
match the Goodreads catalog best — `data/profile-match.md` shows which
entries were recognized.

| Title | Author | Type |
|-------|--------|------|
| The Martian | Andy Weir | other |
| A Brief History of Time | Stephen Hawking | nonfiction |
