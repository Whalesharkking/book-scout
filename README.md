# Buch-Scout

Ein endlos laufender Agent, der mit einem lokalen LLM (Ollama, Gemma3 12B auf
AMD/ROCm) Bücher sucht, die zu deinem Leseprofil passen, und zwei getrennte
Top-20-Listen pflegt: **Fachbücher** und **Andere Bücher** (deutsch oder
englisch). Er lernt aus deinem Geschmack: die Bücher, die du in
`data/leseprofil.md` als gelesen und gut befunden einträgst, steuern die
künftigen Empfehlungen. Jeder Vorschlag
wird gegen Open Library verifiziert, damit keine erfundenen Titel in die
Listen gelangen.

## Voraussetzungen (einmalig)

```sh
sudo dnf install podman podman-compose
```

Der AMD-Treiber (`/dev/kfd`, `/dev/dri`) ist bereits vorhanden.

## Starten

```sh
podman-compose up -d --build
```

Beim ersten Start lädt Ollama das Modell `gemma3:12b` (~8 GB) herunter –
Fortschritt steht im Log. (Tipp: In `compose.yaml` steht, wie sich das
Ollama-Volume aus dem `llm`-Projekt wiederverwenden lässt. Beide Projekte
gleichzeitig laufen zu lassen lohnt sich nicht – sie teilen sich die GPU.)

### Warum gemma3:12b?

Für Buchempfehlungen zählt vor allem Weltwissen über reale Bücher (deutsch
und englisch) – Gemma3 ist dort und bei deutscher Sprache stärker als Qwen3.
Mit ~10 GB VRAM-Bedarf (Q4 + 8k Kontext) läuft es komfortabel in den 16 GB
der RX 9070 XT, nie am Limit. Über `ITERATION_SLEEP` lässt sich der
Suchdurchsatz steuern: kurze Pause = viele Iterationen pro Tag, lange Pause
(z.B. 120) = GPU idlet die meiste Zeit und der 24/7-Betrieb wird noch
sparsamer. Wer das bereits geladene `qwen3:14b` aus
dem Domain-Projekt weiterverwenden will: einfach `MODEL` umstellen,
funktioniert ebenfalls.

## Dein Leseprofil pflegen

[data/leseprofil.md](data/leseprofil.md) ist deine Eingabedatei, der Agent
liest sie bei jeder Iteration neu ein. Sie ist bewusst nicht im Repo
(persönliche Daten, siehe `.gitignore`) – als Vorlage dient
[data/leseprofil.example.md](data/leseprofil.example.md); fehlt die Datei,
legt der Agent beim Start automatisch eine leere Vorlage an.

- **Aktueller Lesewunsch:** Stichpunkte, welche Genres/Themen/Typen du gerade
  lesen möchtest (z.B. "Science-Fiction mit harter Wissenschaft",
  "Fachbücher zu Softwarearchitektur").
- **Gelesene Bücher:** Tabelle mit Titel, Autor und Typ (`fach`/`andere`) –
  einfach alle Bücher, die du gelesen und gut gefunden hast. Sie werden nie
  mehr vorgeschlagen, sofort aus den Listen entfernt und dienen Generator
  und Scorer als Geschmacksprofil ("mehr in diese Richtung").

Jede Änderung am Profil löst automatisch eine Neubewertung beider Listen aus.

## Beobachten

- **Empfehlungen:** [data/top_fach.md](data/top_fach.md) und
  [data/top_andere.md](data/top_andere.md) (werden laufend aktualisiert)
- **Log (kompakt, 1 Zeile pro Iteration):** `data/agent.log` oder
  `podman logs -f book-agent`
- **Zustand/geprüfte Bücher:** `data/state.json` (überlebt Neustarts,
  verhindert doppelte Vorschläge)

## Stoppen

```sh
podman-compose down
```

## Funktionsweise

1. **Profil einlesen:** Lesewunsch + gelesene Bücher aus `leseprofil.md`;
   gelesene Bücher werden gesperrt und aus den Listen entfernt.
2. **Generator** (LLM, kreativ): schlägt 10 reale Bücher für die aktuelle
   Kategorie vor – bekommt Profil, Top-20 und zuletzt geprüfte Titel mit,
   damit nichts wiederholt wird und die Vorschläge die Liste schlagen müssen.
3. **Existenz-Check** (Open Library): Titel + Autor müssen als echtes Buch
   auffindbar sein, sonst fliegt der Vorschlag raus (Halluzinations-Schutz).
   Liefert zudem kanonische Schreibweise und Erscheinungsjahr.
4. **Scorer** (LLM, streng, tiefe Temperatur): bewertet drei Dimensionen
   getrennt 0-100 – Passung zum Lesewunsch (45 %), Geschmacksnähe zu den
   gelesenen Büchern (35 %), Qualität/Renommee (20 %). Jede Bewertung läuft
   `SCORER_PASSES`-mal und wird gemittelt (weniger Zufallsrauschen). Dazu
   fließen echte Open-Library-Leserbewertungen mit 15 % in den Endscore ein
   (Bayes-gedämpft, damit wenige Einzelstimmen nicht dominieren; ohne
   Leserbewertungen gibt es einen leichten Malus statt Ausschluss). Das
   Score-Detail steht als eigene Spalte in den Listen.
5. **Top-20-Pflege:** schlechtere Einträge fliegen raus, maximal 2 Bücher
   pro Autor und Liste. Bei Profil-Änderung sofort, sonst alle
   ~24 Iterationen, wird jede Liste komplett neu bewertet.

## Konfiguration (compose.yaml)

| Variable | Default | Bedeutung |
|----------|---------|-----------|
| `MODEL` | `gemma3:12b` | Ollama-Modell (`qwen3:14b` funktioniert ebenfalls) |
| `ITERATION_SLEEP` | `10` | Pause in Sekunden zwischen Iterationen (höher = GPU-schonender) |
| `SCORER_PASSES` | `2` | Scoring-Durchläufe pro Bewertung, Ergebnisse werden gemittelt |

Falls die GPU nicht erkannt wird: in `compose.yaml` die Zeile
`HSA_OVERRIDE_GFX_VERSION: "12.0.1"` einkommentieren.
