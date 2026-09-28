# Setsmith

Setsmith is a transition-aware DJ set builder for Rekordbox. It reads your Rekordbox XML export, scores how well any two tracks mix, suggests what to play next, and builds whole sets that follow an energy curve. It writes those sets back as Rekordbox playlists. Every score comes with its reasons.

Setsmith only reads your data. It never modifies your Rekordbox database or your audio files.

Track discovery uses data from [Last.fm](https://www.last.fm), BPM and key data from [GetSongBPM](https://getsongbpm.com), and BPM and key estimated from [Deezer](https://www.deezer.com)'s 30-second previews.

**Status: all six phases (suggestions, set building, Rekordbox export, listening feedback, local audio analysis, DJ style profiles, live-set analysis, learned preferences, local web UI).**

## Setup

Requires Python 3.11+. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run setsmith --help
```

For audio analysis (`setsmith analyze`), add the extras:

```bash
uv sync --extra audio --extra essentia
```

- `audio` installs librosa. It is required for analysis and handles beats, loudness, energy features, intro/outro detection, and a fallback key detector.
- `essentia` adds Essentia for EDM-tuned key detection. Essentia is **AGPL-3.0**: fine for personal use, but distributing Setsmith or serving it to others with Essentia installed brings AGPL obligations. Leave the extra out to avoid that; analysis then falls back to librosa.

Pass the same `--extra` flags every time you run `uv sync`, or it removes the extras. On Apple silicon, Setsmith pins Essentia to the last build with wheels for macOS 11+, which needs NumPy 1.x. Other platforms get the latest build.

## Export your collection from Rekordbox

1. In Rekordbox, choose **File > Export Collection in xml format**.
2. Save the file somewhere (for example `~/Music/rekordbox.xml`).
3. Run Setsmith on that file. It never writes to it.

## Commands

Every command supports `--help` and `--json`.

### `suggest`: what should I play next?

```bash
uv run setsmith suggest ~/Music/rekordbox.xml --track "Kaya Sol - Sunwater"
```

```
Seed  Kaya Sol - Sunwater (Extended Mix)  122 BPM · 8A (A minor) · E6 · Afro House
┏━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━┳━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃# ┃ Track                                       ┃ BPM ┃ Key ┃ E ┃ Score ┃ Transition      ┃
┡━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━╇━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│1 │ Moana Reyes - Lanterns                      │ 123 │ 9A  │ 7 │    97 │ long_blend 32b  │
│2 │ Nils Ødegaard - Glass Harbour (Café Remix)  │ 124 │ 8B  │ 6 │    87 │ long_blend 32b  │
│3 │ untitled_final_v3                           │   - │ -   │ - │    50 │ filter_sweep 16b│
│  │ missing_key, missing_bpm, missing_energy,   │     │     │   │       │                 │
│  │ missing_genre                               │     │     │   │       │                 │
│4 │ Dex Marlo - Floor Tom                       │ 128 │ 11A │ - │    35 │ cut 8b          │
│  │ key_clash, tempo_stretch, variable_tempo,   │     │     │   │       │                 │
│  │ missing_energy                              │     │     │   │       │                 │
└──┴─────────────────────────────────────────────┴─────┴─────┴───┴───────┴─────────────────┘
```

On terminals 140 or more columns wide, the table also shows each component score and puts the flags in their own column. Add `--explain` (`-x`) to see the reasoning behind each score:

```
1. Moana Reyes - Lanterns  97.0
   harmonic     0.90 x 0.30 = 27.0  8A -> 9A (adjacent, +1)
   tempo        1.00 x 0.25 = 25.0  122 -> 123 BPM, needs 0.8% pitch change
   energy       1.00 x 0.20 = 20.0  E6 -> E7 (+1, target +0)
   genre_style  1.00 x 0.15 = 15.0  same genre (afro house)
   extras       1.00 x 0.10 = 10.0  rated 5/5
   transition   long_blend, 32 bars: tempo and key compatible; swap basslines on a phrase boundary
```

Options:

- `--track "Artist - Title"`, or just the title. Matching is fuzzy and ignores accents and punctuation. If several tracks match, Setsmith lists them with their TrackIDs so you can pick one with `--id`.
- `--id 12345` picks the seed by Rekordbox TrackID.
- `--top 20` sets the number of suggestions.
- `--energy-delta 2` asks for tracks about 2 energy points above the seed. Use `-1` to ease off. The default is `0`, which holds the energy steady.
- `--json` prints the full breakdown, including every reason, for scripting.

### `info`: check what Setsmith sees in your collection

```bash
uv run setsmith info ~/Music/rekordbox.xml
```

This shows metadata coverage (BPM, key, energy, genre, rating, cues), the most common keys and genres, and parse warnings such as key tags it could not read.

### `analyze`: analyze your audio files

```bash
uv run setsmith analyze ~/Music/rekordbox.xml                    # whole collection
uv run setsmith analyze ~/Music/rekordbox.xml --playlist "Gigs/Friday"
```

`analyze` reads the file at each track's Location (never modifying it) and stores derived numbers in the Setsmith database:

- **Key**: Essentia's `KeyExtractor` with the `edma` EDM profile (`--key-profile bgate` and others work too), or Krumhansl-Kessler template matching when Essentia isn't installed.
- **Energy features**: RMS loudness, onset rate, spectral centroid and spectral flux.
- **Intro and outro length in bars**: Setsmith walks your Rekordbox beat grid bar by bar and measures harmonic energy. Harmonic/percussive separation removes kicks and hats, so a drums-only intro reads as low and the section where the bassline and chords arrive reads as high. Results snap to 8-bar phrases. Tracks without a grid use the tag BPM, with a beat tracker for phase.
- Detected tempo, danceability (Essentia), and optionally EBU R128 loudness (`--lufs`, slower, informational only).

It runs in parallel (`--workers`, default: cores − 1) and shows progress. Unchanged files are skipped on later runs; results are keyed by path, modification time and size. Ctrl-C keeps everything finished so far. On this machine a 6-minute MP3 takes about 2.7s, so about 1,000 tracks per 6-7 minutes on 7 workers. Files that fail to decode, are shorter than 30s, or can't be found are listed at the end.

After analysis, `suggest`, `build`, `info` and `feedback` use the results automatically (`--no-analysis` turns that off):

- **Energy** (1-10) is calibrated to your library. Each feature becomes a z-score across your analyzed tracks, the weighted sum is ranked, and ranks are spread over 1-10. An energy tag you wrote in Comments still wins.
- **Key confidence**: the detected key is compared with the Rekordbox tag. Agreement keeps confidence 1.0; adjacent or relative keys (typical detector confusions) give 0.7; anything else gives 0.4, which pulls harmonic scores toward neutral. The tag itself is kept, and `analyze` lists the disagreements so you can check them by ear. Tracks without a key tag take the detected key at confidence 0.8.
- **Intro/outro bars** feed the arrangement score and the long-blend decision.

**allin1 (optional, untested here).** `--structure allin1` uses [allin1](https://github.com/mir-aidj/all-in-one)'s music-structure segments instead of the beat-grid method. It is not a declared extra, because its dependencies (PyTorch, NATTEN and madmom) don't install cleanly on every platform: madmom's PyPI release fails to build here. Install it yourself following allin1's instructions. Note that madmom's model files are licensed non-commercial (CC BY-NC-SA 4.0), and allin1 downloads pretrained models on first use.

**Any song as the seed.** `--song` takes a song whether it's in your library or not: `--song "Francis Mercier - Kamili"`, or free text such as `--song "kamili francis mercier"`, which is searched on Last.fm (needs a Last.fm key; see [`discover`](#discover-find-tracks-you-dont-own-yet)). A song you own is used as is. For anything else, BPM and key come from GetSongBPM or are estimated from the Deezer preview (see [BPM and key for songs outside your library](#bpm-and-key-for-songs-outside-your-library)), and the genre from Last.fm. The same works in the web UI: type the song and pick **Use "..." (any song)**, or just press the button.

### `build`: build a whole set

```bash
uv run setsmith build ~/Music/rekordbox.xml --minutes 60 --curve journey \
    --name "Friday opener" --out ~/Music/setsmith.xml
```

```
Friday opener
11 tracks, ~61 min, 119.58-126.71 BPM, mean transition 92 (lowest 86), energy off-curve by 0.54 on average
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ # ┃ Track                          ┃    BPM ┃ Key ┃  E (tgt) ┃ Next ┃ Transition      ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 1 │ Nils Okafor - Fable Bloom 145  │ 122.77 │ 2A  │  4 (3.0) │   98 │ long_blend 32b  │
│ 2 │ Tomas Vance - Static Dust 25   │ 121.61 │ 2A  │  3 (3.9) │   90 │ long_blend 32b  │
│ 3 │ Ines Marlo - Static Dust 285   │ 123.65 │ 2A  │  4 (4.7) │   91 │ long_blend 32b  │
│ 4 │ Rafa Okafor - Signal Fable 114 │ 126.71 │ 2A  │  6 (5.6) │   90 │ long_blend 32b  │
│ 5 │ Oli Reyes - Drift Lantern 163  │ 124.66 │ 2A  │  6 (6.4) │   95 │ long_blend 32b  │
│ 6 │ Kaya Vance - Harbour Drift 99  │ 123.73 │ 1A  │  8 (7.3) │   92 │ long_blend 32b  │
│ 7 │ Sade Keel - Fable Pulse 34     │ 124.43 │ 12A │  8 (8.1) │   97 │ long_blend 32b  │
│ 8 │ Juno Vance - Echo Dust 278     │ 125.32 │ 11A │ 10 (9.0) │   86 │ filter_sweep 16b│
│   │ tempo_stretch                  │        │     │          │      │                 │
│ 9 │ Kaya Marlo - Ember Fable 174   │ 119.58 │ 11A │  8 (7.7) │   90 │ long_blend 32b  │
│10 │ Sade Noor - Dust Fable 173     │ 121.43 │ 10A │  6 (6.3) │   92 │ long_blend 32b  │
│11 │ Ines Vance - Ember Tidal 287   │ 123.67 │ 9A  │  5 (5.0) │      │ end             │
└───┴────────────────────────────────┴────────┴─────┴──────────┴──────┴─────────────────┘
Alternates (fit both neighbors; swap in to read the crowd)
   1  Rafa Brandt - Dust Harbour 98 121.01 1A E4  ·  Tomas Keel - Drift Tidal 29 122.74 3A E2
   2  Sade Sol - Fable Echo 226 124.84 1A E4  ·  Rafa Brandt - Canopy Dust 167 121.36 1A E3
   ...
Wrote 'Friday opener', 'Friday opener (alternates)' to setsmith.xml.
```

`E (tgt)` is the track's energy with the curve's target in brackets. `Next` is the transition score into the following track. Each position also lists two alternates that fit both its neighbors, so you can branch while reading the crowd.

**Length and shape:**

- `--minutes 90` or `--tracks 16` sets the length. The default is 60 minutes. Minutes are converted to tracks using the pool's median track length minus a 32-bar blend.
- `--curve` takes one of the templates below or custom points: `3,5,8,6` (evenly spaced) or `0:3,0.6:9,1:5` (position:energy).

| Curve | Shape |
|---|---|
| `warm_up` | 3 rising slowly to 5, never above 6 |
| `peak_time` | 6 rising to 9, with a 1.5-point dip every 5th track |
| `closing` | holds 8, then eases to 5.5 over the last quarter |
| `journey` (default) | 3 up to 9 at 70%, then down to 5 |

**Filters:**

- `--around "Artist - Title"` (repeat, up to 5; any songs) mixes in songs you don't own, found around those songs: similar songs from the same scene, like [`discover`](#discover-find-tracks-you-dont-own-yet), with BPM and key from GetSongBPM or Deezer previews. `--new-share` sets roughly how much of the set is new songs (default 0.5; 1 means as many as fit). New songs get their genre from your own tags for the same artist when you have any, else from the song they were found around. Their estimated keys and missing energy make them score lower than your analyzed tracks, so the search paces them in toward the share (a bonus while behind, only new songs once a whole song behind); the transition scores shown are never adjusted. New songs are marked `(new)`, listed under **Songs to get** with SoundCloud, Beatport and Deezer links, and left out of the Rekordbox export, which keeps your own tracks in set order. The web UI has the same options under **Mix in new songs around**.
- `--start "Artist - Title"` or `--start-id` fixes the opening track. `--start-song` takes any song, even one you don't own (looked up like `suggest --song`). The set is built to follow it, but it can't go into the exported playlist, which then starts at track 2: add the song in Rekordbox and put it first.
- `--bpm-min` / `--bpm-max` limit the tempo range.
- `--genre` limits genres. Repeat it for several; spellings are normalized, so `afro-house` matches `Afro House`.
- `--exclude` leaves out a TrackID or `"Artist - Title"`. Repeat it for several.
- `--tag` keeps only tracks with a given Rekordbox My Tag (after `setsmith rekordbox import`).
- `--style` applies a DJ style profile (see [`styles`](#styles-dj-style-profiles)).
- `--artist-gap 4` sets how many tracks must pass before an artist repeats (the default is 4; `0` turns the rule off).

**Output:**

- `--out setsmith.xml` writes the set, plus an alternates playlist, into a new Rekordbox XML file (`--no-alternates-playlist` skips the second one). Build again into the same file and your earlier Setsmith playlists are kept; a playlist with the same name is replaced. Setsmith refuses to write over your input file, or over any file it didn't create unless you pass `--force`.
- `--report set.md` writes the full breakdown: every transition's reasoning and all alternates. Use `--report set.json` for JSON.
- `--write-comments` puts each track's position and score into Comments in the exported file only. Rekordbox may copy those into your library's Comments when you import, so it is off by default.
- `--json` prints everything for scripting.

Pair scores are cached in SQLite, so rebuilding from the same library is faster (see [Data Setsmith stores](#data-setsmith-stores)).

### `styles`: DJ style profiles

```bash
uv run setsmith styles list
uv run setsmith styles show keinemusik
uv run setsmith build ~/Music/rekordbox.xml --minutes 90 --style keinemusik
uv run setsmith suggest ~/Music/rekordbox.xml -t "Artist - Title" --style franky_rizardo --explain
uv run setsmith build ~/Music/rekordbox.xml --minutes 60 --style afro_house     # a genre style
uv run setsmith styles artist "Black Coffee" --collection ~/Music/rekordbox.xml  # any artist
```

A style profile is a small JSON file describing a DJ style's musical parameters: BPM band, preferred key modes and key moves, genre weights, energy curve, vocal share, and typical transition lengths and types. Built in are three artist styles, **inspired by** Keinemusik, Brunello and Franky Rizardo, and six genre styles: Afro House, Deep House, Tech House, Melodic House & Techno, Organic House and House, with each genre's usual tempo range and mixing habits. They are starting points written from public track metadata and press descriptions, not measured from real sets, and the artist styles imply no endorsement by the artists. `liveset` can also draft a profile from a set you analyze.

**A style for any artist.** `setsmith styles artist "Name"` (or **Style → Any artist...** in the web UI) estimates one:
- **Tempo range and minor/major balance** come from the artist's 12 most popular tracks on Last.fm, with BPM and key from GetSongBPM or Deezer previews. The range spans the 20th to 80th percentile, so an odd slow or fast production doesn't stretch it.
- **Mixing habits** (key moves, transition lengths and types, energy curve) come from the genre style that matches the artist. Your own genre tags for the artist are used first, then Last.fm's, specific genres before the umbrella "house".
- **Where it's saved:** your styles folder, as e.g. `black_coffee.json`, labeled "Inspired by …". Re-estimating replaces only an earlier estimate, never a built-in or a profile you made.
- **Requirements:** a Last.fm key, and tempos for at least 3 of the tracks.

What a style changes:

| Profile field | Effect |
|---|---|
| `bpm_band` | `build`: a hard limit, unless you pass `--bpm-min`/`--bpm-max` (each replaces its side). Style fit: 1.0 inside, falling to 0 at 4 BPM outside. |
| `bpm_preferred` | `build`: breaks ties when choosing the opening track. |
| `max_tempo_drift_bpm` | `build`: tempo range allowed before the drift penalty (`--max-drift` overrides). |
| `energy_curve` | `build`: the default `--curve`. |
| `allowed_key_moves` | Blended 50/50 into the harmonic score. A move the profile doesn't list counts as 0. |
| `genre_weights` | Blended 50/50 into the genre score, using the incoming track's genre. Genres the profile doesn't list count as 0.2. |
| `key_mode_preference`, `vocal_density` | Part of style fit. |
| `transition_length_bars` | A 64-bar share of 0.25 or more makes long blends 64 bars instead of 32. |
| `transition_type_mix` | Chooses between cut and echo out when a pair can't be blended. `build` reports the set's actual mix next to the profile's. |
| `reference_artists`, `reference_labels`, `notes`, `sources` | Documentation only. |

**Style fit** is a weighted average of BPM band (0.35), genre weight (0.35), key mode (0.2) and vocal share (0.1), using whichever parts are known for the track. In `build`, each step earns style fit × 15 points. In `suggest`, the fit is shown in its own column but not added to the transition score.

Vocal information comes from metadata for now: whole words like "vocal" or "vox" versus "dub" or "instrumental" in the title, the Mix field or Comments. Contradictory or missing words mean unknown, and unknown parts are left out of style fit.

**Your own profiles.** `setsmith styles copy keinemusik my_style` copies a profile into `~/.config/setsmith/styles/` (or `$SETSMITH_STYLES_DIR`), where you can edit it. A file there with the same name as a built-in replaces it. `--style` also takes a path to any `.json` file. Profiles are validated on load: mixes must sum to 1 and weights must be between 0 and 1. `styles list` flags broken files.

### `rekordbox`: import My Tags and history from master.db (optional)

```bash
uv sync --extra rekordbox          # plus your other extras
export SETSMITH_REKORDBOX_KEY=...   # your master.db key; Setsmith never fetches one
uv run setsmith rekordbox import --backed-up
uv run setsmith rekordbox status
```

Rekordbox keeps My Tags and play history in its own database (`master.db`), not in the XML export. This command reads them through [pyrekordbox](https://github.com/dylanljones/pyrekordbox), with these safeguards:

- **Back up first.** In Rekordbox, choose File > Library > Backup Library. The command refuses to run without `--backed-up`.
- **Setsmith never opens your live database.** It copies `master.db` (and its `-wal`/`-shm` files) into its own data folder (`rekordbox-copies/` next to the Setsmith database, newest 3 kept) and reads only the copy. It never commits anything, even to the copy.
- **You supply the key.** Rekordbox 6/7 encrypt `master.db`. Pass the key with `--key` or `$SETSMITH_REKORDBOX_KEY`. Setsmith never downloads a key and never uses the key pyrekordbox bundles. Reading the encrypted database may fall outside AlphaTheta's terms; that is your call.
- **Quit Rekordbox first** for a clean snapshot. The command warns you if Rekordbox is running.

What it imports, matched to your XML tracks by file path:

- **My Tags** as "Category: Tag". A My Tag named "Vocal" or "Instrumental" (or "Dub") sets the track's vocal flag and overrides guesses from titles and comments. `build --tag Opener` (repeatable) keeps only tracks with any of the given tags. `info` shows tag coverage.
- **History sessions** (each played playlist, in order) are stored for Phase 5, which will learn your key-move and tempo habits from them.

A new import replaces the previous one.

### `liveset`: learn from sets you played or admire

```bash
uv run setsmith liveset analyze ~/Music/rekordbox.xml friday.txt                      # tracklist only
uv run setsmith liveset analyze ~/Music/rekordbox.xml friday.txt --audio friday.mp3   # + recording
uv run setsmith liveset analyze ~/Music/rekordbox.xml friday.txt --audio friday.mp3 --save-profile my_fridays
uv run setsmith liveset list
uv run setsmith liveset show 1
uv run setsmith liveset profile 1 --save-profile my_fridays
```

You supply the inputs: a **tracklist** you paste into a text file, and optionally a **recording** of the set that you are entitled to analyze. Setsmith never scrapes tracklist sites or downloads audio.

Tracklist lines can look like `01. Artist - Title`, `[00:06:10] Artist – Title [Label]`, `1:02:33 Artist - Title`, `w/ Artist - Title` (layered with the previous track) or `ID - ID` (unidentified). A CSV with Artist/Title(/Time) columns also works. Each line is fuzzy-matched to your collection; version labels such as "(Original Mix)" or "(Extended Mix)" are ignored, but remix names still count. Lines that don't match are listed.

What you get depends on what you provide:

| Input | Result |
|---|---|
| Tracklist | Key moves and tempo changes between consecutive tracks (from your tags), genre mix, energy order. |
| + recording | Tempo, key and loudness every 30 seconds. Transition points come from the tracklist timestamps, or from novelty detection if there are none; either way they are marked approximate. |
| + your original files for the matched tracks | Each original is aligned to the recording with beat-synchronous chroma + MFCC features and subsequence DTW (after Kim et al., ISMIR 2020), which finds where it plays and from which point of the original. Setsmith then estimates each track's gain beat by beat (non-negative least squares on mel spectra) and extrapolates the fades, so each transition gets a **cue-out**, a **cue-in** and an **overlap length in bars**. |

The recording is read in 30-second blocks straight from disk (WAV, FLAC or MP3). It is never copied, and only the derived numbers are stored. On this machine a 30-minute MP3 takes about 13s to read, and aligning each track takes well under a second.

Each analysis ends with a **draft style profile**: BPM band from the 10th–90th percentile of track tempos, key-move weights from how often each move occurs (with add-one smoothing), energy curve from the recording's loudness or the tracks' energies, genre weights, vocal share, and transition lengths and types from the measured overlaps. It is printed for you to review. Save it with `--save-profile NAME`, edit it in your styles folder, and use it with `--style NAME`. When nothing was measured (no recording or no original files), transition lengths and types fall back to neutral defaults, and the profile's notes say so.

Limits: transition types are estimated from overlap length alone (long blend ≥ 16 bars, filter sweep ≥ 4, otherwise cut), so echo outs and drop swaps aren't told apart. Alignment was verified on synthesized mixes with known answers (overlaps within about a bar), not yet on real recordings. Two tracks built on identical loops are hard to tell apart. Tempo-stretched playback is supported in principle by the beat-synchronous features but hasn't been tested.

### `learn`: fold your own habits into the weights

```bash
uv run setsmith learn ~/Music/rekordbox.xml
uv run setsmith suggest ~/Music/rekordbox.xml -t "Artist - Title" --learned
uv run setsmith build ~/Music/rekordbox.xml --minutes 90 --learned
```

`learn` counts how often you make each Camelot move and each size of tempo change. It uses consecutive tracks in your Rekordbox history sessions (after `setsmith rekordbox import`) and transitions in analyzed live sets. It then shows defaults next to learned values. A learned value is a move's frequency relative to your most frequent move, blended with the default using weight n / (n + prior). The default prior is 50, so 10 transitions shift the scores a little and 500 mostly replace them. `--prior` changes that. The learned weights apply only when you pass `--learned`, and the pair-score cache keeps them separate from the defaults.

### `web`: a local web UI

```bash
uv sync --extra web               # plus your other extras
uv run setsmith web ~/Music/rekordbox.xml --open
```

The UI opens at <http://127.0.0.1:8765/> and has three tabs:

- **Build a set** has the same options as `build`: length, curve, style, BPM range, opening track, genres and learned weights. It shows the set on a **timeline**. Tracks alternate between two deck lanes, colored by Camelot key. Each transition is a shaded overlap with its score between the lanes (hover for the type and length), and an energy strip on top plots the tracks' energy against the curve's targets. Click a track for its tags and the full reasoning for the transitions in and out, plus its alternates. **Export Rekordbox XML** downloads the set and alternates as a new file; **Report** opens the Markdown breakdown.
- **Suggest next** searches as you type (any words from the artist or title), then ranks what to play next, with an optional style fit column and the reasoning behind each score.
- **Discover** finds candidate tracks you don't own for after a seed, from the same scene (see [`discover`](#discover-find-tracks-you-dont-own-yet)).
- **Live sets** analyzes a pasted tracklist (matched to your library: key moves and tempo changes between tracks) and shows analyzed sets on the same timeline. Sets analyzed with a recording in the terminal (`liveset analyze --audio`) also show where each track played, measured overlaps, cue points and the loudness curve.

Safety: the server listens on 127.0.0.1 only by default. It reads your collection but never writes it. It only answers requests addressed to a local host name, which guards against DNS rebinding. It accepts style profiles by name only, never as a file path. Exports are built in a temporary folder and streamed to your browser as downloads. `--host` exposes the UI to your network with no login, and prints a warning (including the AGPL note when Essentia is installed). The API docs are at `/api/docs`.

### `discover`: find tracks you don't own yet

```bash
uv run setsmith keys set lastfm                  # required, free; asks for the key without echoing it
uv run setsmith keys set getsongbpm              # optional, free: adds BPM and key
uv run setsmith keys set lastfm --from-clipboard # or: copy the key, then read it from the clipboard (macOS)
uv run setsmith discover ~/Music/rekordbox.xml -t "Artist - Title" --style keinemusik
uv run setsmith discover ~/Music/rekordbox.xml --song "Francis Mercier - Kamili"  # any song
```

The web UI has the same feature in its **Discover** tab.

For a seed track, Setsmith asks **Last.fm** for similar tracks and for popular tracks by similar artists, and drops anything already in your library. For the most relevant candidates (25 by default) it gets BPM and key from **GetSongBPM**, or else estimates them from the **Deezer** preview, and takes a genre from the artist's Last.fm tags. Each candidate with a known BPM and key is scored as a transition from your seed, with your style profile if you give one. Estimated values show as `≈122` in the web UI. Results link to the track on Last.fm and Deezer and to searches on SoundCloud and Beatport, so you can listen and add or buy it.

**Scene match.** Last.fm's similar tracks mean "listeners also played", so a crossover hit as the seed pulls in mainstream EDM and pop. Setsmith therefore also asks how similar each candidate's *artist* is to the seed's artist on Last.fm (the **Scene** column). For "Adam Port - Move", Rampa, MoBlack and &ME score high, while Martin Garrix scores 13% and Fred again.. isn't in the list at all. Candidates below 20% show as `other` and go to the bottom of the list.

Discover is a **candidate finder**, not a set builder: it tells you what to listen to next, and **Suggest** and **Build** judge the mix once a track is in Rekordbox. Results come in four groups:

1. Known BPM and key, and a good transition (score 60 or more), by score.
2. Unknown BPM or key, by relevance: Last.fm similarity combined with scene match. These show `?` instead of a score, because Setsmith can't judge a mix it knows nothing about.
3. Known BPM and key, but a poor transition (a key clash or a big tempo jump), by score.
4. Artists outside the seed artist's scene, by relevance.

GetSongBPM covers mainstream releases well but has little for underground dance music; the Deezer previews fill most of that gap. `--json` output marks each result with `bpm_key_known`, `bpm_key_source`, `scene_match`, `off_scene` and `rank_basis`.

**Getting the keys (both free):**

- **Last.fm:** sign in and create an API account at <https://www.last.fm/api/account/create>. You only need the "API key".
- **GetSongBPM:** register at <https://getsongbpm.com/api>. Its terms require a visible link back to GetSongBPM; Setsmith shows one with every result.

`setsmith keys set` saves keys to `~/.config/setsmith/keys.json`, readable only by you. The environment variables `SETSMITH_LASTFM_KEY` and `SETSMITH_GETSONGBPM_KEY` take precedence. `setsmith keys status` shows what is configured without printing the keys.

Without a GetSongBPM key, BPM and key come from Deezer previews only. Without the `audio` extra either, every result is unknown and ranked by Last.fm similarity.

#### BPM and key for songs outside your library

GetSongBPM is asked first. For songs it doesn't know, Setsmith finds the track on Deezer's public API (no key needed), checking that the title and an artist match, and analyzes its official 30-second preview. This needs the `audio` extra (`uv sync --extra audio`); with the `essentia` extra, keys use Essentia's EDM profile.

- **BPM:** three tempo estimators vote (librosa's beat tracker and Essentia's rhythm extractor and Percival estimator); the BPM is the mean of those that agree within 2%. If none agree, no BPM is given, because a wrong tempo is worse than none. Against Rekordbox on 60 tracks this gave a BPM for all 60, within 1.5 BPM for 54; most misses were DJ edits at a different tempo from the original. Tempos are folded into 88-176 BPM.
- **Key:** an estimate. Checked against Rekordbox's own analysis of 40 tracks, Essentia matched the key exactly for about half, and was at most one step off on the Camelot wheel for two thirds. Estimated keys carry a key confidence of 0.6, so their harmonic scores are pulled toward neutral.
- The preview is decoded in memory and discarded. Only the BPM, key and key strength are stored, cached per song for 180 days, so repeat searches need no Deezer requests. Deezer search results are kept for only 10 minutes, because the preview links in them expire. It's for personal use: Deezer's previews are meant for listening, so check their terms before using this beyond that.

Once you add a track to Rekordbox, its analysis of the full track replaces these estimates.

**Responsible use:** only official APIs are used, with your own keys and non-commercially, as Last.fm's terms require. Responses are cached in the Setsmith database for 7 days, with a size cap well under Last.fm's 100 MB limit. Requests are paced below both services' limits (GetSongBPM allows 3,000 an hour), so repeating a search is instant. Deezer requests stay under its 50 per 5 seconds, and previews are fetched only from Deezer's CDN. Coverage depends on the services: brand-new or unreleased tracks may be missing from Last.fm or Deezer. Beatport's API is partner-only, and SoundCloud's requires an Artist Pro subscription, so neither is used yet.

### `feedback`: log how transitions sounded on real decks

```bash
uv run setsmith feedback log ~/Music/rekordbox.xml good \
    --from "Kaya Sol - Sunwater" --to "Moana Reyes - Lanterns" --note "chant lines up"
uv run setsmith feedback log ~/Music/rekordbox.xml bad --from-id 101 --to-id 104 --type cut --bars 8
uv run setsmith feedback list
```

Each entry stores your verdict next to Setsmith's full score breakdown for that pair at the time, for later weight tuning. `--type` and `--bars` record what you actually played; they default to the suggestion. `feedback list` shows how Setsmith's average score compares for the transitions you liked and the ones you didn't. See [docs/listening-checklist.md](docs/listening-checklist.md) for a routine.

## How scoring works

Each pair is scored from track A (playing) to track B (incoming), so scores are directional: +2 on the Camelot wheel is an energy boost, but −2 is not. The total is 100 × the weighted sum of five components:

| Component | Weight | Summary |
|---|---|---|
| harmonic | 0.30 | Same key 1.0, ±1 0.9, relative 0.85, +2 boost 0.65, diagonal 0.5, +7 semitone lift 0.5, clash 0.1. Pulled toward 0.5 when key confidence is low. |
| tempo | 0.25 | Pitch change B needs to match A, also checking half and double time: ≤2% 1.0, ≤4% 0.8, ≤6% 0.5, ≤8% 0.2, above that 0. |
| energy | 0.20 | Distance between the actual energy change and the target: ≤1 1.0, ≤2 0.7, ≤3 0.4, above that 0.1. |
| genre_style | 0.15 | Same genre 1.0, neighboring genre 0.7 (for example afro house and organic house), otherwise 0.3. |
| extras | 0.10 | Average of vocal clash, B's star rating, and intro/outro length fit, using whichever are known. |

Missing data never causes a crash. A missing component scores a neutral 0.5 and adds a flag such as `missing_key`.

Every number above lives in [`setsmith/scoring/weights.py`](setsmith/scoring/weights.py). Genre aliases and neighbor groups are also there.

**Suggested transition types:**

- **cut** or **echo_out** (8 bars) for key clashes, semitone lifts, and tempo gaps over 6%.
- **drop_swap** (16 bars) for energy jumps of +3 or more.
- **long_blend** (32 bars) when the tempo is within 4%, the harmonic score is at least 0.85, and the intro and outro are long enough.
- **filter_sweep** (16 bars) otherwise.

### Where energy comes from

Rekordbox has no energy field. Setsmith uses the first of these that exists:

1. A tag in Comments: `E7`, `Energy 7`, or Mixed In Key's `8A - Energy 7`.
2. Audio analysis (`setsmith analyze`), calibrated to your library.
3. The star rating × 2 (for example, 3 stars becomes E6).
4. Otherwise unknown (neutral score, flag `missing_energy`).

### Key notation

Setsmith reads Camelot (`8A`), Open Key (`1m`), and classic notation (`Am`, `F#m`, `Abm`, `C`, `Db`, `A minor`, `G♯m`), including all enharmonic spellings. Key tags from Rekordbox are treated as noisy. When `analyze` detects a different key from the audio, the track's key confidence drops and its harmonic scores move toward neutral.

## How set building works

1. **Pool.** Setsmith applies your filters and leaves out tracks without a BPM (it warns you about them).
2. **Graph.** Only pairs within ±8% tempo, including half and double time, are candidates. Each pair's score, minus the energy component, is computed on demand and cached. Setsmith first ranks candidates by a cheap ceiling that uses the exact tempo and key scores with genre and extras assumed perfect, and fully scores only pairs that could still make the beam.
3. **Beam search** (width 20). At position *i* the energy target is chosen so the next track lands on the curve: target change = curve(*i*) − current energy. Each step scores the transition plus these penalties:

| Penalty | Points |
|---|---|
| Artist repeats within 4 tracks | −15 |
| Same label 3 times in a row | −5 |
| More than 2 energy-boost key moves (+2) within 6 tracks | −10 |
| Tempo range over the set grows beyond 8 BPM | −20 |
| Tempo drops more than 3 BPM while the curve rises | −10 |

With `--style`, each step also earns style fit × 15 points (see [`styles`](#styles-dj-style-profiles)).

The same song is never used twice, even when your collection holds duplicate entries of it. Half- and double-time tracks are folded into the opening track's tempo range before drift is measured.

4. **Alternates.** For each position, Setsmith picks the two best unused tracks that fit both neighbors and stay within 2 energy points of the curve.

All of these numbers live in `SetConfig` and `CurveTemplate` in [`setsmith/scoring/weights.py`](setsmith/scoring/weights.py).

## Data Setsmith stores

`analyze` (analysis results), `rekordbox import` (My Tags, history), `liveset` (analyzed sets), `learn` (learned weights), `discover` (7-day cache of API responses; 180 days for BPM/key estimated from previews), `build` (pair-score cache) and `feedback` use one SQLite file: `$SETSMITH_DB` if set, else `~/.local/share/setsmith/setsmith.db` (or under `$XDG_DATA_HOME`). Pass `--db PATH` to use another file, or `--no-cache` to skip the cache for one build. Commands that only read (`suggest`, `info`) never create the file. Cached scores are keyed by a fingerprint of each track's tags plus the scoring config, so editing a track's tags or changing a weight recomputes only what changed. The file holds numbers and your notes, never audio. Delete it at any time to start fresh; your feedback is in the same file, so back it up first if you want to keep it.

## Importing sets into Rekordbox

`setsmith build --out setsmith.xml` writes your sets into a new XML file, inside a **Setsmith** playlist folder. To load it in Rekordbox:

1. **Preferences > View**: enable **rekordbox xml** in the tree view.
2. **Preferences > Advanced > Database > rekordbox xml**: set the path to the file Setsmith generated.
3. In the sidebar, open **rekordbox xml > Setsmith**, right-click a playlist, and choose **Import Playlist**.

The exported file copies every track entry exactly as it appears in your export: all attributes, beat grid (TEMPO) and cues (POSITION_MARK), with Location untouched. Rekordbox matches the tracks to the files already in your library.

## Development

```bash
uv sync --extra audio --extra essentia --extra rekordbox --extra web   # those tests skip without these
uv run pytest                  # all tests, including performance and audio tests
uv run pytest -m "not perf and not audio"   # fast core tests
uv run ruff check . && uv run ruff format --check .
uv run mypy setsmith tests
uv run python tests/synth.py   # regenerate tests/fixtures/collection_50.xml
```

Test fixtures use made-up artists and titles. Audio tests analyze tracks synthesized on the fly by `tests/audio_synth.py` (a 124 BPM A-minor house loop with known intro and outro lengths), so no real recordings are involved.

## Guardrails

- Read-only access to your library. Output always goes to new files: Setsmith never writes to its input and only replaces files it created itself (or any file with `--force`).
- No Spotify Web API audio features, which are unavailable to new apps since 27 November 2024. No scraping of 1001Tracklists: tracklists are pasted by you, and `TracklistProvider` in `setsmith/analysis/tracklist.py` is the extension point for a licensed source. No downloading from streaming platforms: tracks are never saved. The one exception to local-only analysis is Deezer's official 30-second previews, fetched from its public API and analyzed in memory for BPM and key (see [`discover`](#bpm-and-key-for-songs-outside-your-library)); the audio is never written to disk. Otherwise only local files you provide are analyzed, including set recordings you are entitled to process. Audio fingerprinting services such as ACRCloud are not integrated; if added, they would run only with your own credentials.
- Only derived features are stored. Audio and full tracklists are never redistributed. Audio files are opened read-only and never copied; previews are never stored.
- Style profiles describe musical parameters in our own words. They are labeled "inspired by" and do not imply endorsement by any artist.
- Rekordbox's own database is only ever read by copying it, behind `--backed-up`, with a key you supply.

## License

MIT; see [LICENSE](LICENSE). The optional Essentia extra is AGPL-3.0, and the optional madmom/allin1 models carry their own non-commercial licenses (see above).
