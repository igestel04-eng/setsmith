# Listening checklist

Scores are only as good as the weights behind them. This routine turns your ears into data that later weight tuning can use.

## Before a session

1. Export a fresh collection from Rekordbox (**File > Export Collection in xml format**).
2. Pick about 20 transitions to test. Two good sources:
   - A built set: `setsmith build rekordbox.xml --tracks 21 --report session.md`. That gives you 20 transitions, and the report lists each one with its reasoning.
   - Seeds you know well: `setsmith suggest rekordbox.xml -t "Artist - Title" --top 5 --explain`.
3. Include a few lower scores (50-70) as well as the top ones. Feedback on middling pairs teaches the most.

## For each transition

Play it on real decks the way Setsmith suggests: same transition type, same length in bars, on a phrase boundary. Then ask:

- **Keys:** any clash while both tracks were audible?
- **Tempo:** did the pitch change sound unnatural, or did the grids drift?
- **Energy:** did the lift or drop feel right for that point in the set?
- **Bass:** did the bass swap land cleanly on the phrase?
- **Vocals:** did two vocals fight?

Log it straight away:

```bash
setsmith feedback log rekordbox.xml good --from "A - Title" --to "B - Title"
setsmith feedback log rekordbox.xml bad  --from-id 101 --to-id 104 --note "vocal clash at the drop"
```

If you played a different transition than suggested, record what you actually did with `--type cut --bars 8`.

## After the session

```bash
setsmith feedback list
```

If the transitions you called "bad" score about as high as the "good" ones, the weights are off. The `--json` output includes every component score, so you can see which component misled the scorer, for example harmonic scores that are high on pairs you rejected for key reasons.
