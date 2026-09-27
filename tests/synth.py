"""Synthetic Rekordbox XML collections for golden and performance tests.

Run as a script to regenerate tests/fixtures/collection_50.xml:

    uv run python tests/synth.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

CLASSIC_KEYS = [
    "Am", "Em", "Bm", "F#m", "Dbm", "Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm",
    "C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F",
]  # fmt: skip
GENRES = [
    "Afro House", "Melodic House & Techno", "Organic House", "Deep House",
    "House", "Tech House", "Minimal / Deep Tech", "Techno", "Indie Dance",
]  # fmt: skip
ARTIST_WORDS = ["Kaya", "Nils", "Moana", "Dex", "Ines", "Tomas", "Rafa", "Juno", "Oli", "Sade"]
SURNAMES = ["Sol", "Reyes", "Marlo", "Brandt", "Okafor", "Lind", "Vance", "Noor", "Keel"]
TITLE_WORDS = ["Tidal", "Ember", "Lantern", "Drift", "Pulse", "Harbour", "Signal", "Velvet",
               "Orbit", "Dust", "Canopy", "Mirage", "Static", "Bloom", "Echo", "Fable"]  # fmt: skip


@dataclass
class SynthTrack:
    id: str
    artist: str
    title: str
    bpm: float
    key: str
    genre: str
    energy: int | None = None
    rating: int = 0
    tempo_markers: list[float] = field(default_factory=list)
    extra_comments: str = ""
    location: str | None = None  # default: a made-up path under /Users/dj/Music
    total_time: int = 384

    def to_xml(self) -> str:
        comments = " ".join(
            p for p in (f"E{self.energy}" if self.energy else "", self.extra_comments) if p
        )
        slug = f"{self.artist} - {self.title}".replace(" ", "%20")
        attrs = {
            "TrackID": self.id, "Name": self.title, "Artist": self.artist, "Composer": "",
            "Album": "", "Grouping": "", "Genre": self.genre, "Kind": "MP3 File",
            "Size": "12000000", "TotalTime": str(self.total_time), "DiscNumber": "0",
            "TrackNumber": "0",
            "Year": "2024", "AverageBpm": f"{self.bpm:.2f}", "DateAdded": "2024-06-01",
            "BitRate": "320", "SampleRate": "44100", "Comments": comments, "PlayCount": "0",
            "Rating": str(self.rating * 51),
            "Location": self.location or f"file://localhost/Users/dj/Music/Synth/{slug}.mp3",
            "Remixer": "", "Tonality": self.key, "Label": "Synthetic", "Mix": "",
        }  # fmt: skip
        attr_text = " ".join(f"{k}={quoteattr(v)}" for k, v in attrs.items())
        bpms = self.tempo_markers or [self.bpm]
        children = "".join(
            f'\n      <TEMPO Inizio="{0.05 + 60 * i:.3f}" Bpm="{b:.2f}" Metro="4/4" Battito="1"/>'
            for i, b in enumerate(bpms)
        )
        children += '\n      <POSITION_MARK Name="" Type="0" Start="0.050" Num="-1"/>'
        return f"    <TRACK {attr_text}>{children}\n    </TRACK>"


def write_collection(path: Path, tracks: list[SynthTrack]) -> None:
    body = "\n".join(t.to_xml() for t in tracks)
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="rekordbox" Version="7.0.4" Company="AlphaTheta"/>
  <COLLECTION Entries="{len(tracks)}">
{body}
  </COLLECTION>
  <PLAYLISTS>
    <NODE Type="0" Name="ROOT" Count="0"/>
  </PLAYLISTS>
</DJ_PLAYLISTS>
""",
        encoding="utf-8",
    )


def random_tracks(
    n: int,
    *,
    start_id: int = 1,
    seed: int = 7,
    bpm_ranges: tuple[tuple[float, float], ...] = ((100.0, 140.0),),
) -> list[SynthTrack]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        lo, hi = rng.choice(bpm_ranges)
        missing = rng.random() < 0.05
        out.append(
            SynthTrack(
                id=str(start_id + i),
                artist=f"{rng.choice(ARTIST_WORDS)} {rng.choice(SURNAMES)}",
                title=f"{rng.choice(TITLE_WORDS)} {rng.choice(TITLE_WORDS)} {start_id + i}",
                bpm=0.0 if missing else round(rng.uniform(lo, hi), 2),
                key="" if missing else rng.choice(CLASSIC_KEYS),
                genre=rng.choice(GENRES),
                energy=rng.randint(1, 10) if rng.random() < 0.7 else None,
                rating=rng.randint(0, 5),
            )
        )
    return out


def golden_50() -> list[SynthTrack]:
    """Two hand-built clusters with known best matches, padded with filler.

    Filler BPMs (95-110, 138-145) are more than 6% from both clusters (122 and 128 BPM)
    and not half/double of either, so no filler track can outscore the crafted matches.
    """
    crafted = [
        # Cluster A, seed 1: 122 BPM, A minor, E6, afro house
        SynthTrack("1", "Kaya Sol", "Tidal Drum", 122.0, "Am", "Afro House", 6, 4),
        SynthTrack("2", "Moana Reyes", "Canopy", 122.0, "Am", "Afro House", 6, 5),  # 100
        SynthTrack("3", "Ines Okafor", "Lantern Song", 123.0, "Em", "Afro House", 7, 5),  # 97
        SynthTrack("4", "Tomas Lind", "Velvet Drift", 121.0, "C", "Organic House", 6, 4),  # 89
        SynthTrack(
            "5", "Kaya Sol", "Tidal Drum", 122.0, "Am", "Afro House", 6, 4
        ),  # duplicate of 1
        # Cluster B, seed 6: 128 BPM, G minor, E8, tech house
        SynthTrack("6", "Dex Marlo", "Snare Theory", 128.0, "Gm", "Tech House", 8, 4),
        SynthTrack("7", "Rafa Vance", "Static Bloom", 128.0, "6A", "Tech House", 8, 5),  # 100
        SynthTrack("8", "Juno Brandt", "Signal", 129.0, "Dm", "House", 8, 4),  # 90.5
    ]
    filler = random_tracks(
        50 - len(crafted), start_id=len(crafted) + 1, seed=50,
        bpm_ranges=((95.0, 110.0), (138.0, 145.0)),
    )  # fmt: skip
    return crafted + filler


if __name__ == "__main__":
    target = Path(__file__).parent / "fixtures" / "collection_50.xml"
    write_collection(target, golden_50())
    print(f"wrote {target}")
