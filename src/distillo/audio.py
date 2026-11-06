"""Shrink and split recordings so they fit an upload limit.

Speech survives aggressive compression: mono, 16 kHz, Opus at 24 kbit/s is about 11 MB per
hour and still transcribes well. Recordings longer than ``part_seconds`` are split into
parts, each with the offset at which it starts, so timestamps can be put back together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16_000
BIT_RATE = 24_000


class AudioError(RuntimeError):
    pass


@dataclass(frozen=True)
class Part:
    path: Path
    offset: float


def prepare_for_upload(
    source: Path, workdir: Path, *, max_bytes: int, part_seconds: float = 3600.0
) -> list[Part]:
    """Return ``source`` itself if it is small enough, otherwise compressed parts."""
    if source.stat().st_size <= max_bytes:
        return [Part(source, 0.0)]
    parts = compress(source, workdir, part_seconds=part_seconds)
    too_big = [p.path.name for p in parts if p.path.stat().st_size > max_bytes]
    if too_big:  # pragma: no cover - would need parts far longer than an hour
        raise AudioError(f"still over the upload limit after compression: {', '.join(too_big)}")
    return parts


def compress(source: Path, workdir: Path, *, part_seconds: float = 3600.0) -> list[Part]:
    """Decode any audio or video file and write mono Opus parts of at most ``part_seconds``."""
    try:
        import av
    except ImportError as exc:  # pragma: no cover - av is a core dependency
        raise AudioError("compressing audio needs PyAV: pip install av") from exc

    workdir.mkdir(parents=True, exist_ok=True)
    parts: list[Part] = []
    try:
        with av.open(str(source)) as container:
            if not container.streams.audio:
                raise AudioError(f"{source.name} has no audio track")
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            writer: _PartWriter | None = None
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    continue
                seconds = float(frame.pts * frame.time_base)
                if writer is None or seconds - writer.offset >= part_seconds:
                    if writer is not None:
                        parts.append(writer.close())
                    offset = seconds if writer is not None else 0.0
                    writer = _PartWriter(
                        workdir / f"{source.stem}.part{len(parts) + 1}.ogg", offset
                    )
                for resampled in resampler.resample(frame):
                    writer.write(resampled)
            if writer is not None:
                for resampled in resampler.resample(None):
                    writer.write(resampled)
                parts.append(writer.close())
    except AudioError:
        raise
    except Exception as exc:  # PyAV raises many error types for unreadable media
        raise AudioError(f"cannot read the audio in {source.name}: {exc}") from exc
    if not parts:
        raise AudioError(f"no audio decoded from {source.name}")
    return parts


class _PartWriter:
    def __init__(self, path: Path, offset: float) -> None:
        import av

        self.path = path
        self.offset = offset
        self.container: Any = av.open(str(path), mode="w", format="ogg")
        self.stream: Any = self.container.add_stream("libopus", rate=SAMPLE_RATE)
        self.stream.bit_rate = BIT_RATE
        self.stream.layout = "mono"

    def write(self, frame: Any) -> None:
        frame.pts = None  # let the encoder number samples from the start of this part
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> Part:
        for packet in self.stream.encode(None):
            self.container.mux(packet)
        self.container.close()
        return Part(self.path, self.offset)
