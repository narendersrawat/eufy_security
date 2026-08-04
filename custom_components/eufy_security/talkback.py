"""Eufy talkback audio helpers."""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from .eufy_security_api.camera import StreamStatus
from collections.abc import Iterator
import logging
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

# AAC-LC uses 1024 samples per frame.
# At 16 kHz, each frame represents exactly 64 milliseconds.
AAC_FRAME_DURATION_SECONDS = 1024 / 16000

TALKBACK_START_TIMEOUT_SECONDS = 5.0


class InvalidAdtsStreamError(ValueError):
    """Raised when a file is not a valid AAC/ADTS stream."""


def iter_adts_frames(data: bytes) -> Iterator[bytes]:
    """Yield complete AAC frames, including each ADTS header."""
    offset = 0
    data_length = len(data)

    while offset < data_length:
        if data_length - offset < 7:
            raise InvalidAdtsStreamError(
                f"Incomplete ADTS header at byte {offset}"
            )

        # ADTS syncword is twelve 1 bits: 0xFFF.
        if data[offset] != 0xFF or (data[offset + 1] & 0xF6) != 0xF0:
            raise InvalidAdtsStreamError(
                f"Invalid ADTS syncword at byte {offset}"
            )

        protection_absent = data[offset + 1] & 0x01
        header_length = 7 if protection_absent else 9

        frame_length = (
            ((data[offset + 3] & 0x03) << 11)
            | (data[offset + 4] << 3)
            | ((data[offset + 5] & 0xE0) >> 5)
        )

        if frame_length < header_length:
            raise InvalidAdtsStreamError(
                f"Invalid ADTS frame length {frame_length} at byte {offset}"
            )

        frame_end = offset + frame_length
        if frame_end > data_length:
            raise InvalidAdtsStreamError(
                f"Incomplete ADTS frame at byte {offset}"
            )

        yield data[offset:frame_end]
        offset = frame_end

def _extract_next_adts_frame(data: bytes) -> tuple[bytes | None, bytes]:
    """Extract one complete ADTS frame from buffered stream data."""
    if len(data) < 7:
        return None, data

    if data[0] != 0xFF or (data[1] & 0xF6) != 0xF0:
        raise InvalidAdtsStreamError("Invalid ADTS syncword in live stream")

    protection_absent = data[1] & 0x01
    header_length = 7 if protection_absent else 9

    frame_length = (
        ((data[3] & 0x03) << 11)
        | (data[4] << 3)
        | ((data[5] & 0xE0) >> 5)
    )

    if frame_length < header_length:
        raise InvalidAdtsStreamError(
            f"Invalid ADTS frame length: {frame_length}"
        )

    if len(data) < frame_length:
        return None, data

    return data[:frame_length], data[frame_length:]

class TalkbackSession:
    """Play an AAC/ADTS file through a Eufy camera speaker."""

    def __init__(self, camera: Any) -> None:
        """Initialize the player with the API camera object."""
        self._camera = camera
        self._play_lock = asyncio.Lock()

    async def play_file(self, file_path: Path) -> None:
        """Play an AAC/ADTS file through a Eufy camera speaker."""
        audio_data = await asyncio.to_thread(file_path.read_bytes)
        await self.play_frames(iter_adts_frames(audio_data))

    async def play_frames(self, frames: Iterator[bytes]) -> None:
        """Play an iterator of AAC/ADTS frames through the camera speaker."""
        async with self._play_lock:
            frames = list(frames)

            if not frames:
                raise InvalidAdtsStreamError("No AAC/ADTS frames found")

            _LOGGER.debug(
                "Starting talkback playback: %s frames",
                len(frames),
            )

            await self.start()

            try:
                await self._send_frames(frames)
            finally:
                try:
                    await self.stop()
                except Exception:
                    _LOGGER.exception("Unable to stop Eufy talkback session")

    async def start(self) -> None:
        """Start the livestream and talkback session."""
        if self._camera.stream_status != StreamStatus.STREAMING:
            await self._camera.start_livestream()

        await self._camera.start_talkback()
        await self._wait_until_started()

    async def stop(self) -> None:
        """Stop the talkback and livestream session."""
        try:
            await self._camera.stop_talkback()
        finally:
            await self._camera.stop_livestream()

    async def play_stream(self, reader: asyncio.StreamReader) -> None:
        """Forward a live AAC/ADTS byte stream to the camera speaker."""
        async with self._play_lock:
            await self.start()
            try:
                buffer = b""
                while True:
                    chunk = await reader.read(1024)

                    if not chunk:
                        break
                    buffer += chunk

                    while True:
                        frame, buffer = _extract_next_adts_frame(buffer)

                        if frame is None:
                            break

                        await self._camera.send_talkback_audio(frame)
            finally:
                try:
                    await self.stop()
                except Exception:
                    _LOGGER.exception("Unable to stop Eufy talkback session")

    async def stream_wav_file(self, file_path: Path) -> None:
        """Encode a WAV file to AAC/ADTS using FFmpeg and stream it live."""

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg executable not found")

        process = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(file_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "aac",
            "-b:a",
            "20k",
            "-f",
            "adts",
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            assert process.stdout is not None
            await self.play_stream(process.stdout)
        finally:
            with contextlib.suppress(ProcessLookupError):
                process.kill()

            await process.wait()

    async def _wait_until_started(self) -> None:
        """Wait until the websocket server reports active talkback."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + TALKBACK_START_TIMEOUT_SECONDS

        while loop.time() < deadline:
            if await self._camera.is_talkback_ongoing():
                return

            await asyncio.sleep(0.1)

        raise TimeoutError(
            "Eufy talkback did not start within "
            f"{TALKBACK_START_TIMEOUT_SECONDS} seconds"
        )

    async def _send_frames(self, frames: list[bytes]) -> None:
        """Send AAC frames at their natural 16-kHz playback rate."""
        loop = asyncio.get_running_loop()
        next_frame_time = loop.time()

        for frame in frames:
            await self._camera.send_talkback_audio(frame)

            next_frame_time += AAC_FRAME_DURATION_SECONDS
            delay = next_frame_time - loop.time()

            if delay > 0:
                await asyncio.sleep(delay)