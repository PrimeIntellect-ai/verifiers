import hashlib
from collections.abc import Iterable


def framebuffer_sha256(pixels: bytes | bytearray | Iterable[int]) -> str:
    if isinstance(pixels, (bytes, bytearray)):
        payload = bytes(pixels)
    else:
        payload = bytes(int(pixel) & 0xFF for pixel in pixels)
    return hashlib.sha256(payload).hexdigest()


def frame_sequence_sha256(frames: Iterable[bytes | bytearray | Iterable[int]]) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        if isinstance(frame, (bytes, bytearray)):
            payload = bytes(frame)
        else:
            payload = bytes(int(pixel) & 0xFF for pixel in frame)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()
