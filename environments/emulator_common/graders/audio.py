import zlib
from collections.abc import Iterable


def audio_crc(samples: bytes | bytearray | Iterable[int]) -> str:
    if isinstance(samples, (bytes, bytearray)):
        payload = bytes(samples)
    else:
        payload = bytes(int(sample) & 0xFF for sample in samples)
    return f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
