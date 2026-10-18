from __future__ import annotations

import base64
import io
import mimetypes
from collections.abc import Sequence
from pathlib import Path

try:
    from PIL import Image  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


ImageInput = str | Path | bytes | bytearray | object


def build_user_content(
    text: str, images: Sequence[ImageInput] | None = None
) -> list[dict[str, object]]:
    """Builds a 'content' payload for the user message with text and optional images."""

    parts: list[dict[str, object]] = [{"type": "text", "text": text}]

    if not images:
        return parts

    for img in images:
        url = image_input_to_url(img, encode_format="PNG")
        parts.append({"type": "image_url", "image_url": {"url": url}})

    return parts


def image_input_to_url(img: ImageInput, *, encode_format: str = "PNG") -> str:
    """Normalizes supported image inputs into a URL suitable for image_url payloads."""

    if isinstance(img, (str, Path)):
        s = str(img)
        if s.startswith(("http://", "https://", "data:")):
            return s
        return file_to_data_url(s)

    if isinstance(img, (bytes, bytearray)):
        mime = _sniff_mime(bytes(img))
        return _bytes_to_data_url(bytes(img), mime)

    if Image is not None and isinstance(img, Image.Image):  # type: ignore[attr-defined]
        return pil_to_data_url(img, format=encode_format)

    if np is not None and isinstance(img, np.ndarray):  # type: ignore[attr-defined]
        return cv2_to_data_url(img, format=encode_format, assume_bgr=True)

    raise TypeError(
        f"Unsupported image input type: {type(img)!r}. "
        "Supported: str/Path/url/data-url, bytes, PIL.Image.Image, numpy.ndarray"
    )


def file_to_data_url(path: str | Path) -> str:
    """Reads a local file and converts it into a data URL."""

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")

    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or "application/octet-stream"
    return _bytes_to_data_url(p.read_bytes(), mime)


def pil_to_data_url(img: object, *, format: str = "PNG") -> str:
    """Encodes a PIL image into a data URL."""

    if Image is None:
        raise RuntimeError("PIL is not available. Install pillow to pass PIL images.")

    if not isinstance(img, Image.Image):  # type: ignore[attr-defined]
        raise TypeError(f"Expected PIL.Image.Image, got {type(img)!r}")

    bio = io.BytesIO()
    img.save(bio, format=format)
    mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return _bytes_to_data_url(bio.getvalue(), mime)


def cv2_to_data_url(arr: object, *, format: str = "PNG", assume_bgr: bool = True) -> str:
    """Encodes a numpy image array into a data URL via PIL."""

    if np is None:
        raise RuntimeError("numpy is not available. Install numpy to pass cv2 images.")
    if Image is None:
        raise RuntimeError("PIL is not available. Install pillow to pass cv2 images.")

    if not isinstance(arr, np.ndarray):  # type: ignore[attr-defined]
        raise TypeError(f"Expected numpy.ndarray, got {type(arr)!r}")

    if arr.ndim == 2:
        pil_img = Image.fromarray(arr)
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        if arr.shape[2] == 3:
            rgb = arr[..., ::-1] if assume_bgr else arr
            pil_img = Image.fromarray(rgb, mode="RGB")
        else:
            rgba = arr[..., [2, 1, 0, 3]] if assume_bgr else arr
            pil_img = Image.fromarray(rgba, mode="RGBA")
    else:
        raise ValueError(f"Unsupported array shape for image: {arr.shape}")

    return pil_to_data_url(pil_img, format=format)


def _bytes_to_data_url(data: bytes, mime: str) -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _sniff_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"
