"""Input-image handling: data URLs and remote URLs -> validated data URL.

Remote fetches are guarded against SSRF: only http(s), every resolved address of
every redirect hop must be public, size is capped while streaming, and the bytes
must actually be a PNG/JPEG/WEBP (content-type headers are not trusted).
Residual risk: DNS rebinding between our check and httpx's own resolution. It is
limited because only authenticated API keys can trigger fetches; set
ALLOW_REMOTE_IMAGES=false if your keys are shared with untrusted parties.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import re
import socket
from urllib.parse import urljoin, urlsplit

import httpx

from .config import Settings
from .errors import bad_request

_DATA_URL = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?P<params>(;[^,;]*)*?);base64,(?P<data>.*)$", re.S)


def sniff_mime(raw: bytes) -> str | None:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def to_data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
        or ip.is_reserved or ip.is_unspecified
    )


class ImageFetcher:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.image_fetch_timeout),
            follow_redirects=False,
            transport=transport,
            headers={"User-Agent": "glm-bridge/0.1 (+image-fetch)"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def resolve(self, host: str) -> list[str]:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        return [info[4][0] for info in infos]

    async def _check_host(self, host: str | None) -> None:
        if not host:
            raise bad_request("Image URL has no host", code="invalid_image_url")
        try:
            addresses = [host] if ipaddress.ip_address(host) else []
        except ValueError:
            try:
                addresses = await self.resolve(host)
            except OSError as exc:
                raise bad_request(f"Could not resolve image host '{host}'", code="invalid_image_url") from exc
        if not addresses:
            raise bad_request(f"Could not resolve image host '{host}'", code="invalid_image_url")
        for addr in addresses:
            if not _is_public(ipaddress.ip_address(addr.split("%")[0])):
                raise bad_request("Image URL points to a non-public address", code="invalid_image_url")

    def decode_data_url(self, url: str) -> str:
        m = _DATA_URL.match(url.strip())
        if not m:
            raise bad_request("Malformed image data URL (expected data:<mime>;base64,...)", code="invalid_image")
        payload = re.sub(r"\s+", "", m.group("data"))
        if len(payload) * 3 // 4 > self.settings.max_image_bytes + 3:
            raise self._too_big()
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise bad_request("Image data URL is not valid base64", code="invalid_image") from exc
        return self._validated(raw)

    def from_bytes(self, raw: bytes) -> str:
        return self._validated(raw)

    def _validated(self, raw: bytes) -> str:
        if len(raw) > self.settings.max_image_bytes:
            raise self._too_big()
        mime = sniff_mime(raw)
        if mime is None:
            raise bad_request("Unsupported image format; use PNG, JPEG or WEBP", code="invalid_image")
        return to_data_url(raw, mime)

    def _too_big(self):
        mb = self.settings.max_image_bytes / (1024 * 1024)
        return bad_request(f"Image is larger than {mb:g} MB", code="image_too_large")

    async def fetch_bytes(self, url: str, *, max_hops: int = 3) -> bytes:
        current = url
        for _ in range(max_hops + 1):
            parts = urlsplit(current)
            if parts.scheme not in ("http", "https"):
                raise bad_request("Image URL must be http(s) or a data: URL", code="invalid_image_url")
            await self._check_host(parts.hostname)
            try:
                async with self._client.stream("GET", current) as resp:
                    if resp.is_redirect and "location" in resp.headers:
                        current = urljoin(current, resp.headers["location"])
                        continue
                    if resp.status_code != 200:
                        raise bad_request(f"Fetching image failed with HTTP {resp.status_code}", code="image_fetch_failed")
                    buf = bytearray()
                    async for chunk in resp.aiter_bytes():
                        buf.extend(chunk)
                        if len(buf) > self.settings.max_image_bytes:
                            raise self._too_big()
                    return bytes(buf)
            except httpx.HTTPError as exc:
                raise bad_request(f"Fetching image failed: {exc.__class__.__name__}", code="image_fetch_failed") from exc
        raise bad_request("Too many redirects while fetching image", code="image_fetch_failed")

    async def to_data_url(self, url: str) -> str:
        url = (url or "").strip()
        if url.startswith("data:"):
            return self.decode_data_url(url)
        if url.startswith(("http://", "https://")):
            if not self.settings.allow_remote_images:
                raise bad_request("Remote image URLs are disabled; send a base64 data URL", code="remote_images_disabled")
            return self._validated(await self.fetch_bytes(url))
        raise bad_request("Image must be a data: URL or an http(s) URL", code="invalid_image_url")
