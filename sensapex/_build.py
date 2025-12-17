"""Custom setuptools commands for downloading required Sensapex binaries."""

from __future__ import annotations

import os
from io import BytesIO
import platform
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse
from zipfile import ZipFile
import urllib.request

from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install


if platform.system() == "Windows":
    UMSDK_URL = "https://github.com/sensapex/umsdk/releases/download/v1.504/umsdk_Windows_X64_gcc_v1.504.3_Release.zip"
    UMSDK_MEMBERS = ["bin/um.dll"]
elif platform.system() == "Darwin" and platform.machine() == "arm64":
    UMSDK_URL = "https://github.com/sensapex/umsdk/releases/download/v1.504/umsdk_macOS_ARM64_gcc_v1.504.3_Release.zip"
    UMSDK_MEMBERS = ["bin/shared/libum.dylib"]
elif platform.system() == "Linux" and platform.machine() == "x86_64":
    UMSDK_URL = "https://github.com/sensapex/umsdk/releases/download/v1.504/umsdk_Linux_X64_gcc_v1.504.3_Release.zip"
    UMSDK_MEMBERS = ["bin/shared/libum.so"]
else:
    UMSDK_URL = None  # Unsupported platform

UMSDK_ENV = "SENSAPEX_UMSDK_ARCHIVE"

UMPCLI_URL = "http://dist.sensapex.com/misc/umpcli/umpcli-0_957-beta.zip"
UMPCLI_MEMBERS = ["umpcli.exe"]
UMPCLI_ENV = "SENSAPEX_UMPCLI_ARCHIVE"

FORCE_BINARIES_ENV = "SENSAPEX_FORCE_WINDOWS_BINARIES"

CACHE_DIR = Path(
    os.environ.get(
        "SENSAPEX_DRIVER_CACHE",
        Path.home() / ".cache" / "sensapex",
    )
)
REPO_ROOT = Path(__file__).resolve().parent.parent


class DownloadBinariesAndInstall(install):
    """pip install ."""

    def run(self):
        super().run()
        install_bin(Path(self.install_purelib) / "sensapex", force=_force_binary_download())


class DownloadBinariesAndDevelop(develop):
    """pip install -e ."""

    def run(self):
        super().run()
        install_bin(Path(self.egg_path) / "sensapex", force=_force_binary_download())


class DownloadBinariesAndBuild(build_py):
    """python -m build"""

    def run(self):
        super().run()
        install_bin(Path(self.build_lib) / "sensapex", force=_force_binary_download())


def install_bin(path: Path, force: bool = False) -> None:
    """Install libum.dll and umpcli.exe to *path*."""
    if not force and platform.system() != "Windows":
        return

    path.mkdir(parents=True, exist_ok=True)

    dll_data = download_from_zip(UMSDK_URL, UMSDK_MEMBERS, env_var=UMSDK_ENV)[0]
    (path / "libum.dll").write_bytes(dll_data)

    umpcli_data = download_from_zip(UMPCLI_URL, UMPCLI_MEMBERS, env_var=UMPCLI_ENV)[0]
    (path / "umpcli.exe").write_bytes(umpcli_data)


def download_from_zip(url: str, files: List[str], env_var: str | None = None) -> List[bytes]:
    content_file = BytesIO(_get_archive_bytes(url, env_var))
    data = []
    with ZipFile(content_file, "r") as zip_file:
        for filename in files:
            with zip_file.open(filename) as req_file:
                data.append(req_file.read())
    return data


def _get_archive_bytes(url: str, env_var: str | None) -> bytes:
    filename = Path(urlparse(url).path).name
    for candidate in _archive_sources(filename, url, env_var):
        if isinstance(candidate, Path):
            if candidate.is_file():
                return candidate.read_bytes()
            raise FileNotFoundError(f"Configured archive {candidate} does not exist.")

        data = _download_url(candidate)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / filename).write_bytes(data)
        return data

    raise RuntimeError(f"Unable to fetch archive for {url}")


def _archive_sources(filename: str, url: str, env_var: str | None) -> Iterable[Path | str]:
    if env_var:
        explicit = os.environ.get(env_var)
        if explicit:
            yield Path(explicit).expanduser()

    repo_candidate = REPO_ROOT / filename
    if repo_candidate.exists():
        yield repo_candidate

    cache_candidate = CACHE_DIR / filename
    if cache_candidate.exists():
        yield cache_candidate

    yield url


def _download_url(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=60) as req:
            return req.read()
    except Exception as exc:  # pragma: no cover - best effort error reporting
        raise RuntimeError(f"Unable to download {url}: {exc}") from exc


def _force_binary_download() -> bool:
    """Return True when downloads should be attempted on non-Windows hosts."""
    env_value = os.environ.get(FORCE_BINARIES_ENV)
    if env_value is None:
        return False

    return env_value.strip().lower() in {"1", "true", "yes", "on"}
