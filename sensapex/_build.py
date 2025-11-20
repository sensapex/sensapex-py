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


UMSDK_DLL_URL = "https://github.com/sensapex/umsdk/releases/download/v1.400/umsdk-1.400-binaries.zip"
UMSDK_DLL_MEMBERS = ["umsdk-1.400-binaries/x64/libum.dll"]
UMSDK_ENV = "SENSAPEX_UMSDK_ARCHIVE"

UMPCLI_URL = "http://dist.sensapex.com/misc/umpcli/umpcli-0_957-beta.zip"
UMPCLI_MEMBERS = ["umpcli.exe"]
UMPCLI_ENV = "SENSAPEX_UMPCLI_ARCHIVE"

FORCE_BINARIES_ENV = "SENSAPEX_FORCE_WINDOWS_BINARIES"
SKIP_BUILD_BINARIES_ENV = "SENSAPEX_SKIP_BUILD_WINDOWS_BINARIES"

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
        install_bin(Path(self.install_purelib) / "sensapex", force=_should_install_runtime_binaries())


class DownloadBinariesAndDevelop(develop):
    """pip install -e ."""

    def run(self):
        super().run()
        install_bin(Path(self.egg_path) / "sensapex", force=_should_install_runtime_binaries())


class DownloadBinariesAndBuild(build_py):
    """python -m build"""

    def run(self):
        super().run()
        install_bin(Path(self.build_lib) / "sensapex", force=_should_bundle_build_binaries())


def install_bin(path: Path, force: bool = False) -> None:
    """Install libum.dll and umpcli.exe to *path*."""
    if not force and platform.system() != "Windows":
        return

    path.mkdir(parents=True, exist_ok=True)

    dll_data = download_from_zip(UMSDK_DLL_URL, UMSDK_DLL_MEMBERS, env_var=UMSDK_ENV)[0]
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


def _should_install_runtime_binaries() -> bool:
    """Return True when binaries should be installed into site-packages."""
    env_value = os.environ.get(FORCE_BINARIES_ENV)
    if env_value is not None:
        return _env_value_truthy(env_value)

    return platform.system() == "Windows"


def _should_bundle_build_binaries() -> bool:
    """Return True when linux builds should package the Windows binaries."""
    env_value = os.environ.get(SKIP_BUILD_BINARIES_ENV)
    if env_value is not None:
        return not _env_value_truthy(env_value)

    return True


def _env_value_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
