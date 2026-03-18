"""Custom build backend to download Sensapex SDK binaries during installation.
This extends setuptools.build_meta to add download hooks."""

from pathlib import Path
from setuptools import build_meta as _orig

# Re-export all the standard hooks
__all__ = [
    "build_wheel",
    "build_sdist",
    "build_editable",
    "get_requires_for_build_wheel",
    "get_requires_for_build_sdist",
    "get_requires_for_build_editable",
    "prepare_metadata_for_build_wheel",
    "prepare_metadata_for_build_editable",
]


def _download_binaries(wheel_directory, config_settings):
    """Download SDK binaries after build."""
    from sensapex._build import install_bin

    # Install binaries to the sensapex package directory
    source_dir = Path(__file__).parent / "sensapex"
    print(f"Downloading Sensapex SDK binaries to {source_dir}")
    install_bin(source_dir)


# Expose standard build backend functions
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a wheel and include downloaded binaries."""
    _download_binaries(wheel_directory, config_settings)
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    """Build a source distribution."""
    return _orig.build_sdist(sdist_directory, config_settings)


# Handle editable installs
if hasattr(_orig, "get_requires_for_build_editable"):
    get_requires_for_build_editable = _orig.get_requires_for_build_editable
    prepare_metadata_for_build_editable = _orig.prepare_metadata_for_build_editable

    def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
        """Build an editable install and include downloaded binaries."""
        _download_binaries(wheel_directory, config_settings)
        return _orig.build_editable(wheel_directory, config_settings, metadata_directory)
