import os

from io import BytesIO
import platform
from zipfile import ZipFile
import urllib.request
from distutils.command.install import install
from os import path

from setuptools import setup, find_packages
from setuptools.command.develop import develop


class DownloadBinariesAndInstall(install):
    def run(self):
        super(DownloadBinariesAndInstall, self).run()
        install_bin(os.path.join(self.install_purelib, 'sensapex'))


class DownloadBinariesAndDevelop(develop):
    def run(self):
        super(DownloadBinariesAndDevelop, self).run()
        install_bin(os.path.join(self.egg_path, 'sensapex'))


def install_bin(path):
    """Install libum.dll and umpcli.exe to *path*."""
    if platform.system() == 'Windows':
        dll_data = download_from_zip('http://dist.sensapex.com/misc/um-sdk/latest/umsdk-1.022-binaries.zip', ["libum.dll"])[0]
        print(f"Install libum.dll to {path}")
        with open(os.path.join(path, "libum.dll"), "wb") as install_target:
            install_target.write(dll_data)
        umpcli_data = download_from_zip('http://dist.sensapex.com/misc/umpcli/umpcli-0_951-beta.zip', ["umpcli-0_951-beta.exe"])[0]
        with open(os.path.join(path, "umpcli.exe"), "wb") as install_target:
            install_target.write(umpcli_data)


def download_from_zip(url, files):
    req = urllib.request.urlopen(url)
    content_file = BytesIO(req.read())
    data = []
    with ZipFile(content_file, "r") as zip_file:
        for fname in files:
            with zip_file.open(fname) as req_file:
                data.append(req_file.read())
    return data


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    author="Luke Campagnola",
    author_email="luke.campagnola@gmail.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    cmdclass={"install": DownloadBinariesAndInstall, "develop": DownloadBinariesAndDevelop},
    description="Python wrapper for the Sensapex SDK",
    install_requires=["numpy",],
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="sensapex",
    packages=find_packages(),
    python_requires=">=3.7",
    url="https://github.com/sensapex/sensapex-py",
    version="1.022.6",  # in lock step with umsdk version. don't forget __version__ and changelog!
)
