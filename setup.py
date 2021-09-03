import os

from io import BytesIO

from zipfile import ZipFile

import requests
from distutils.command.install import install
from os import path

from setuptools import setup, find_packages


class DownloadBinariesAndInstall(install):
    def run(self):
        super(DownloadBinariesAndInstall, self).run()
        req = requests.get("http://dist.sensapex.com/misc/um-sdk/latest/umsdk-1.022-binaries.zip")
        if req.status_code == 200:
            content_file = BytesIO(req.content)
            with ZipFile(content_file, "r") as zip_file:
                with zip_file.open("libum.dll") as dll_file:
                    with open(os.path.join(self.install_purelib, "sensapex", "libum.dll"), "wb") as install_target:
                        install_target.write(dll_file.read())
        req = requests.get("http://dist.sensapex.com/misc/umpcli/umpcli-0_951-beta.zip")
        if req.status_code == 200:
            content_file = BytesIO(req.content)
            with ZipFile(content_file, "r") as zip_file:
                with zip_file.open("umpcli-0_951-beta.exe") as umpcli_file:
                    with open(os.path.join(self.install_purelib, "sensapex", "umpcli.exe"), "wb") as install_target:
                        install_target.write(umpcli_file.read())


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
    cmdclass=dict(install=DownloadBinariesAndInstall),
    description="Python wrapper for the Sensapex SDK",
    install_requires=["numpy",],
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="sensapex",
    packages=find_packages(),
    python_requires=">=3.7",
    url="https://github.com/sensapex/sensapex-py",
    version="1.022.1",  # in lock step with umsdk version
)
