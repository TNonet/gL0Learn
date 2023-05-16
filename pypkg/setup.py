#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# This was borrowed heavily form https://github.com/RUrlus/diptest/
import sys

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

import io
import re
from os.path import dirname
from os.path import join


PACKAGE_NAME = "gl0learn"

MAJOR = 0
MINOR = 1
MICRO = 0
DEVELOPMENT = False

VERSION = f"{MAJOR}.{MINOR}.{MICRO}"
FULL_VERSION = VERSION
if DEVELOPMENT:
    FULL_VERSION += ".dev"


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


def write_version_py(filename: str = f"src/{PACKAGE_NAME}/version.py") -> None:
    """Write package version to version.py.
    This will ensure that the version in version.py is in sync with us.
    Parameters
    ----------
    filename : str
        the path the file to write the version.py
    """
    # Do not modify the indentation of version_str!
    version_str = """\"\"\"THIS FILE IS AUTO-GENERATED BY gl0learn setup.py.\"\"\"
name = '{name!s}'
version = '{version!s}'
full_version = '{full_version!s}'
release = {is_release!s}
"""

    with open(filename, "w") as version_file:
        version_file.write(
            version_str.format(
                name=PACKAGE_NAME.lower(),
                version=VERSION,
                full_version=FULL_VERSION,
                is_release=not DEVELOPMENT,
            )
        )


if __name__ == "__main__":
    write_version_py()

    setup(
        name="gl0learn",
        version=FULL_VERSION,
        packages=["gl0learn"],
        package_dir={"": "src"},
        cmake_install_dir="src/gl0learn",
        cmake_args=[
            f"-DGL0LEARN_VERSION_INFO:STRING={VERSION}",
        ],
        long_description_content_type="text/x-rst",
        long_description="%s\n%s"
        % (
            re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
                "", read("README.rst")
            ),
            re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
        ),
    )
