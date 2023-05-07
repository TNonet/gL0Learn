from skbuild import setup  # This line replaces 'from setuptools import setup'

setup(
    name="test1",
    version="1.2.3",
    description="a minimal example package (cpp version)",
    author='The scikit-build team',
    license="MIT",
    packages=["hello"],
    package_dir={"": "src"},
    cmake_install_dir="src/hello",
    python_requires=">=3.7",
)