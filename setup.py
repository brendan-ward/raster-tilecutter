import os
from setuptools import setup

setup(
    name="raster-tilecutter",
    version="0.2.0",
    packages=["tilecutter"],
    url="https://github.com/brendan-ward/raster-tilecutter",
    license="MIT",
    author="Brendan C. Ward",
    author_email="bcward@astutespruce.com",
    description="Cut and render a raster data source into mbtiles",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    install_requires=[
        "rasterio>=1.1.5",
        "Pillow",
        "numpy",
        "mercantile",
        "pymbtiles",
        "click",
        "progress",
    ],
    include_package_data=True,
    extras_require={"test": ["pytest", "pytest-cov"]},
)
