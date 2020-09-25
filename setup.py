import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="shsh",
    version="0.1.dev0",
    author="Jo Bovy",
    author_email="bovy@astro.utoronto.ca",
    description="Tools for dealing with the shearing sheet",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    packages=["shsh"],
    install_requires=['numpy','jax','jaxlib','galpy']
    )
