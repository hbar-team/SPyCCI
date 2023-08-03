import setuptools

setuptools.setup(
    name="SPyCCI",
    version="0.0.1",
    description="Simple Python Computational Chemistry Interface",
    long_description="Simple Python Computational Chemistry Interface",
    packages=["spycci"],
    package_data={
        "spycci": ["*", "wrappers/*", "engines/*", "functions/*", "tools/*", "core/*"],
    },
    install_requires=[],
)
