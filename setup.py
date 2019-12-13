from setuptools import setup, find_packages

# find_packages() only seems to look one directory deep
packages = find_packages(exclude=[ "setup"])

# DO NOT EDIT!  This string is modified by bumpversion!
__version__ = "1.0.0"

setup(
    name='cdeepbind',
    version=__version__,
    description="cdeepbind",
    author="Shreshth Gandhi",
    author_email='shreshth@deepgenomics.com',
    packages=packages,
    include_package_data=True,
    zip_safe=False,
)
