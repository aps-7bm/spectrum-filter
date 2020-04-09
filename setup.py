from setuptools import setup, find_packages
import os

setup(
    name='spectrum-filter',
    version=open('VERSION').read().strip(),
    #version=__version__,
    author='Alan Kastengren',
    author_email='akastengren@anl.gov',
    url='https://github.com/aps-7bm/spectrum-filter',
    packages=find_packages(),
    include_package_data = True,
    scripts=['bin/spectrum'],
    description='Find effective spectrum of filtered polychromatic x-rays',
    zip_safe=False,
)

