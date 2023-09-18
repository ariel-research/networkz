from setuptools import setup, find_packages

with open('requirements/default.txt') as r:
    install_requires = r.readlines()

packages = find_packages()

setup(
    name='networky',
    version='1.0.2',
    author='Ariel University',
    author_email='networky@csariel.xyz',
    description='Extended Graph-Algorithms Library on Top of NetworkX',
    packages=packages,
    install_requires= install_requires,
)
