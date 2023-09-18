import pathlib
import setuptools

NAME = "networky"
URL = "https://github.com/ariel-research/" + NAME
HERE = pathlib.Path(__file__).parent
print(f"\nHERE = {HERE.absolute()}\n")

README = (HERE / "README.md").read_text()
REQUIRES = (HERE / "requirements/default.txt").read_text().strip().split("\n")
REQUIRES = [lin.strip() for lin in REQUIRES]

print(f'\nVERSION = {(HERE / NAME / "VERSION").absolute()}\n')
VERSION = (HERE / NAME / "VERSION").read_text().strip()
# See https://packaging.python.org/en/latest/guides/single-sourcing-package-version/

setuptools.setup(
    name='networky',
    packages=setuptools.find_packages(),
    version=VERSION,
    install_requires=REQUIRES,

    author='Ariel University',
    author_email='networky@csariel.xyz',
    description='Extended Graph-Algorithms Library on Top of NetworkX',

    url=URL,
    project_urls={"Bug Reports": URL + "/issues", "Source Code": URL},

    long_description=README,
    long_description_content_type="text/markdown",
)


# Build:
#   Delete old folders: build, dist, *.egg_info, .venv_test.
#   Then run:
#        python -m build
#   Or (old version):
#        python setup.py sdist bdist_wheel

# Publish to test PyPI:
#   twine upload --repository testpypi dist/*

# Publish to real PyPI:
#   twine upload --repository pypi dist/*
