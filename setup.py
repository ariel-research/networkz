from setuptools import setup, find_packages

setup(
    name='networky',
    version='1.0.0',
    author='Ariel University',
    author_email='ariel_research23@gmail.com',
    description='Extended Graph Library on Top of NetworkX',
    packages=['networky'],
    install_requires=open('requirements.txt').readlines(),
    scripts=['scripts/update_nx.py'],

)
