
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='wrar',
    version='0.2',
    description='Weighted Relevance and Redundancy Scoring',
    long_description=readme,
    author='Daniel Thevessen',
    author_email='me@danolithe.com',
    url='https://github.com/KDD-OpenSource/wRaR',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'examples'))
)
