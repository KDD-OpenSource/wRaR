
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='csrar',
    version='0.2',
    description='Cost-Sensitive RaR',
    long_description=readme,
    author='Daniel Thevessen',
    author_email='me@danolithe.com',
    url='https://github.com/danthe96/CSHiCS-python',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
