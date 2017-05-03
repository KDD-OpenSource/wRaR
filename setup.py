from setuptools import setup


setup(
    name='csrar',
    version='0.19',
    author='Daniel Thevessen',
    install_requires=[
        'pandas',
        'numpy'
    ],
    packages=['rar', 'hics']
)
