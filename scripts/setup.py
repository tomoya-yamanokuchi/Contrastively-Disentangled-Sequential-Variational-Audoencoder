from setuptools import setup, find_packages

setup(
    name        = 'custom',
    version     = '0.1.0',
    description = 'pytorch custom network',
    packages    = find_packages(where='custom'),
    package_dir = {'': 'custom'},
)


'''
Usage (in terminal):
    python setup.py install --user

'''