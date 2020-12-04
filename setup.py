'''
Created on 04 December 2020
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
#from distutils.core import setup


setup(name='emulator',
      version='0.0.1',
      author='Sambit Giri',
      author_email='sambit.giri@gmail.com',
      package_dir = {'emulator' : 'src'},
      packages=['emulator'],
      package_data={'share':['*'],},
      install_requires=['numpy','scipy','scikit-learn','scikit-image', 'pyDOE'],
      #include_package_data=True,
)