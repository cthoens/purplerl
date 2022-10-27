from setuptools import setup

setup(
   name='purplerl',
   version='0.1',
   description='Module for playing with reinforcement learning algorithms',
   author='Christoph Thoens',
   author_email='christoph.thoens@gmail.com',
   packages=['purplerl'],
   install_requires=[
      'cloudpickle',
      'gym',
      'ipython',
      'ipywidgets',
      'joblib',
      'matplotlib',
      'numpy',
      'pandas',
      'pytest',
      'psutil',
      'torch',
      'tqdm'
   ],
)
