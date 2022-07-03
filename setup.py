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
      'gym[box2d,classic_control]',
      'ipython',
      'ipywidgets',
      'joblib',
      'matplotlib',
      'numpy',
      'pandas',
      'pytest',
      'psutil',
      'scipy',
      'seaborn',
      'torch',
      'tqdm'
   ],
)
