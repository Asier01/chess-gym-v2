from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='chess_gym_v2',
      version='0.0.5.1',
      description='OpenAI Gymnasium environment for Chess, using the game engine of the python-chess module',
      url='https://github.com/Asier01/chess-gym-v2',
      packages= find_packages(),
      author='Ryan Rudes',
      author_email='ryanrudes@gmail.com',
      license='MIT License',
      install_requires=['gymnasium', 'python-chess', 'numpy', 'cairosvg', 'pillow'],
      long_description=long_description,
      long_description_content_type="text/markdown",
)
