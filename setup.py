from setuptools import setup, find_packages

setup(name='emperorviopy',
      version='0.1',
      description='Visual Inertial Odometry',
      url='https://github.com/Emperor-Robotics/VisualOdometry',
      author='Kadyn Martinez; Wataru Oshima',
      author_email='',
      license='',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'opencv-python',
          'pyserial',
          'pyyaml'
      ],
      zip_safe=False)
