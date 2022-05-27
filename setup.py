from setuptools import setup, find_packages

version = '1.8.2'

setup(name='robotframework-eyes',
      version=version,
      description="Visual regression library and report generator for robot framework",
      long_description="""\
Visual regression library and report generator for robot framework. Capture elements, fullscreens and compare them against baseline images. Extends Selenium2Libary. Visit https://github.com/jz-jess/RobotEyes for documentation.""",
      classifiers=[
        'Framework :: Robot Framework',
        'Programming Language :: Python',
        'Topic :: Software Development :: Testing',
      ],
      keywords='visual-regression image-comparison robotframework robot-eyes compare-image',
      author='Orion c',
      author_email='t880216t@gmail.com',
      url='https://github.com/t880216t/RobotEyes',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'pillow',
          'robotframework',
          'flask',
          'opencv-python',
          'requests',
          'scikit-image',
      ],
      entry_points={
          'console_scripts': [
              'eyes = RobotEyes.runner:main',
              'reportgen = RobotEyes.reportgen:report_gen'
          ]
      },
)
