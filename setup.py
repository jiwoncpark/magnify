from setuptools import setup, find_packages
#print(find_packages())
#required_packages = []
#with open('requirements.txt') as f:
#    required_packages = f.read().splitlines()
#required_packages += ['corner.py @ https://github.com/jiwoncpark/corner.py/archive/master.zip']
#print(required_packages)

setup(
      name='magnify',
      version='v0.1',
      author='Ji Won Park',
      author_email='jiwoncpark@gmail.com',
      packages=find_packages(),
      license='LICENSE.md',
      description='',
      long_description=open("README.rst").read(),
      long_description_content_type='text/x-rst',
      url='https://github.com/jiwoncpark/magnify',
      #install_requires=required_packages,
      #dependency_links=['http://github.com/jiwoncpark/corner.py/tarball/master#egg=corner_jiwoncpark'],
      include_package_data=True,
      entry_points={
      'console_scripts': [],
      },
      test_suite='nose.collector',
      tests_require=['nose'],
      classifiers=['Development Status :: 4 - Beta',
      'License :: OSI Approved :: BSD License',
      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python'],
      keywords='physics'
      )