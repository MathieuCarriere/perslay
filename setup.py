import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='perslay',
     version='0.0.1',
     author="Mathieu Carriere",
     author_email="mc4660@columbia.edu",
     description="Implementation of the PersLay layer for Persistence Diagrams",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/MathieuCarriere/perslay",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
