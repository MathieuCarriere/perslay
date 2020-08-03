from distutils.core import setup

setup(
    name                           = "perslay",
    author                         = "Mathieu Carriere, Théo Lacombe, Martin Royer",
    author_email                   = "mathieu.carriere3@gmail.com",
    description                    = "A tensorflow layer for handling persistence diagrams in neural networks",
    packages                       = ["perslay"],
    version                        = "2.0",
    long_description_content_type  = "text/markdown",
    long_description               = open("README.md", "r").read(),
    url                            = "https://github.com/MathieuCarriere/perslay/",
    classifiers                    = ("Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License", "Operating System :: OS Independent"),
)
