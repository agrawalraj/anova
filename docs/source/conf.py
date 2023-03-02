# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../tutorials'))


# -- Project information

project = 'fava'
copyright = '2023, Raj Agrawal'
author = 'Raj Agrawal'

release = '1.0'
version = '1.0.0'

import pydata_sphinx_theme

# -- General configuration

# The master toctree document.
master_doc = 'index'

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    "nbsphinx",
    "nbsphinx_link"
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output
html_static_path = ['_static']
html_logo = "fava_logo.png"


html_theme = 'pydata_sphinx_theme'


# -- Options for EPUB output
epub_show_urls = 'footnote'
