# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import toml
import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dtaianomaly'
copyright = '2023-2025, DTAI'
author = 'Louis Carpentier \\and Nick Seeuws'

with open('../pyproject.toml', 'r') as f:
    config = toml.load(f)
release = config['project']['version']
version = config['project']['version']

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
]

numpydoc_show_class_members = False
numpydoc_validation_checks = {
    "all",
    "SA01",  # no see also section
    "EX01",  # no examples
}
numpydoc_validation_exclude = {
    r'dtaianomaly\.anomaly_detection\.Supervision',  # skip the enum
}

numpydoc_class_members_toctree = False

# Default settings for matplotlib.sphinxext.plot_directive
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False

# Logo
html_favicon = 'logo/favicon.svg'
html_logo = 'logo/html-logo.svg'
html_theme_options = {
    'logo_only': True
}

# References
bibtex_bibfiles = ['bibliography.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin theme
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
os.makedirs("_static", exist_ok=True)  # Ensure the _static directory exists
html_static_path = ['_static']

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'dtaianomalydoc'

# Prevent issue with pyximport when generating docs through readthedocs
autodoc_mock_imports = ["pyximport"]
