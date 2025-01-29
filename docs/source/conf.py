# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from inspect import getsourcefile
import os

project = 'dual_autodiff'
copyright = '2024, Phong-Anh Nguyen Trinh'
author = 'Phong-Anh Nguyen Trinh'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
    'nbsphinx',
    'sphinx_gallery.load_style',
    'recommonmark'
]

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# def ensure_pandoc_installed(_):
#     import pypandoc

#     # Download pandoc if necessary. If pandoc is already installed and on
#     # the PATH, the installed version will be used. Otherwise, we will
#     # download a copy of pandoc into docs/bin/ and add that to our PATH.
#     pandoc_dir = os.path.join("dual_autodiff\docs", "bin")
#     # Add dir containing pandoc binary to the PATH environment variable
#     if pandoc_dir not in os.environ["PATH"].split(os.pathsep):
#         os.environ["PATH"] += os.pathsep + pandoc_dir
#     pypandoc.ensure_pandoc_installed(
#         quiet=True,
#         targetfolder=pandoc_dir,
#         delete_installer=True,
#     )


# def setup(app):
#     app.connect("builder-inited", ensure_pandoc_installed)
