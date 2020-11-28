# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import subprocess

#sys.path.insert(0, os.path.abspath('.'))
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../../G4HepEm/'))


# call doxygen to generate the XML files
subprocess.call('cd ..; doxygen Doxyfile.in', shell=True)


# -- Project information -----------------------------------------------------
project = 'The G4HepEm R&D project'
copyright = '2020, M.Novak'
author = 'Mihaly Novak'

# The full version, including alpha/beta/rc tags
release = '0.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',    # autogenerate documentation i.e. the .rst files
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',    
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',    # to have the [source] option in the doc
    'nbsphinx',               # for jupiter notebook conversions  
    'sphinx.ext.napoleon',    # for the google-style doc
    'breathe',                # mixed doxygen and sphinx doc with breathe
    'sphinxcontrib.bibtex'    # to use bibtex
]

# set path to the doxygen-generated XML for breathe
breathe_projects = { 'The G4HepEm R&D project': '../doxygen/xml' }


numfig = True
source_suffix = '.rst'
master_doc = 'index'

# option for autodoc
autodoc_member_order = 'bysource'  # default is alphabetical

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
#html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
#def setup(app):
#   app.add_css_file("style.css")
   
html_logo = "logo_HepEM3.png"
html_show_sourcelink = False
#html_show_copyright = False
#html_show_sphinx = False
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}




# -- Options for LaTeX output ---------------------------------------------
latex_elements = {
  "preamble": r"""
    \usepackage{amsmath}
    \usepackage{amsfonts}
    \usepackage{amssymb}
    \usepackage{bm}
    \usepackage{bbm}
    \usepackage{booktabs}
%    \usepackage[table,xcdraw]{xcolor}
    \usepackage{rotating,tabularx}
    \usepackage{multirow}
  """,
#  'fncychap': '\\usepackage[Conny]{fncychap}',
#    \usepackage{mathtools}

  'maketitle': r'''
     \pagenumbering{Roman} %%% to avoid page 1 conflict with actual page 1 
     \begin{titlepage}
       %% * give space from top 
       \vspace*{30mm} 
       \textbf{\Huge {The \texttt{G4HepEm} R\&D Project Documentation}}
       \rule{1.0\linewidth}{2.4pt}\\[-3.7ex] \rule{1.0\linewidth}{0.6pt}
       %% add logo
          \vspace*{20mm}
          \begin{figure}[!h]
             \centering
             \includegraphics[scale=1.1]{logo_HepEM3.png}
          \end{figure}
       %% add some space
       %% add space till the bottom
       \vfill
       \vspace*{-50mm} 
       \centering
       \Large \textbf{Mih{\'a}ly Nov{\'a}k}\\ CERN EP-SFT\\
       \vspace*{30mm} 
       \small \textbf{\today}
    \end{titlepage}
    \pagenumbering{arabic}
''',
}
latex_logo = "logo_HepEM3.png"
