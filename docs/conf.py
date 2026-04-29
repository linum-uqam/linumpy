# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""Sphinx configuration for the linumpy documentation."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Make the project package importable for autoapi/autodoc.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))

# -- Project information ----------------------------------------------------
project = "linumpy"
author = "The LINUM developers"
copyright = f"{datetime.now().year}, LINUM"

# Pull version from installed package metadata when available.
try:
    from importlib.metadata import version as _get_version

    release = _get_version("linumpy")
except Exception:
    release = "0.1.1"
version = ".".join(release.split(".")[:2])

# -- General configuration --------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinxarg.ext",
    "myst_parser",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "notfound.extension",
    "sphinx_sitemap",
    "sphinxext.opengraph",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST: render the existing Markdown docs.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "linkify",
    "substitution",
    "tasklist",
]
myst_fence_as_directive = ["mermaid"]

# Mermaid: interactive zoom/pan + fullscreen, with readable defaults.
mermaid_d3_zoom = True
mermaid_fullscreen = True
mermaid_fullscreen_button = "⛶"
mermaid_height = "640px"
mermaid_light_theme = "neutral"
mermaid_dark_theme = "dark"
mermaid_init_config = {
    "startOnLoad": True,
    "securityLevel": "loose",
    "flowchart": {"htmlLabels": True, "curve": "basis", "useMaxWidth": True},
    "themeVariables": {"fontSize": "16px"},
}
myst_heading_anchors = 4

# Autoapi: generate API reference from the linumpy package.
autoapi_type = "python"
autoapi_dirs = [str(ROOT / "linumpy")]
autoapi_root = "api"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_ignore = ["*/tests/*", "*/config/threads*"]
autoapi_keep_files = True
autoapi_add_toctree_entry = True

# Autodoc settings.
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Napoleon: support Google + NumPy docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True
# Render "Attributes:" docstring sections as :ivar: fields instead of
# emitting a separate :py:attribute: directive for each — avoids the
# "duplicate object description" warnings when autoapi also documents
# the same class attributes from their type annotations.
napoleon_use_ivar = True

# Intersphinx mappings.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
    "SimpleITK": ("https://simpleitk.readthedocs.io/en/master/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
}

# Nitpicky mode is kept off. autoapi-extracted API pages contain many
# informal type names from numpy-style docstrings (``ndarray``, ``optional``,
# ``array-like``, ...) that are not real Python xrefs and cannot be resolved
# without rewriting every docstring. Strict builds use ``-W`` (treat warnings
# as errors) but not ``-n``; run ``sphinx-build -n`` ad-hoc when reviewing API
# docstring quality.
nitpicky = False

# -- HTML output ------------------------------------------------------------
# pydata-sphinx-theme: https://pydata-sphinx-theme.readthedocs.io/
html_theme = "pydata_sphinx_theme"
html_title = "linumpy"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/linum-uqam/linumpy",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_prev_next": True,
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/linum-uqam/linumpy",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

# Hide the right "On this page" sidebar by default for narrative docs;
# autoapi pages still show it. This lets prose pages use the full width.
html_theme_options["secondary_sidebar_items"] = {
    "**": ["page-toc"],
    "index": [],
    "getting_started": [],
    "pipelines": [],
    "formats": [],
    "reference": [],
}

html_context = {
    "github_user": "linum-uqam",
    "github_repo": "linumpy",
    "github_version": "dev",
    "doc_path": "docs",
}

# Suppress noisy warnings from autoapi when imports fail in optional modules.
suppress_warnings = [
    "autoapi.python_import_resolution",
    # autoapi sometimes emits the same attribute twice when it's both a
    # dataclass field and a property — harmless, just noisy.
    "ref.python",
    # Tolerate ambiguous Python xrefs like ``shape`` resolving to several
    # classes; autoapi can't disambiguate without manual annotations.
    "misc.highlighting_failure",
]

# -- UX extensions ----------------------------------------------------------
# sphinx-copybutton: copy button on code blocks; strip prompt characters.
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

# sphinx-notfound-page: serve a friendly 404 with absolute links to assets.
notfound_context = {
    "title": "Page not found",
    "body": (
        "<h1>Page not found</h1>"
        "<p>Sorry, we couldn't find that page. Try the "
        "<a href='/'>documentation home</a> or use the search box above.</p>"
    ),
}
notfound_urls_prefix = "/"

# sphinx-sitemap: emit sitemap.xml at the docs root for SEO.
html_baseurl = "https://linumpy.readthedocs.io/en/latest/"
sitemap_url_scheme = "{link}"

# sphinxext-opengraph: rich link previews on social platforms.
ogp_site_url = html_baseurl
ogp_site_name = "linumpy documentation"
ogp_image = "https://linumpy.readthedocs.io/en/latest/_static/linumpy-logo.png"
ogp_use_first_image = True
ogp_enable_meta_description = True
