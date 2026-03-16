"""Sphinx configuration for the Exordium documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Exordium"
copyright = "2024-2026, Ádám Fodor"
author = "Ádám Fodor"
release = "latest"

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# ── sphinx-autoapi ─────────────────────────────────────────────────────────────

autoapi_dirs = ["../exordium"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
autoapi_add_toctree_entry = True
autoapi_member_order = "source"
autoapi_python_class_content = "both"


def skip_undocumented_attributes(app, what, name, obj, skip, options):
    """Hide undocumented attributes from API docs.

    Keeps all classes, functions, methods, and modules even without docstrings,
    but hides bare attributes that have no docstring.

    Args:
        app: The Sphinx application object.
        what: The type of the object (e.g. "attribute", "class").
        name: The fully-qualified name of the object.
        obj: The object itself.
        skip: Whether the object would be skipped by default.
        options: The options given to the directive.

    Returns:
        True if the object should be skipped, False otherwise.

    """
    if what == "attribute" and not obj.docstring:
        return True
    return skip


def setup(app):
    """Connect custom skip handler to autoapi."""
    app.connect("autoapi-skip-member", skip_undocumented_attributes)


# ── Napoleon (Google docstrings) ───────────────────────────────────────────────

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# ── MyST ───────────────────────────────────────────────────────────────────────

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# ── Furo theme ─────────────────────────────────────────────────────────────────

html_theme = "furo"
html_static_path = ["assets"]
html_baseurl = "https://fodorad.github.io/exordium/"
