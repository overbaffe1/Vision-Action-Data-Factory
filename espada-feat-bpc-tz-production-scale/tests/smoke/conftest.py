"""Smoke-package conftest.

Heavy-dependency stubs (torch/transformers/ultralytics) are installed by the
root ``tests/conftest.py`` so ``runner`` and the model wrappers import cleanly in
a minimal environment. Nothing extra is needed here; this file is kept as a
marker for the smoke package.
"""
