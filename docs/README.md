# SimpleDSPy Documentation

This directory contains the Sphinx documentation for SimpleDSPy.

## Building the Documentation

To build the documentation locally:

```bash
# Install dependencies
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
cd docs
make html
```

The built documentation will be available in `_build/html/`.

## Documentation Structure

- `index.rst` - Main documentation page
- `getting_started.rst` - Getting started guide
- `api_simple.rst` - API reference (simplified version)
- `examples.rst` - Usage examples
- `conf.py` - Sphinx configuration

## Viewing the Documentation

After building, open `_build/html/index.html` in your web browser to view the documentation.

## Publishing

The documentation can be published to:
- GitHub Pages
- Read the Docs
- Any static hosting service

For Read the Docs, simply connect your repository and it will build automatically.