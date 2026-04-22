#!/bin/bash
# Build MCAP paper
# Requires: texlive-full or tectonic
#
# Option 1: pdflatex
#   cd nve/paper && bash build.sh
#
# Option 2: tectonic (auto-downloads packages)
#   cd nve/paper && tectonic mcap.tex
#
# Option 3: Overleaf
#   Upload mcap.tex, neurips_2025.sty, and the figures_paper/ directory

set -e
cd "$(dirname "$0")"

echo "=== Building MCAP paper ==="

if command -v tectonic &>/dev/null; then
    echo "Using tectonic..."
    tectonic mcap.tex
elif command -v pdflatex &>/dev/null; then
    echo "Using pdflatex..."
    pdflatex -interaction=nonstopmode mcap.tex
    pdflatex -interaction=nonstopmode mcap.tex  # second pass for refs
    echo "=== Done: mcap.pdf ==="
else
    echo "ERROR: No LaTeX compiler found."
    echo "Install one of:"
    echo "  apt install texlive-full"
    echo "  cargo install tectonic"
    echo "  pip install tectonic"
    echo ""
    echo "Or upload to Overleaf (https://www.overleaf.com):"
    echo "  - mcap.tex"
    echo "  - neurips_2025.sty"
    echo "  - ../evidence/figures_paper/*.png"
    exit 1
fi

echo "=== Output: mcap.pdf ==="
ls -la mcap.pdf 2>/dev/null
