#!/bin/bash
# generate_requirements.sh - Keeps requirements.txt in sync with uv.lock

set -e # Exit immediately if a command exits with a non-zero status.

echo "🔄 Syncing dependencies and updating lockfile..."
uv lock

echo "📦 Exporting dependencies to requirements.txt..."
# --no-dev: Excludes development dependencies
# --no-hashes: Removes package hashes for a cleaner requirements file
uv export --format requirements-txt --output-file requirements.txt --no-dev --no-hashes

echo "🔌 Injecting extra index for PyTorch CPU..."
# Extract the extra index URL from pyproject.toml
EXTRA_INDEX_URL=$(grep -A 2 '\[\[tool.uv.index\]\]' pyproject.toml | grep "url =" | cut -d '"' -f 2)
if [ -n "$EXTRA_INDEX_URL" ] && ! grep -q "$EXTRA_INDEX_URL" requirements.txt; then
    sed -i "1i --extra-index-url $EXTRA_INDEX_URL" requirements.txt
fi

echo "🔍 Extracting metadata..."
# Extract versions dynamically from the project configuration
GRADIO_VERSION=$(grep -A 1 'name = "gradio"' uv.lock | grep "version =" | head -n 1 | cut -d '"' -f 2)
PYTHON_VERSION=$(grep "requires-python =" pyproject.toml | head -n 1 | cut -d '"' -f 2 | sed 's/[>=]//g')

echo "📍 Gradio version: $GRADIO_VERSION"
echo "📍 Python version: $PYTHON_VERSION"

sed -i "s/^sdk_version: .*/sdk_version: \"$GRADIO_VERSION\"/" README.md
sed -i "s/^python_version: .*/python_version: \"$PYTHON_VERSION\"/" README.md

echo "✅ requirements.txt has been successfully updated."
echo "💡 Note: This file is used for HuggingFace Space deployment."
