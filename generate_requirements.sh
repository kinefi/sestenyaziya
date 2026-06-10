#!/bin/bash

uv pip freeze > requirements.txt

sed -i "1i --extra-index-url https://download.pytorch.org/whl/cpu" requirements.txt

GRADIO_VERSION=$(uv pip show gradio | grep "Version" | awk '{print $2}')
PYTHON_VERSION=$(cat .python-version)

echo "Gradio version: $GRADIO_VERSION"
echo "Python version: $PYTHON_VERSION"

sed -i "s/sdk_version:.*/sdk_version: \"$GRADIO_VERSION\"/" README.md

sed -i "s/python_version:.*/python_version: \"$PYTHON_VERSION\"/" README.md
