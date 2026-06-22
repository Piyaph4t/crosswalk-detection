#!/bin/bash

# Crosswalk Detection PoC - Raspbian Environment Setup Script
# Targeted for Raspberry Pi 4 Model B (4GB)

set -e

echo "🚀 Starting environment setup for Crosswalk Detection PoC..."

# 1. Update System
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# 2. Install System-Level Dependencies
# These are critical for OpenCV and NumPy performance on ARM
echo "Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-dev \
    libopencv-dev \
    git \
    libatlas-base-dev \
    libgfortran5

# 3. Install UV (Ultra-fast Python package manager)
if ! command -v uv &> /dev/null
then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
else
    echo "uv is already installed."
fi

# 4. Initialize Project Environment
echo "Initializing Python project with uv..."
# Use the system python binary to ensure compatibility with Raspbian repo
uv venv --python $(which python3)
source .venv/bin/activate

# Install dependencies from pyproject.toml
echo "Installing project dependencies..."
uv pip install -e .

# 5. Verify installation
echo "Verifying installation..."
python -c "import cv2; import numpy; import yaml; print('✅ Core dependencies installed successfully')"

# 6. Enable I2C for HuskyLens (optional, only if using HuskyLens)
echo "Enabling I2C interface..."
# This typically requires a reboot to take effect
sudo raspi-config nonint do_i2c 0

echo "------------------------------------------------------------"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source .venv/bin/activate"
echo "  2. Configure your camera in config.yaml"
echo "  3. Run inference: uv run python src/main.py"
echo "  4. Run tests: uv run pytest tests/"
echo ""
echo "⚠️  IMPORTANT: Please REBOOT your Raspberry Pi to activate I2C."
echo "Command: sudo reboot"
echo "------------------------------------------------------------"
