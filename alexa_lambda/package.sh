#!/bin/bash
# Package the Lambda function for deployment.
# Creates lambda_deployment.zip ready to upload to AWS Lambda.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Packaging Diet Assistant Lambda ==="

# Clean previous build
rm -rf package lambda_deployment.zip
mkdir -p package

# Install Python dependencies (targeting Lambda's Linux x86_64 runtime)
echo "Installing dependencies..."
pip install --target ./package -q \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --python-version 3.11 \
    --only-binary=:all: \
    openai \
    requests \
    lxml \
    python-dotenv

# Copy source files
echo "Copying source files..."
cp lambda_function.py package/
cp assistant_core.py package/
cp myfitnesspal.py package/

# Extract MFP cookie from .env and bundle as cookie.txt
# (Cookie is too large for Lambda env vars, so we store it in a file)
echo "Extracting MFP cookie..."
ENV_FILE="../.env"
if [ -f "$ENV_FILE" ]; then
    COOKIE=$(grep "^MFP_COOKIE=" "$ENV_FILE" | sed 's/^MFP_COOKIE=//')
    if [ -n "$COOKIE" ]; then
        echo "$COOKIE" > package/cookie.txt
        echo "  Cookie bundled ($(wc -c < package/cookie.txt | tr -d ' ') bytes)"
    else
        echo "  WARNING: MFP_COOKIE not found in .env"
    fi
else
    echo "  WARNING: .env file not found at $ENV_FILE"
fi

# Also copy meals.json if it exists (for presets)
if [ -f "../meals.json" ]; then
    cp ../meals.json package/
fi

# Create ZIP
echo "Creating deployment ZIP..."
cd package
zip -r ../lambda_deployment.zip . -q
cd ..

# Show size
SIZE=$(du -h lambda_deployment.zip | cut -f1)
echo ""
echo "=== Done! ==="
echo "  File: lambda_deployment.zip ($SIZE)"
echo ""
echo "Next steps:"
echo "  1. Go to AWS Lambda console"
echo "  2. Upload lambda_deployment.zip"
echo "  3. Set handler to: lambda_function.lambda_handler"
echo "  4. Set environment variables:"
echo "     - OPENAI_API_KEY (from your .env)"
echo "     - USER_NAME (default: Aaran)"
echo "  5. Set timeout to 15 seconds, memory to 512 MB"
echo "  6. Add Alexa Skills Kit trigger"
echo ""
echo "  NOTE: MFP cookie is bundled in the ZIP as cookie.txt."
echo "  When the cookie expires, re-run this script and re-upload."
