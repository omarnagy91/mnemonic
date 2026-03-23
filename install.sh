#!/bin/bash
set -euo pipefail

echo "🧠 Mnemonic — Self-hosted AI Memory for OpenClaw"
echo "================================================="
echo ""

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "❌ Docker required. Install: https://docs.docker.com/get-docker/"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3.10+ required"; exit 1; }
command -v pip >/dev/null 2>&1 || command -v pip3 >/dev/null 2>&1 || { echo "❌ pip required"; exit 1; }

# Check OpenAI key
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "⚠️  OPENAI_API_KEY not set. Please set it:"
    echo "   export OPENAI_API_KEY='sk-...'"
    exit 1
fi

echo "✅ Prerequisites OK"
echo ""

# Step 1: Start Qdrant
echo "📦 Starting Qdrant vector database..."
if docker ps --format '{{.Names}}' | grep -q '^qdrant$'; then
    echo "   Already running"
else
    docker run -d --name qdrant --restart unless-stopped \
        -p 6333:6333 -v ~/.data/qdrant:/qdrant/storage \
        qdrant/qdrant >/dev/null 2>&1
    echo "   Started on port 6333"
fi

# Step 2: Install Python deps
echo "📦 Installing Python dependencies..."
pip install -q mem0ai fastapi uvicorn 2>/dev/null || pip3 install -q mem0ai fastapi uvicorn

# Step 3: Copy server
INSTALL_DIR="${MNEMONIC_DIR:-$HOME/.mnemonic}"
mkdir -p "$INSTALL_DIR"
cp server/server.py "$INSTALL_DIR/server.py"
echo "   Server installed to $INSTALL_DIR"

# Step 4: Copy plugin
PLUGIN_DIR="${OPENCLAW_HOME:-$HOME/.openclaw}/extensions/openclaw-mem0"
mkdir -p "$PLUGIN_DIR"
cp plugin/index.ts "$PLUGIN_DIR/"
cp plugin/package.json "$PLUGIN_DIR/"
cp plugin/openclaw.plugin.json "$PLUGIN_DIR/"
echo "   Plugin installed to $PLUGIN_DIR"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Start the server:  cd $INSTALL_DIR && python3 server.py"
echo "  2. Add plugin config to ~/.openclaw/openclaw.json (see README.md)"
echo "  3. Restart OpenClaw:  openclaw gateway restart"
echo ""
echo "🧠 Your AI will now remember everything."
