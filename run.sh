#!/bin/bash

# SQL Query Buddy - Startup Script

echo "🚀 Starting SQL Query Buddy..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "📋 Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your OpenAI API key if you want full functionality"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate || . venv/Scripts/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Create sample database if using SQLite
if grep -q "sqlite" .env; then
    echo "💾 Creating sample database..."
    python -c "from src.components.executor import SQLiteDatabase; SQLiteDatabase.create_sample_database('retail.db')"
fi

# Run the application
echo "🎉 Launching SQL Query Buddy..."
echo "📱 Open your browser and go to: http://localhost:7860"
echo ""

python -m src.main
