#!/bin/bash

# Development Setup Script for News Aggregator
# This script sets up a complete development environment

set -e

echo "🚀 Setting up News Aggregator Development Environment..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

print_success "Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    print_error "pip is not installed"
    exit 1
fi

print_success "pip found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
print_info "Installing development dependencies..."
pip install pytest pytest-cov pytest-mock black flake8

# Install production dependencies (uncommenting them temporarily)
print_info "Installing production dependencies..."
pip install \
    requests>=2.28.0 \
    feedparser>=6.0.0 \
    trafilatura>=1.6.0 \
    beautifulsoup4>=4.12.0 \
    lxml>=4.9.0 \
    loguru>=0.7.0 \
    python-dateutil>=2.8.0 \
    sqlalchemy>=2.0.0 \
    psycopg2-binary>=2.9.0 \
    redis>=4.5.0 \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    pydantic>=2.0.0 \
    celery>=5.3.0 \
    prometheus-client>=0.19.0 \
    python-dotenv>=1.0.0

print_success "All dependencies installed"

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p logs outputs cache exports
mkdir -p monitoring/alerts

# Set up environment file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file from template..."
    cp .env.example .env
    print_success ".env file created"
else
    print_info ".env file already exists"
fi

# Create test database (SQLite for development)
print_info "Setting up test database..."
python3 -c "
from db.database import db_manager
try:
    db_manager.create_tables()
    print('Database tables created successfully')
except Exception as e:
    print(f'Warning: Could not create tables: {e}')
    print('This is expected if PostgreSQL is not running')
"

# Verify installation
print_info "Verifying installation..."
python3 -c "
import sys
try:
    from fastapi import FastAPI
    from sqlalchemy import create_engine
    from redis import Redis
    import celery
    import prometheus_client
    print('All core dependencies imported successfully')
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
"

print_success "All verifications passed"

# Run tests if requested
if [ "$1" = "--test" ]; then
    print_info "Running tests..."
    pytest tests/ -v --tb=short
fi

# Run linting if requested
if [ "$1" = "--lint" ]; then
    print_info "Running linting..."
    black --check .
    flake8 .
fi

print_success "Development environment setup complete!"
print_info ""
print_info "Next steps:"
print_info "1. Start Docker services: docker-compose up -d"
print_info "2. Initialize database: python db/init_db.py"
print_info "3. Pull LLM models: docker-compose exec ollama ollama pull qwen3:4b"
print_info "4. Run tests: pytest tests/ -v"
print_info "5. Start development: python api/main.py"
print_info ""
print_success "Happy coding! 🚀"
