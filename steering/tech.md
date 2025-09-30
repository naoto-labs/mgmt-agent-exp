# Technology Stack

## Core Framework
- **FastAPI**: Main web framework for API endpoints
- **Python 3.9+**: Primary programming language
- **Pydantic**: Data validation and settings management
- **Uvicorn**: ASGI server for FastAPI

## AI & ML Integration
- **Anthropic Claude**: Primary AI model for agent decision-making
- **OpenAI GPT**: Secondary/fallback AI model
- **Multiple model support**: Unified interface with fallback capabilities

## Data Storage
- **SQLite**: Transactional data (accounting, inventory, transactions)
- **MongoDB/JSON**: Conversation logs and unstructured data
- **Hybrid architecture**: SQL for structured business data, NoSQL for AI interactions

## Key Dependencies
```
fastapi = "^0.104.0"
uvicorn = "^0.24.0"
pydantic = "^2.0.0"
python-dotenv = "^1.0.0"
anthropic = "latest"
openai = "latest"
motor = "latest"  # MongoDB async driver
sqlalchemy = "latest"
alembic = "latest"  # Database migrations
```

## Development Tools
- **pytest**: Testing framework with async support
- **black**: Code formatting
- **flake8**: Linting
- **poetry**: Dependency management (pyproject.toml)

## Security & Configuration
- **Environment variables**: All API keys and secrets via .env
- **Pydantic BaseSettings**: Type-safe configuration management
- **JWT**: Authentication tokens
- **Encryption**: Sensitive data protection

## Common Commands

### Development
```bash
# Install dependencies
poetry install

# Run development server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

### Database
```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Initialize database
python -m src.database.init_db
```

### AI Safety
- All AI interactions must go through safety monitoring
- Guardrails enabled by default in production
- Decision logging required for all AI actions