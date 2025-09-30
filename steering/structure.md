# Project Structure

## Directory Organization

```
mgmt-agent-exp/
├── src/                        # Main source code
│   ├── main.py                 # FastAPI application entry point
│   ├── config/                 # Configuration management
│   │   ├── settings.py         # System settings (Pydantic BaseSettings)
│   │   ├── ai_config.py        # AI model configuration and safety
│   │   └── security.py         # Security and encryption settings
│   ├── agents/                 # AI agent implementations
│   │   ├── vending_agent.py    # Main vending machine agent
│   │   ├── procurement_agent.py # Autonomous procurement
│   │   ├── customer_agent.py   # Customer interaction agent
│   │   └── analytics_agent.py  # Data analysis agent
│   ├── models/                 # Pydantic data models
│   │   ├── product.py          # Product and inventory models
│   │   ├── transaction.py      # Transaction and payment models
│   │   ├── customer.py         # Customer profile models
│   │   ├── conversation.py     # NoSQL conversation models
│   │   └── journal_entry.py    # Accounting models
│   ├── services/               # Business logic layer
│   │   ├── inventory_service.py # Inventory management
│   │   ├── payment_service.py  # Payment processing
│   │   ├── conversation_service.py # NoSQL conversation handling
│   │   └── web_search_service.py # Price comparison searches
│   ├── accounting/             # Financial systems
│   │   ├── journal_entry.py    # Automated bookkeeping
│   │   ├── financial_reports.py # P&L, balance sheet generation
│   │   └── management_accounting.py # KPI analysis
│   ├── analytics/              # Data analysis and monitoring
│   │   ├── event_tracker.py    # System event logging
│   │   ├── anomaly_detector.py # Statistical anomaly detection
│   │   └── report_generator.py # Comprehensive reporting
│   ├── api/                    # REST API endpoints
│   │   ├── vending.py          # Vending machine operations
│   │   ├── customer.py         # Customer engagement APIs
│   │   ├── accounting.py       # Financial data APIs
│   │   └── dashboard.py        # Dashboard data APIs
│   └── utils/                  # Shared utilities
│       ├── logger.py           # Secure logging
│       └── validators.py       # Input validation
├── tests/                      # Test suite
│   ├── test_agents.py          # AI agent testing
│   ├── test_vending_flow.py    # End-to-end purchase flow
│   ├── test_ai_safety.py       # Safety and security tests
│   └── conftest.py             # Test configuration
├── data/                       # Data storage
│   ├── accounting/             # SQLite financial data
│   ├── conversations/          # NoSQL conversation logs
│   └── reports/                # Generated reports
├── static/                     # Web dashboard assets
│   └── dashboard/              # Real-time monitoring UI
├── docs/                       # Documentation
└── .kiro/                      # Kiro IDE configuration
    ├── specs/                  # Feature specifications
    └── steering/               # AI assistant guidance
```

## Architecture Principles

### Modular Design
- Each agent is self-contained with clear responsibilities
- Services layer abstracts business logic from API endpoints
- Models define clear data contracts using Pydantic

### AI Safety Integration
- All AI interactions go through safety monitoring
- Guardrails and anomaly detection built into agent base classes
- Decision logging required for audit trails

### Data Separation
- **SQL (SQLite)**: Structured business data (transactions, accounting, inventory)
- **NoSQL (MongoDB/JSON)**: Unstructured conversation logs and AI interactions
- Clear separation allows for optimal storage and querying

### Configuration Management
- Environment-based configuration using .env files
- Type-safe settings with Pydantic BaseSettings
- Separate configs for development, testing, and production

## Naming Conventions

### Files and Directories
- Snake_case for Python files: `vending_agent.py`
- Lowercase for directories: `src/agents/`
- Descriptive names that indicate purpose

### Classes and Functions
- PascalCase for classes: `VendingAgent`, `PaymentService`
- Snake_case for functions and variables: `process_payment()`, `customer_id`
- Async functions prefixed when appropriate: `async def process_transaction()`

### API Endpoints
- RESTful conventions: `/api/v1/vending/purchase`
- Plural nouns for collections: `/customers`, `/transactions`
- Clear versioning: `/api/v1/`

## Import Organization
```python
# Standard library imports
from datetime import datetime
from typing import Dict, List, Optional

# Third-party imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local imports
from src.config.settings import settings
from src.models.transaction import Transaction
from src.services.payment_service import PaymentService
```