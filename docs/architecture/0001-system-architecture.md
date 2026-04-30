# Architecture Decision Record: Advanced Credit Card Fraud Detection System

## Title
Layered Architecture for Credit Card Fraud Detection

## Status
Accepted

## Context
The Advanced Credit Card Fraud Detection System requires a scalable, maintainable, and highly reliable architecture to process transactions in real-time, train machine learning models, and orchestrate complex fraud detection workflows (including classical ML, Quantum ML, Graph Neural Networks, and Federated Learning). We need to ensure clear separation of concerns, facilitate team collaboration, and allow independent scaling of components.

## Decision
We will adopt a layered architecture based on the following principles:

1. **Separation of Concerns:** Code is organized into distinct layers:
   - **UI/Dashboard (`dashboard/`):** Data-agnostic UI components built with Streamlit for monitoring and analytics.
   - **Business Logic (`src/block/`, `src/fraud_detection/`):** Contains the core logic for fraud detection, payment validation, device fingerprinting, and risk score calculation.
   - **Data Processing & ML (`src/analytics/`, `src/models/`):** Responsible for data ingestion, preprocessing, and training/inference of various models (Quantum, GNN, Ensemble, Streaming).
   - **Utilities (`src/utils/`, `src/telemetry/`):** Shared utilities for error handling, logging, and telemetry.

2. **Code Organization (Block, Helpers, Utils):**
   - **Block:** Core business logic (e.g., `fraud_detection_block.py`) is separated from presentation and data fetching.
   - **Helpers:** Project-specific utilities (e.g., specific data preprocessing).
   - **Utils:** Highly reusable, shared code (e.g., `error_handling.py`, `logging_utils.py`).

3. **Telemetry and Monitoring:**
   - Telemetry is distinctly separated from business analytics. We use `src/telemetry/` to monitor system health, request latency, and error rates.

4. **Error Handling:**
   - Strict error handling conventions are enforced (e.g., `src/utils/error_handling.py`). No silent failures are allowed. All exceptions must be logged with full stack traces.

5. **Environment Isolation:**
   - Production and development environments are cleanly isolated. Dependency management ensures reproducible builds.

## Consequences
- **Positive:**
  - High maintainability and testability due to the clear separation of concerns.
  - Easier onboarding for new engineers.
  - Flexibility to swap out underlying ML models without affecting the UI or core business logic orchestration.
- **Negative:**
  - Slight overhead in passing data between layers.
  - Requires strict adherence to architectural guidelines during code reviews.

## Implementation Guidelines
- Dependencies must flow downward (UI → BLoC/Business Logic → Data/ML).
- All new features must handle the "sunny case" first, followed by robust edge case and failure scenario handling.
- Comments must explain "why" the code does what it does, not just "what" it does.
