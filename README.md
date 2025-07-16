# Adaptive RAG Example

## Setup

The service requires LLM API keys. Copy `.env.example`, rename it to `.env` in the project root directory, and set your OpenAI API key and Tavily API key.

### Backend

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start backend service:
   ```bash
   uvicorn main:app --reload --port 8000 --host 0.0.0.0
   ```

### Frontend

Frontend source code is in the `ui` directory.

1. Change to directory `ui` and install dependencies:

   ```bash
   cd ui
   npm install
   ```

2. Start frontend server:
   ```bash
   npm run dev
   ```
