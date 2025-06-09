# AI Market Research Platform Architecture

---

## Overview
This platform enables users to upload market research data (SPSS, CSV, Excel), preview and confirm variable mappings (powered by LLM/Deepseek), and run advanced analyses (e.g., Van Westendorp) with interactive chat and visualization. The architecture is modular, scalable, and designed for explainability and extensibility.

---

## High-Level Architecture

```mermaid
graph TD
    A[User (Browser)] -->|Upload Data, Chat, Confirm| B[Frontend (Next.js)]
    B -->|API Calls| C[Backend (FastAPI)]
    C -->|LLM Prompt| D[Deepseek/LLM]
    C -->|Analysis| E[MCP Servers]
    C -->|Data/Metadata| F[Data Storage]
    E -->|Results| C
    D -->|Variable Mapping| C
    C -->|Results| B
```

---

## Components

### 1. Frontend (Next.js)
- **File Upload:** Supports SPSS, CSV, Excel. Sends data to backend.
- **Data Preview:** Shows preview and metadata after upload.
- **Chat Interface:** User requests analysis, confirms variable mapping, and receives results.
- **Interactive Results:** Displays charts, tables, and insights.

### 2. Backend (FastAPI)
- **API Layer:** Handles file upload, analysis requests, and chat.
- **Data Loader:** Loads data and extracts metadata (esp. for SPSS).
- **Agent Controller:** Orchestrates analysis requests, LLM prompts, and MCP server calls.
- **LLM Integration:** Uses Deepseek (or other LLM) to interpret variable descriptions and propose mappings.
- **MCP Servers:** Modular analysis engines (e.g., VanWestendorpMCP) that:
    - Use LLM-proposed variable mapping
    - Ask user for confirmation if needed
    - Run analysis and return results

### 3. Data Flow
1. **Upload:** User uploads data file → Backend extracts data + metadata
2. **Preview:** Frontend displays preview and metadata
3. **Analysis Request:** User requests analysis via chat
4. **LLM Variable Mapping:** Backend uses LLM to propose variable mapping
5. **User Confirmation:** User confirms/edits mapping via chat
6. **Analysis Execution:** MCP server runs analysis with confirmed mapping
7. **Results:** Backend returns results (tables, charts, insights) to frontend

---

## Key Design Principles
- **Modularity:** Each analysis type is a separate MCP server; easy to add new analyses.
- **Explainability:** LLM explains variable mapping and analysis steps to the user.
- **Interactivity:** User confirms variable mapping before analysis runs.
- **Extensibility:** Supports new file types, LLMs, and analysis modules.

---

## Example Sequence
1. **User uploads SPSS file** → Backend extracts data and metadata
2. **User requests "Van Westendorp analysis"**
3. **Agent Controller**:
    - Uses LLM to map variables from metadata
    - Proposes mapping to user via chat
    - Waits for user confirmation
4. **User confirms mapping**
5. **MCP runs analysis** and returns results
6. **Frontend displays results** (charts, tables, insights)

---

## Technologies Used
- **Frontend:** Next.js, React, Tailwind CSS
- **Backend:** FastAPI, Pydantic, Uvicorn
- **AI/LLM:** Deepseek (or pluggable LLM)
- **Data Analysis:** Pandas, Numpy, Matplotlib
- **File Support:** SPSS (pyreadstat), CSV, Excel

---

## Extending the Platform
- **Add new MCP:** Create a new MCP class in `backend/app/mcp/` and register it.
- **Add new analysis type:** Update LLM prompt and frontend options.
- **Swap LLM:** Replace Deepseek integration in Agent Controller.

---

## Diagram: Data & Control Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant B as Backend
    participant L as LLM
    participant M as MCP
    U->>F: Upload data file
    F->>B: POST /api/upload
    B->>F: Data preview + metadata
    U->>F: Request analysis via chat
    F->>B: POST /api/chat (with metadata)
    B->>L: Prompt for variable mapping
    L-->>B: Proposed mapping
    B->>F: Ask user to confirm mapping
    U->>F: Confirm mapping
    F->>B: POST /api/chat (confirmed mapping)
    B->>M: Run analysis
    M-->>B: Results
    B->>F: Results (charts, tables, insights)
    F->>U: Display results
```

---

## License
MIT 