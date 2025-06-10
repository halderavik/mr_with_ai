# AI Market Research Analysis Tool

A modern web application for analyzing market research data, with a focus on SPSS files. This tool provides a user-friendly interface for uploading, previewing, and analyzing market research data.

## Features

- **File Upload Support**
  - SPSS (.sav) files
  - CSV files
  - Excel files (.xlsx, .xls)
  - Drag and drop interface
  - Progress tracking
  - File validation

- **Data Preview**
  - Tabular view of uploaded data
  - First 10 rows preview
  - Column headers
  - Data type detection

- **Metadata Analysis**
  - SPSS metadata extraction
  - Variable labels
  - Value labels
  - Missing values
  - Column formats
  - Column widths
  - Column alignments

- **Advanced Analysis**
  - Van Westendorp Price Sensitivity Analysis
    - Point of Marginal Cheapness (PMC)
    - Point of Marginal Expensiveness (PME)
    - Optimal Price Point (OPP)
    - Price sensitivity curves
    - Interactive visualizations
    - Detailed insights and recommendations
  - More analyses coming soon...
  - Van Westendorp Price Sensitivity Analysis
    - Point of Marginal Cheapness (PMC)
    - Point of Marginal Expensiveness (PME)
    - Optimal Price Point (OPP)
    - Price sensitivity curves
    - Interactive visualizations
    - Detailed insights and recommendations
    - **Conversational variable mapping confirmation:**
      - The backend proposes a variable mapping for required analysis variables (e.g., too_cheap, bargain, getting_expensive, too_expensive).
      - The user must confirm or edit the mapping via chat before analysis runs.
      - The AgentController and MCPs robustly handle this flow, ensuring the analysis only runs after confirmation.
      - Users can confirm with 'yes'/'confirm' or provide a new mapping in natural language.
  - More analyses coming soon...

- **Modern UI**
  - Clean, responsive design
  - Dark/light mode support
  - Tabbed interface
  - Loading states
  - Error handling
  - Real-time visualization updates
  - Interactive charts and tables

## Tech Stack

### Frontend
- Next.js 15.3.2
- React 19
- TypeScript
- Tailwind CSS
- Radix UI Components
- Shadcn UI

### Backend
- FastAPI
- Python 3.11+
- Pandas
- Pyreadstat (for SPSS files)
- SQLAlchemy
- Pydantic
- Matplotlib (for visualization generation)

## Architecture

### MCP (Market Research Control Protocol) Architecture
- Modular analysis engines that generate and push visualizations
- Each MCP is responsible for:
  - Data validation and preprocessing
  - Analysis execution
  - Visualization generation
  - Results formatting
  - Pushing visualizations to frontend

### Visualization Pipeline
1. MCP generates visualizations using matplotlib
2. Visualizations are converted to base64-encoded PNG
3. Results are sent to frontend via API
4. Frontend displays visualizations in real-time
5. Interactive features (zoom, pan, export) available

## Prerequisites

- Python 3.11 or higher
- Node.js 18 or higher
- npm or yarn
- Git

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai_mr_v0
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate

pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

## Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

3. Open your browser and navigate to:
```
http://localhost:3000
```

## Project Structure

```
ai_mr_v0/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── models/
│   │   ├── services/
│   │   ├── mcp/           # Market Research Control Protocol servers
│   │   └── utils/
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   └── styles/
│   ├── public/
│   └── package.json
└── README.md
```

## API Documentation

### Base URL
```
http://localhost:8000
```

### Authentication
Currently, the API does not require authentication. All endpoints are publicly accessible.

### Endpoints

#### 1. File Upload
```http
POST /api/upload
```

Upload a data file for analysis.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: The file to upload (supported formats: .sav, .csv, .xlsx, .xls)

**Response:**
```json
{
  "dataset_id": "string",
  "filename": "string",
  "preview_rows": [
    {
      "column1": "value1",
      "column2": "value2"
    }
  ],
  "metadata": {
    "column_names": ["string"],
    "column_labels": ["string"],
    "value_labels": {
      "column1": {
        "value1": "label1"
      }
    },
    "missing_values": {
      "column1": ["value1"]
    },
    "column_formats": {
      "column1": "format1"
    },
    "column_widths": {
      "column1": 10
    },
    "column_alignments": {
      "column1": "left"
    }
  }
}
```