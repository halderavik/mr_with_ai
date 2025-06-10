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

- **Modern UI**
  - Clean, responsive design
  - Dark/light mode support
  - Tabbed interface
  - Loading states
  - Error handling

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

**Status Codes:**
- `200 OK`: File uploaded successfully
- `400 Bad Request`: Invalid file type or size
- `413 Payload Too Large`: File size exceeds limit
- `500 Internal Server Error`: Server-side error

**Error Response:**
```json
{
  "detail": "Error message"
}
```

#### 2. Data Preview
```http
GET /api/preview/{dataset_id}
```

Get a preview of the uploaded dataset.

**Parameters:**
- `dataset_id` (path parameter): ID of the dataset to preview

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
    }
  }
}
```

**Status Codes:**
- `200 OK`: Preview retrieved successfully
- `404 Not Found`: Dataset not found
- `500 Internal Server Error`: Server-side error

#### 3. Dataset Analysis
```http
POST /api/analyze/{dataset_id}
```

Analyze the uploaded dataset.

**Parameters:**
- `dataset_id` (path parameter): ID of the dataset to analyze

**Request Body:**
```json
{
  "analysis_type": "string",
  "parameters": {
    "key": "value"
  }
}
```

**Response:**
```json
{
  "analysis_id": "string",
  "status": "string",
  "results": {
    "summary": {
      "total_rows": 0,
      "total_columns": 0,
      "missing_values": 0
    },
    "statistics": {
      "column1": {
        "mean": 0,
        "median": 0,
        "mode": "value",
        "std_dev": 0
      }
    }
  }
}
```

**Status Codes:**
- `200 OK`: Analysis completed successfully
- `400 Bad Request`: Invalid analysis parameters
- `404 Not Found`: Dataset not found
- `500 Internal Server Error`: Server-side error

#### 4. Van Westendorp Analysis
```http
POST /api/analyze/{dataset_id}/van-westendorp
```

Run a Van Westendorp price sensitivity analysis on the dataset.

**Parameters:**
- `dataset_id` (path parameter): ID of the dataset to analyze

**Request Body:**
```json
{
  "too_cheap": "column_name",
  "bargain": "column_name",
  "getting_expensive": "column_name",
  "too_expensive": "column_name"
}
```

**Response:**
```json
{
  "price_points": {
    "pmc": 0.0,
    "pme": 0.0,
    "opp": 0.0,
    "price_sensitivity": 0.0
  },
  "curves": {
    "price_grid": [0.0],
    "too_cheap": [0.0],
    "too_expensive": [0.0],
    "bargain": [0.0],
    "getting_expensive": [0.0]
  },
  "insights": "string"
}
```

**Status Codes:**
- `200 OK`: Analysis completed successfully
- `400 Bad Request`: Invalid column mapping
- `404 Not Found`: Dataset not found
- `500 Internal Server Error`: Server-side error

### Rate Limiting
- Maximum file size: 100MB
- Maximum concurrent uploads: 5
- Rate limit: 100 requests per minute

### Error Handling
All error responses follow this format:
```json
{
  "detail": "Error message",
  "code": "ERROR_CODE",
  "timestamp": "ISO-8601 timestamp"
}
```

Common error codes:
- `INVALID_FILE_TYPE`: Unsupported file format
- `FILE_TOO_LARGE`: File exceeds size limit
- `DATASET_NOT_FOUND`: Dataset ID not found
- `INVALID_ANALYSIS_TYPE`: Unsupported analysis type
- `SERVER_ERROR`: Internal server error

### WebSocket Events
The API also supports real-time updates via WebSocket:

```http
WS /ws/analysis/{analysis_id}
```

**Events:**
- `analysis_started`: Analysis has begun
- `analysis_progress`: Progress update
- `analysis_completed`: Analysis finished
- `analysis_error`: Error occurred

**Example WebSocket Message:**
```json
{
  "event": "analysis_progress",
  "data": {
    "progress": 50,
    "status": "Processing data"
  }
}
```

### CORS Configuration
The API supports CORS with the following configuration:
- Allowed Origins: `http://localhost:3000`
- Allowed Methods: GET, POST, OPTIONS
- Allowed Headers: Content-Type, Authorization
- Max Age: 3600 seconds

## Development

### Backend Development
- Follow PEP 8 style guide
- Use type hints
- Write docstrings for all functions
- Create unit tests for new features

### Frontend Development
- Use TypeScript for type safety
- Follow component-based architecture
- Use Tailwind CSS for styling
- Implement responsive design

## Testing

### Backend Tests
```bash
cd backend
pytest
```

### Frontend Tests
```bash
cd frontend
npm test
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Acknowledgments

- Pyreadstat for SPSS file support
- FastAPI for the backend framework
- Next.js for the frontend framework
- Shadcn UI for the component library 