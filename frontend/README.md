# AI Market Research Platform (Ai_mr_v0)

A full-stack AI-powered market research platform for interactive data analysis, supporting SPSS, CSV, and Excel files. Users can upload data, preview and confirm variable mappings (powered by LLM/Deepseek), and run advanced analyses (e.g., Van Westendorp) with interactive chat and visualization.

---

## Features
- **Data Upload:** Supports SPSS (.sav), CSV, and Excel files. Extracts both data and metadata (variable descriptions).
- **AI/LLM Variable Mapping:** Uses Deepseek LLM to interpret variable descriptions and propose mappings for analysis (e.g., price sensitivity variables for Van Westendorp).
- **Interactive Confirmation:** Before running any analysis, the system asks the user to confirm or edit the proposed variable mapping via chat.
- **Analysis Engine (MCP):** Modular MCP servers (e.g., VanWestendorpMCP) run the analysis using the confirmed variables.
- **Modern Frontend:** Next.js app with file upload, data preview, chat interface, and interactive results (charts, tables, insights).
- **Backend API:** FastAPI backend with robust pipeline for data, metadata, and LLM-powered analysis.
- **Interactive Visualizations:** Advanced charting capabilities using Chart.js for data visualization and analysis.
- **Segmented Analysis Carousel:** For analyses segmented by group (e.g., age, gender), all charts are shown in a modern carousel/slider UI with navigation arrows, dot indicators, and keyboard support. Improved alignment and spacing for multi-segment results.

---

## Getting Started

### Backend (FastAPI)
1. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. **Run the backend server:**
   ```bash
   uvicorn app.main:app --reload
   ```
3. **Environment:**
   - Configure `.env` for API keys (e.g., Deepseek) and settings.

### Frontend (Next.js)
1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   # Install visualization dependencies
   npm install chart.js react-chartjs-2
   ```
2. **Run the frontend server:**
   ```bash
   npm run dev
   ```
3. **Open [http://localhost:3000](http://localhost:3000) in your browser.**

---

## Dependencies

### Frontend Dependencies
- **Core:**
  - Next.js 15.3.2
  - React 19
  - TypeScript
  - Tailwind CSS
  - Radix UI Components
  - Shadcn UI

- **Visualization:**
  - Chart.js - For interactive charts and graphs
  - react-chartjs-2 - React wrapper for Chart.js
  - Features:
    - Line charts for price sensitivity analysis
    - Interactive tooltips
    - Custom annotations
    - Responsive design
    - Export capabilities
    - **Carousel/slider for multi-segment results**

### Backend Dependencies
- FastAPI
- Python 3.11+
- Pandas
- Pyreadstat (for SPSS files)
- SQLAlchemy
- Pydantic

---

## How It Works
1. **Upload Data:** User uploads a data file (SPSS, CSV, Excel). Backend extracts data and metadata.
2. **Preview & Variable Mapping:** User previews data and metadata. When requesting analysis, the backend uses LLM to propose variable mappings based on metadata.
3. **User Confirmation:** The system asks the user to confirm or edit the mapping via chat before running the analysis.
4. **Run Analysis:** Once confirmed, the MCP server runs the analysis and returns results (tables, charts, insights).
5. **Interactive Results:** Results are shown in the frontend with interactive charts, tables, and business insights. **For segmented analyses, all charts are shown in a carousel/slider for easy navigation.**

---

## AI/LLM-Powered Variable Mapping
- The backend uses Deepseek (or any LLM) to interpret variable descriptions from metadata (especially for SPSS files) and propose the correct columns for analysis.
- The user is always asked to confirm or edit the mapping before analysis is run, ensuring accuracy and transparency.

---

## Project Structure
- `backend/` - FastAPI app, MCP servers, LLM integration, data/metadata handling
- `frontend/` - Next.js app, file upload, chat, data preview, analysis UI
  - `src/components/` - Reusable UI components
    - `analysis-chart.tsx` - Chart.js integration for data visualization
    - `analysis-table.tsx` - Tabular data display
    - Other UI components

---

## Contributing
- Fork the repo and create a feature branch.
- Submit pull requests for review.

---

## License
MIT
