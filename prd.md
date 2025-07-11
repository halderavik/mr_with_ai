# Market Pro: AI-Powered Market Research Analysis Platform

## Product Overview
Market Pro empowers analysts to upload survey data (SPSS, CSV, Excel), run advanced AI-driven analyses, and generate visualizations and insights through a conversational interface.

## Key Features
1. **Data File Upload & Preview**  
   - SPSS (.sav), CSV, Excel import.  
   - Full SPSS metadata JSON export (variable labels, value labels, missing codes, measurement levels).  
   - Tabular preview with filter/search before analysis.  
   - Comprehensive data type detection and statistics extraction.
2. **Intelligent Chat Interface**  
   - Natural language requests (e.g., "run Van Westendorp on the pricing variables", "run CBC analysis on the conjoint data").  
   - Deepseek Chat Model for context-aware follow-ups.  
   - LLM-powered variable mapping for robust analysis setup.
3. **Modular Analysis (MCP Architecture)**  
   - Standardized API endpoints (`/api/analysis/run`) for each MCP.  
   - **Van Westendorp Price Sensitivity Analysis**: Point of Marginal Cheapness (PMC), Point of Marginal Expensiveness (PME), Optimal Price Point (OPP), price sensitivity curves with cumulative distributions.
   - **Choice-Based Conjoint (CBC) Analysis**: Hierarchical Bayesian estimation with multiple MCMC chains, part-worth utilities, importance scores, market simulations, and preference predictions.
   - Extendable to Driver Analysis, Segmentation, MaxDiff, etc.  
   - **Conversational variable mapping confirmation:** The system always proposes a variable mapping for required analysis variables. The user must confirm or edit the mapping via chat before analysis runs. Users can confirm with 'yes'/'confirm' or provide a new mapping in natural language. The AgentController and MCPs robustly enforce this conversational flow.
4. **Visualization & Export**  
   - Interactive charts (Recharts) and tables.  
   - **Segmented analysis (e.g., by age group) is shown in a chart carousel/slider UI, with navigation arrows, dot indicators, and keyboard support.**
   - PNG & PPTX downloads directly from UI.  
   - Improved alignment and spacing for multi-segment results.
   - CBC-specific visualizations: utility plots, importance charts, market simulation results.
5. **Automated Insights & Recommendations**  
   - AI-crafted narrative with key findings and prioritized business actions.  
   - CBC insights include feature importance rankings, market share predictions, and pricing recommendations.
6. **Smart Report Generator**  
   - Slide-by-slide report builder; export as PPTX, PDF, or DOCX.

## Technical Requirements
- **Frontend**: Next.js + Tailwind + Shadcn/UI; three-panel layout (File Upload/Chat, Visualizations, Insights).  
- **Backend**: FastAPI (Python) for APIs; SPSS parser service; orchestrator service with MCP discovery.  
- **AI Agent**: Deepseek model via secure API; orchestration logic to dispatch analysis calls.  
- **MCP Servers**: Python/FastAPI microservices, registered in central registry with automatic discovery.  
- **Data Storage**: File-based storage with metadata caching; Redis for cache (future).  
- **Authentication**: JWT + OAuth (GitHub/Google) - planned for future.  

## User Flows
1. **Upload & Validate**: User logs in → uploads data file → previews data & metadata.  
2. **Analysis**: User requests analysis → orchestrator selects MCP → LLM maps variables → user confirms mapping → MCP computes & returns results → UI displays charts/tables (**multi-segment results shown in a carousel/slider for easy navigation**) → insights generated.  
3. **Reporting**: User steps through report slides → refines content → exports final report.

## Current MCP Capabilities
- **Van Westendorp MCP**: Price sensitivity analysis with segmentation support
- **Choice-Based Conjoint MCP**: Hierarchical Bayesian estimation with market simulations
- **Format-Agnostic Data Handling**: Automatically detects and converts various CBC data formats (wide, stacked, long) to required tensor format
- **Robust Error Handling**: Comprehensive validation and user-friendly error messages