# Market Pro: AI-Powered Market Research Analysis Platform

## Product Overview
Market Pro empowers analysts to upload survey data (SPSS, CSV, Excel), run advanced AI-driven analyses, and generate visualizations and insights through a conversational interface.

## Key Features
1. **Data File Upload & Preview**  
   - SPSS (.sav), CSV, Excel import.  
   - Full SPSS metadata JSON export (variable labels, value labels, missing codes, measurement levels).  
   - Tabular preview with filter/search before analysis.  
2. **Intelligent Chat Interface**  
   - Natural language requests (e.g., "run Van Westendrop on the pricing variables").  
   - Deepseek Chat Model for context-aware follow-ups.  
3. **Modular Analysis (MCP Architecture)**  
   - Standardized API endpoints (`/api/analysis/run`) for each MCP.  
   - First MCP: Van Westendrop Price Sensitivity; extendable to Driver Analysis, Segmentation, CBC, etc.  
   - **Conversational variable mapping confirmation:** The system always proposes a variable mapping for required analysis variables. The user must confirm or edit the mapping via chat before analysis runs. Users can confirm with 'yes'/'confirm' or provide a new mapping in natural language. The AgentController and MCPs robustly enforce this conversational flow.
4. **Visualization & Export**  
   - Interactive charts (Recharts) and tables.  
   - **Segmented analysis (e.g., by age group) is shown in a chart carousel/slider UI, with navigation arrows, dot indicators, and keyboard support.**
   - PNG & PPTX downloads directly from UI.  
   - Improved alignment and spacing for multi-segment results.
5. **Automated Insights & Recommendations**  
   - AI-crafted narrative with key findings and prioritized business actions.  
6. **Smart Report Generator**  
   - Slide-by-slide report builder; export as PPTX, PDF, or DOCX.

## Technical Requirements
- **Frontend**: Next.js + Tailwind + Shadcn/UI; three-panel layout (File Upload/Chat, Visualizations, Insights).  
- **Backend**: Node.js (Express/NestJS) for APIs; SPSS parser service; orchestrator service in TypeScript.  
- **AI Agent**: Deepseek model via secure API; orchestration logic to dispatch analysis calls.  
- **MCP Servers**: Python/FastAPI microservices, Dockerized, registered in central registry.  
- **Data Storage**: PostgreSQL for metadata; S3 for raw files; Redis for cache.  
- **Authentication**: JWT + OAuth (GitHub/Google).  

## User Flows
1. **Upload & Validate**: User logs in → uploads data file → previews data & metadata.  
2. **Analysis**: User requests analysis → orchestrator selects MCP → MCP computes & returns results → UI displays charts/tables (**multi-segment results shown in a carousel/slider for easy navigation**) → insights generated.  
3. **Reporting**: User steps through report slides → refines content → exports final report.