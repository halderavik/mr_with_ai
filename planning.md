# Market Pro Implementation Planning

This document outlines the updated planning for Market Pro, including the new file upload feature, SPSS metadata handling, and the AI orchestrator design.

## Project Overview
Market Pro is an advanced market research analysis platform that integrates AI capabilities with specialized analytical tools, enabling users to upload survey data (SPSS/CSV/Excel), run modular analyses, and receive interactive visualizations and insights.

## Folder & Module Structure
market-pro/
├── backend/
│   ├── src/
│   │   ├── controllers/
│   │   │   └── fileUploadController.ts       # handles file uploads & metadata
│   │   ├── routes/
│   │   │   └── fileRoutes.ts                # POST /api/files, GET /api/files/:id/metadata
│   │   ├── services/
│   │   │   ├── spssProcessor.ts             # full SPSS parser, metadata extractor
│   │   │   └── dataService.ts               # data loading, caching
│   │   ├── orchestrator/
│   │   │   └── analysisOrchestrator.ts      # AI agent controller
│   │   └── mcp/                             # Microservice stubs for analyses
│   │       └── vanWestendrop/
│   │           └── app.py                   # FastAPI Van Westendrop MCP
│   ├── Dockerfile
│   └── docker-compose.yml
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUploader.tsx            # UI for SPSS/CSV/Excel upload
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── VisualizationPanel.tsx
│   │   │   └── InsightsPanel.tsx
│   │   ├── pages/
│   │   │   └── analysis/[projectId].tsx    # integrates FileUploader & Chat
│   │   └── utils/
│   │       └── api.ts                      # fetch wrappers for upload & analysis
│   └── package.json
└── mcp-servers/
└── van-westendrop/
├── Dockerfile
└── main.py                           # Python FastAPI service

## Technology Stack Updates
- **File Handling**: `multer` on Node.js + custom SPSS metadata parser (`node-spss`) to produce complete JSON metadata (variable labels, value labels, computed variables) and raw data arrays.
- **AI Orchestrator**: Deepseek Chat Model as primary LLM; orchestrator routes analysis requests to MCPs, manages context, and answers follow-up questions.
- **MCP Framework**: Python/FastAPI services per analysis type; containerized and registered via a central MCP registry for discovery.

## Development Phases (Revised)
1. **MVP Foundation**
   - Set up file upload backend & UI; verify SPSS metadata JSON extraction and preview.  
   - Integrate orchestrator stub; connect chat input to file metadata service.  
2. **Core Analyses**
   - Build Van Westendrop MCP as first microservice.  
   - Validate end-to-end: upload → metadata preview → “run van-westendrop” → charts + insights.  
3. **Agent & UI Polishing**
   - Deepseek integration; refine prompt templates.  
   - Implement full chat-driven variable selection and follow-up.  
4. **Scale & Extend**
   - Add Driver Analysis, Segmentation MCPs.  
   - Harden SPSS parser, handle large datasets with chunking.  
5. **Enterprise & Reports**
   - Build Smart Report Generator module.  
   - Implement security, monitoring, and CI/CD pipelines.