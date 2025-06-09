# Market Pro: Implementation Tasks

## Priority 1: File Upload & SPSS Metadata
- [x] **Frontend**: Add `FileUploader` component to chat interface (supports `.sav`, `.csv`, `.xlsx`).  
- [x] **Backend**: Implement `/api/files/upload` endpoint (multer) and `/api/files/:id/metadata` to extract & return full SPSS metadata JSON.  
- [x] **Validation**: Render preview table + metadata panel in UI; allow user confirmation.

## Priority 2: AI Orchestrator & Chat Integration
- [x] Integrate Deepseek chat model; manage conversation state.  
- [x] Connect chat input to orchestrator: detect "run <analysis>" commands.  
- [x] Enable variable selection follow-ups based on metadata JSON.

## Priority 3: Van Westendrop MCP
- [x] Scaffold Python FastAPI service in `mcp-servers/van-westendrop`.  
- [x] Implement price sensitivity calculations and endpoints (`/run`).  
- [x] Containerize & register with MCP Registry; health checks.

## Priority 4: Frontend Results Panels
- [x] VisualizationPanel: render charts/tables with export buttons.  
- [x] InsightsPanel: display AI-generated narrative + priority tags + export.

## Priority 5: Orchestration & End-to-End Testing
- [x] Wire orchestrator: upload → metadata → chat → MCP call → UI output.  
- [ ] Write integration tests for full flow; unit tests for SPSS parser.
  - TODO: Create test directory and implement tests

## DevOps & CI/CD
- [ ] GitHub Actions: build/test for frontend, backend, and MCPs.  
  - TODO: Create GitHub Actions workflow files
- [ ] Docker Compose for local dev; Kubernetes manifests for production.  
  - TODO: Create Docker and Kubernetes configuration files
- [ ] Monitoring: Prometheus + Grafana for system health and MCP latency.
  - TODO: Set up monitoring infrastructure

## Future MCP Servers
- **Driver Analysis**, **Segmentation**, **CBC**, **MaxDiff**: replicate the Van Westendrop pattern:  
  - Create new folder under `mcp-servers`.  
  - Expose FastAPI endpoint; containerize; register.  
  - Update orchestrator's registry with new capabilities.