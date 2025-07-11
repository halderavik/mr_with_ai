# How to Build a Plug-and-Play MCP (Market Research Control Protocol)

## What is an MCP?
An MCP (Market Research Control Protocol) is a Python class that implements a specific type of market research analysis (e.g., price sensitivity, segmentation, etc.). MCPs are plug-and-play modules: if you follow the interface and conversational flow described here, your MCP will work with our platform—no internal system knowledge required.

---

## Quickstart: Building a New MCP

**Every MCP must:**
- Inherit from `MCPBase` (see `backend/app/services/mcp_base.py`).
- Set a unique `self.name`, `self.required_columns`, and `self.description` in `__init__`.
- Implement the `run(self, data, params)` method.
- **ALWAYS use the provided LLM (`chat_model`) for variable mapping** - this is mandatory.
- Return results in the required format (see below).

### 1. Inherit from MCPBase

```python
from app.services.mcp_base import MCPBase

class MyNewMCP(MCPBase):
    def __init__(self):
        super().__init__()
        self.name = "my_analysis"
        self.required_columns = ["var1", "var2"]
        self.description = "Short description of your analysis."
```

### 2. Implement the `run()` Method

Your `run()` method is called like this:
```python
result = MyNewMCP().run(data, params)
```
- `data`: a pandas DataFrame with the uploaded data.
- `params`: a dict with keys like `column_map`, `column_map_confirmed`, `chat_model`, and `metadata`.

#### The `run()` method must:
1. **ALWAYS propose a variable mapping using LLM** if none is confirmed:
    - **MANDATORY**: Use the provided `chat_model` to suggest which columns to use for each required variable.
    - **DO NOT** use simple heuristics or column name matching - always use LLM for robust mapping.
    - Return a chat message asking the user to confirm or edit the mapping.
2. **Wait for confirmation:**
    - If `params['column_map_confirmed']` is not `True`, return the mapping proposal and do NOT run the analysis.
3. **Run the analysis:**
    - Once mapping is confirmed (or user provides a new mapping), run your analysis using the mapped columns.
4. **Generate visualizations:**
    - Create charts/tables as needed (e.g., using matplotlib, convert to base64 PNG).
    - For segmented analysis, return one chart/table per segment.
5. **Format and return results:**
    - Return a dictionary with keys: `visualizations`, `insights`, `reply`, and `context` (see below).

---

## MANDATORY: LLM-Based Variable Mapping

**Variable mapping MUST be done through LLM - this is not optional.** The LLM approach provides:

- **Robust matching** across different question formats and naming conventions
- **Semantic understanding** of question content and value labels
- **User-friendly mapping** that considers the actual meaning of variables
- **Consistent experience** across all MCPs

### LLM Mapping Template

```python
def propose_mapping_with_llm(self, metadata, chat_model, required_vars):
    """
    MANDATORY: Use LLM to propose variable mapping.
    """
    # Get comprehensive metadata for LLM
    column_labels = metadata.get('column_labels', {})
    value_labels = metadata.get('value_labels', {})
    data_types = metadata.get('data_types', {})
    unique_values = metadata.get('unique_values', {})
    
    # Create comprehensive prompt with all available metadata
    prompt = f"""
    You are a market research expert. I need to map variables for {self.name} analysis.
    
    REQUIRED VARIABLES: {required_vars}
    
    AVAILABLE DATA COLUMNS:
    {json.dumps(column_labels, indent=2)}
    
    VALUE LABELS (possible answers for each question):
    {json.dumps(value_labels, indent=2)}
    
    DATA TYPES:
    {json.dumps(data_types, indent=2)}
    
    UNIQUE VALUES (sample of possible answers):
    {json.dumps(unique_values, indent=2)}
    
    Please analyze the questions and their possible answers to map the required variables.
    Consider the semantic meaning, not just column names.
    
    Reply with ONLY a JSON object mapping each required variable to the best matching column:
    {{"var1": "column_name", "var2": "column_name"}}
    """
    
    try:
        response = chat_model.generate_reply(prompt)
        # Clean and parse the response
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        proposed_map = json.loads(response)
        return proposed_map
    except Exception as e:
        # Fallback to simple mapping if LLM fails
        return self._fallback_mapping(column_labels, required_vars)
```

### Example: Minimal MCP Template with LLM Mapping

```python
from app.services.mcp_base import MCPBase
import pandas as pd
import json

class ExampleMCP(MCPBase):
    def __init__(self):
        super().__init__()
        self.name = "example"
        self.required_columns = ["x", "y"]
        self.description = "Example analysis."

    def run(self, data, params=None):
        metadata = params.get("metadata", {})
        chat_model = params.get("chat_model")

        # 1. MANDATORY: Use LLM to propose mapping if not confirmed
        if not params.get("column_map") or not params.get("column_map_confirmed", False):
            # ALWAYS use LLM for variable mapping
            proposed_map = self.propose_mapping_with_llm(metadata, chat_model, self.required_columns)
            
            return {
                "reply": f"Please confirm the variable mapping: x: {proposed_map['x']}, y: {proposed_map['y']}",
                "context": {
                    "analysis_type": self.name,
                    "proposed_column_map": proposed_map,
                    "variables_used": self.required_columns,
                    "column_map_confirmed": False
                }
            }

        # 2. Run your analysis here using the confirmed mapping
        col_map = params["column_map"]
        # ... analysis logic ...
        # Example: dummy chart and table
        chart = {
            "type": "curve",
            "title": "My Chart",
            "plot_data": "<base64-png>"
        }
        table = {
            "type": "summary",
            "title": "Summary Table",
            "data": [{"metric": "A", "value": 123}]
        }
        return {
            "visualizations": {"charts": [chart], "tables": [table]},
            "insights": "Key findings and recommendations...",
            "reply": "Business-focused summary for the user...",
            "context": {
                "analysis_type": self.name,
                "variables_used": self.required_columns,
                "column_map_confirmed": True
            }
        }
```

---

## Segmentation Support (Optional)

If your analysis supports segmentation (e.g., by age, gender):
- **MANDATORY**: Use the LLM to find the correct segmentation variable from metadata.
- Run your analysis for each segment and return a chart/table per segment.

**Example:**
```python
def handle_segmentation(self, data, params, metadata):
    question = params.get("question", "")
    if "by" in question.lower():
        # MANDATORY: Use LLM to find the segmentation variable
        column_labels = metadata.get('column_labels', {})
        value_labels = metadata.get('value_labels', {})
        
        prompt = f"""
        Given these questions and their possible answers, which column best matches the segmentation request?
        
        Segmentation requested: {question}
        
        Available questions:
        {json.dumps(column_labels, indent=2)}
        
        Value labels (possible answers):
        {json.dumps(value_labels, indent=2)}
        
        Reply with ONLY the EXACT column name that best represents the segmentation variable.
        """
        
        seg_var = params['chat_model'].generate_reply(prompt).strip()
        value_labels = metadata.get('value_labels', {}).get(seg_var, {})
        segments = {}
        for value, label in value_labels.items():
            segment_data = data[data[seg_var] == value]
            if len(segment_data) > 0:
                segments[label] = segment_data
        return segments
    return {"Overall": data}
```

---

## Required Output Format
Your `run()` method must always return a dictionary with these keys:
- `reply`: A chat message for the user (summary, next steps, or mapping prompt).
- `visualizations`: Dict with `charts` and/or `tables` (see below).
    - For segmented analysis: one chart/table per segment.
- `insights`: Narrative insights for the user.
- `context`: Dict with at least `analysis_type`, `variables_used`, and mapping status.

**Example:**
```python
{
    "reply": "Here are your results...",
    "visualizations": {
        "charts": [
            {"type": "curve", "title": "My Chart - 18-24", "plot_data": "<base64-png>"},
            {"type": "curve", "title": "My Chart - 25-34", "plot_data": "<base64-png>"},
        ],
        "tables": [
            {"type": "summary", "title": "Summary Table - 18-24", "data": [{"metric": "A", "value": 123}]},
            {"type": "summary", "title": "Summary Table - 25-34", "data": [{"metric": "A", "value": 456}]},
        ]
    },
    "insights": "Key findings and recommendations...",
    "context": {
        "analysis_type": "example",
        "variables_used": ["x", "y"],
        "column_map_confirmed": True,
        "segments": ["18-24", "25-34"]
    }
}
```

---

## Best Practices & Tips
- **MANDATORY: Always use LLM for variable mapping** - never use simple heuristics or column name matching.
- **Be explicit about required variables** and check for their presence in the mapping and data.
- **Never run analysis without user confirmation** of the mapping.
- **Use comprehensive metadata** in LLM prompts for better mapping accuracy.
- **Support segmentation** when it makes sense for your analysis.
- **Return clear, actionable replies** for the chat agent (use LLM to polish if needed).
- **Log debug info** for easier troubleshooting.
- **Handle metadata gracefully** - check if fields exist before using them.
- **For segmented analysis, return one chart/table per segment for carousel display in the frontend.**
- **Always include error handling** for LLM responses and JSON parsing.

---

## Advanced: See a Full Example
For a full-featured MCP with LLM-based variable mapping and segmentation, see:
- `backend/app/mcp/van_westendorp.py` - Van Westendorp price sensitivity analysis
- `backend/app/mcp/choice_based_conjoint.py` - Choice-based conjoint analysis with hierarchical Bayesian estimation

---

## FAQ
**Q: Do I need to know anything about the rest of the system?**
A: No. If you follow this interface and conversational flow, your MCP will work out of the box.

**Q: Why must I use LLM for variable mapping?**
A: LLM-based mapping provides robust semantic understanding across different question formats, naming conventions, and data structures. It ensures consistent, user-friendly variable mapping that considers the actual meaning of variables.

**Q: How do I use an LLM to propose a mapping?**
A: You will receive a `chat_model` in `params`—call `chat_model.generate_reply(prompt)` with your prompt. Always include comprehensive metadata (column labels, value labels, data types) in your prompt.

**Q: What if the LLM fails to respond or returns invalid JSON?**
A: Always include error handling and fallback mechanisms. Parse the LLM response carefully and provide a simple fallback mapping if needed.

**Q: How do I generate a chart?**
A: Use matplotlib or your preferred library, save to a PNG in memory, and base64-encode it for the `plot_data` field.

**Q: How do I handle segmentation?**
A: Use the LLM with metadata to identify segmentation variables and run your analysis separately for each segment.

**Q: What metadata is available?**
A: Column names, column labels (questions), value labels, variable labels, measurement levels, data types, unique values, basic statistics, and more.

---

## See Also
- Example: `van_westendorp.py` for a full-featured MCP with LLM-based mapping and segmentation
- Example: `choice_based_conjoint.py` for CBC analysis with hierarchical Bayesian estimation
- Main architecture: `ARCHITECTURE.md`
- API/Chat flow: `README.md` 