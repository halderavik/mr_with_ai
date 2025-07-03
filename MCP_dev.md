# How to Build a Plug-and-Play MCP (Market Research Control Protocol) for the AI Market Research Platform

## What is an MCP?
An MCP (Market Research Control Protocol) is a Python class that implements a specific type of market research analysis (e.g., price sensitivity, segmentation, etc.). MCPs are plug-and-play modules: if you follow the interface and conversational flow described here, your MCP will work with our platform—no internal system knowledge required.

---

## How the Workflow Works
1. **User uploads data and requests an analysis via chat.**
2. **Your MCP proposes a variable mapping** (e.g., which columns to use for each required variable).
3. **The user confirms or edits the mapping via chat.**
4. **Your MCP runs the analysis only after confirmation.**
5. **Your MCP returns results, visualizations, and insights in a standard format.**

**All communication is via JSON and chat messages.**

---

## Enhanced Metadata Support

### Available Metadata
Your MCP will receive comprehensive metadata in the `params` dictionary:

```python
metadata = params.get("metadata", {})
# Available fields:
# - columns: List of column names
# - column_labels: Dict mapping column names to question text
# - value_labels: Dict mapping column names to value labels
# - variable_labels: Dict mapping column names to variable descriptions
# - variable_measure: Dict mapping column names to measurement levels
# - variable_formats: Dict mapping column names to format information
```

### LLM-Powered Variable Matching
Use the provided `chat_model` to match user requests with available variables:

```python
def find_segmentation_variable(self, metadata, chat_model, request):
    """Find segmentation variable using LLM."""
    column_labels = metadata.get('column_labels', {})
    
    prompt = f"""
    Given these questions from the survey, find the one that best matches the segmentation request.
    Segmentation requested: {request}
    
    Available questions:
    {json.dumps(column_labels, indent=2)}
    
    Reply with the EXACT column name that best matches.
    """
    
    response = chat_model.generate_reply(prompt)
    return response.strip()
```

---

## What You Need to Do

### 1. MCP Class Requirements
- **Inherit from `MCPBase`** (provided by the platform).
- **Set these attributes:**
  - `self.name`: Unique string key for your analysis (e.g., "vanwestendorp").
  - `self.required_columns`: List of variable roles your analysis needs (e.g., `["too_cheap", "bargain", ...]`).
  - `self.description`: Short description of your analysis.

### 2. Implement the `run()` Method
This is the only method you must implement. It will be called like:
```python
result = MyMCP().run(data, params)
```
- `data`: a pandas DataFrame with the uploaded data.
- `params`: a dict with keys like `column_map`, `column_map_confirmed`, `chat_model`, and `metadata`.

#### Your `run()` method must:
1. **Propose a variable mapping** if none is confirmed:
    - Use LLM or heuristics to suggest which columns to use for each required variable.
    - Return a chat message asking the user to confirm or edit the mapping.
2. **Wait for confirmation:**
    - If `params['column_map_confirmed']` is not `True`, return the mapping proposal and do NOT run the analysis.
3. **Run the analysis:**
    - Once mapping is confirmed (or user provides a new mapping), run your analysis using the mapped columns.
4. **Generate visualizations:**
    - Create charts/tables as needed (e.g., using matplotlib, convert to base64 PNG).
    - **For segmented analysis:**
        - Return one chart/table per segment (e.g., one chart for each age group).
        - The frontend will display these in a carousel/slider UI, allowing users to navigate between segments.
5. **Format and return results:**
    - Return a dictionary with keys: `visualizations`, `insights`, `reply`, and `context` (see below).

### 3. Optional: Segmentation Support
If your analysis supports segmentation, implement automatic variable identification:

```python
def handle_segmentation(self, params, metadata):
    """Handle segmentation requests automatically."""
    if "by" in params.get("question", "").lower():
        # Extract segmentation variable from question
        segmentation_var = self.find_segmentation_variable(metadata, params['chat_model'], "age")
        
        # Get value labels for segmentation groups
        value_labels = metadata.get('value_labels', {}).get(segmentation_var, {})
        
        # Create segments
        segments = {}
        for value, label in value_labels.items():
            segment_data = data[data[segmentation_var] == value]
            if len(segment_data) > 0:
                segments[label] = segment_data
        
        return segments
    return {"Overall": data}
```

---

## Example: Enhanced MCP with Segmentation
```python
from app.services.mcp_base import MCPBase
import pandas as pd
import json

class ExampleMCP(MCPBase):
    def __init__(self):
        super().__init__()
        self.name = "example"
        self.required_columns = ["x", "y"]
        self.description = "Example analysis with segmentation support."

    def find_variable_mapping(self, metadata, chat_model):
        """Use LLM to find variable mapping."""
        column_labels = metadata.get('column_labels', {})
        
        prompt = f"""
        Given these questions, which columns should be used for the analysis?
        We need: x, y
        
        Available questions:
        {json.dumps(column_labels, indent=2)}
        
        Reply with JSON: {{"x": "column_name", "y": "column_name"}}
        """
        
        response = chat_model.generate_reply(prompt)
        return json.loads(response)

    def run(self, data, params=None):
        metadata = params.get("metadata", {})
        
        # 1. Propose mapping if not confirmed
        if not params.get("column_map") or not params.get("column_map_confirmed", False):
            proposed_map = self.find_variable_mapping(metadata, params['chat_model'])
            return {
                "reply": f"Before running the analysis, please confirm the variable mapping:\n"
                         f"x: {proposed_map['x']}\ny: {proposed_map['y']}\n"
                         f"Reply 'yes' to confirm or provide a new mapping.",
                "context": {
                    "analysis_type": self.name,
                    "proposed_column_map": proposed_map,
                    "variables_used": self.required_columns,
                    "column_map_confirmed": False
                }
            }
        
        # 2. Handle segmentation
        segments = self.handle_segmentation(params, metadata)
        
        # 3. Run analysis for each segment
        results = {}
        charts = []
        tables = []
        for segment_name, segment_data in segments.items():
            col_map = params["column_map"]
            # ... your analysis logic here ...
            # For each segment, generate a chart and/or table
            chart = {
                "type": "curve",
                "title": f"My Chart - {segment_name}",
                "plot_data": "<base64-png>"
            }
            table = {
                "type": "summary",
                "title": f"Summary Table - {segment_name}",
                "data": [{"metric": "A", "value": 123}]
            }
            charts.append(chart)
            tables.append(table)
            results[segment_name] = "..."
        
        return {
            "visualizations": {"charts": charts, "tables": tables},
            "insights": "Key findings and recommendations...",
            "reply": "Business-focused summary for the user...",
            "context": {
                "analysis_type": self.name, 
                "variables_used": self.required_columns,
                "segments": list(segments.keys())
            }
        }
```

---

## Required Output Format
Your `run()` method must always return a dictionary with these keys:
- `reply`: A chat message for the user (summary, next steps, or mapping prompt).
- `visualizations`: Dict with `charts` and/or `tables` (see below).
    - **For segmented analysis:**
        - Return one chart/table per segment (e.g., one chart for each age group).
        - The frontend will display these in a carousel/slider UI, allowing users to navigate between segments.
- `insights`: Narrative insights for the user.
- `context`: Dict with at least `analysis_type`, `variables_used`, and mapping status.

### Example Output
```python
{
    "reply": "Here are your results...",
    "visualizations": {
        "charts": [
            {"type": "curve", "title": "My Chart - 18-24", "plot_data": "<base64-png>"},
            {"type": "curve", "title": "My Chart - 25-34", "plot_data": "<base64-png>"},
            ...
        ],
        "tables": [
            {"type": "summary", "title": "Summary Table - 18-24", "data": [{"metric": "A", "value": 123}]},
            {"type": "summary", "title": "Summary Table - 25-34", "data": [{"metric": "A", "value": 456}]},
            ...
        ]
    },
    "insights": "Key findings and recommendations...",
    "context": {
        "analysis_type": "example",
        "variables_used": ["x", "y"],
        "column_map_confirmed": True,
        "segments": ["18-24", "25-34", "35+"]
    }
}
```

---

## Step-by-Step Checklist
- [ ] Inherit from `MCPBase`.
- [ ] Set `self.name`, `self.required_columns`, and `self.description`.
- [ ] Implement `run()` as described above.
- [ ] Always propose and confirm variable mapping before running analysis.
- [ ] Use LLM to match variables with available metadata.
- [ ] Implement segmentation support if applicable.
- [ ] Return results in the required format.
- [ ] Test your MCP by simulating the chat flow (propose mapping, confirm, run analysis).

---

## Best Practices
- **Be explicit about required variables** and check for their presence in the mapping and data.
- **Never run analysis without user confirmation** of the mapping.
- **Use LLM for variable matching** to handle different question formats and naming conventions.
- **Support segmentation** when it makes sense for your analysis.
- **Return clear, actionable replies** for the chat agent (use LLM to polish if needed).
- **Log debug info** for easier troubleshooting.
- **Handle metadata gracefully** - check if fields exist before using them.
- **For segmented analysis, return one chart/table per segment for carousel display in the frontend.**

---

## FAQ
**Q: Do I need to know anything about the rest of the system?**
A: No. If you follow this interface and conversational flow, your MCP will work out of the box.

**Q: How do I use an LLM to propose a mapping?**
A: You will receive a `chat_model` in `params`—call `chat_model.generate_reply(prompt)` with your prompt.

**Q: How do I generate a chart?**
A: Use matplotlib or your preferred library, save to a PNG in memory, and base64-encode it for the `plot_data` field.

**Q: How do I handle segmentation?**
A: Use the metadata to identify segmentation variables and run your analysis separately for each segment.

**Q: What metadata is available?**
A: Column names, column labels (questions), value labels, variable labels, measurement levels, and more.

---

## See Also
- Example: `van_westendorp.py` for a full-featured MCP with segmentation
- Main architecture: `ARCHITECTURE.md`
- API/Chat flow: `README.md` 