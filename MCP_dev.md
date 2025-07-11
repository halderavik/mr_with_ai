# How to Build a Plug-and-Play MCP (Market Research Control Protocol)

## What is an MCP?
An MCP (Market Research Control Protocol) is a Python class that implements a specific type of market research analysis (e.g., price sensitivity, segmentation, etc.). MCPs are plug-and-play modules: if you follow the interface and conversational flow described here, your MCP will work with our platform—no internal system knowledge required.

---

## Quickstart: Building a New MCP

**Every MCP must:**
- Inherit from `MCPBase` (see `backend/app/services/mcp_base.py`).
- Set a unique `self.name`, `self.required_columns`, and `self.description` in `__init__`.
- Implement the `run(self, data, params)` method.
- Use the provided LLM (`chat_model`) for variable mapping and user chat.
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
1. **Propose a variable mapping** if none is confirmed:
    - Use LLM or heuristics to suggest which columns to use for each required variable.
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

## Example: Minimal MCP Template

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

        # 1. Propose mapping if not confirmed
        if not params.get("column_map") or not params.get("column_map_confirmed", False):
            # Use LLM to propose mapping
            column_labels = metadata.get('column_labels', {})
            prompt = f"""
            Given these questions, which columns should be used for the analysis?
            We need: x, y
            Available questions: {json.dumps(column_labels, indent=2)}
            Reply with JSON: {{"x": "column_name", "y": "column_name"}}
            """
            response = chat_model.generate_reply(prompt)
            proposed_map = json.loads(response)
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
- Use the LLM to find the correct segmentation variable from metadata.
- Run your analysis for each segment and return a chart/table per segment.

**Example:**
```python
def handle_segmentation(self, data, params, metadata):
    question = params.get("question", "")
    if "by" in question.lower():
        # Use LLM to find the segmentation variable
        prompt = f"""
        Given these questions, which column best matches the segmentation request?
        Segmentation requested: {question}
        Available questions: {json.dumps(metadata.get('column_labels', {}), indent=2)}
        Reply with the EXACT column name.
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
- **Be explicit about required variables** and check for their presence in the mapping and data.
- **Never run analysis without user confirmation** of the mapping.
- **Use LLM for variable matching** to handle different question formats and naming conventions.
- **Support segmentation** when it makes sense for your analysis.
- **Return clear, actionable replies** for the chat agent (use LLM to polish if needed).
- **Log debug info** for easier troubleshooting.
- **Handle metadata gracefully** - check if fields exist before using them.
- **For segmented analysis, return one chart/table per segment for carousel display in the frontend.**

---

## Advanced: See a Full Example
For a full-featured MCP with segmentation, see `backend/app/mcp/van_westendorp.py`.

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