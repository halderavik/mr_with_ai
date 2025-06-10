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
- `params`: a dict with keys like `column_map`, `column_map_confirmed`, and `chat_model` (for LLM use).

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
5. **Format and return results:**
    - Return a dictionary with keys: `visualizations`, `insights`, `reply`, and `context` (see below).

---

## Example: Minimal MCP
```python
from app.services.mcp_base import MCPBase
import pandas as pd

class ExampleMCP(MCPBase):
    def __init__(self):
        super().__init__()
        self.name = "example"
        self.required_columns = ["x", "y"]
        self.description = "Example analysis."

    def run(self, data, params=None):
        # 1. Propose mapping if not confirmed
        if not params.get("column_map") or not params.get("column_map_confirmed", False):
            proposed_map = {"x": "Q1", "y": "Q2"}
            return {
                "reply": "Before running the analysis, please confirm the variable mapping:\n"
                         "x: Q1\ny: Q2\nReply 'yes' to confirm or provide a new mapping.",
                "context": {
                    "analysis_type": self.name,
                    "proposed_column_map": proposed_map,
                    "variables_used": self.required_columns,
                    "column_map_confirmed": False
                }
            }
        # 2. Run analysis
        col_map = params["column_map"]
        # ... your analysis logic here ...
        return {
            "visualizations": {"charts": [], "tables": []},
            "insights": "Key findings and recommendations...",
            "reply": "Business-focused summary for the user...",
            "context": {"analysis_type": self.name, "variables_used": self.required_columns}
        }
```

---

## Required Output Format
Your `run()` method must always return a dictionary with these keys:
- `reply`: A chat message for the user (summary, next steps, or mapping prompt).
- `visualizations`: Dict with `charts` and/or `tables` (see below).
- `insights`: Narrative insights for the user.
- `context`: Dict with at least `analysis_type`, `variables_used`, and mapping status.

### Example Output
```python
{
    "reply": "Here are your results...",
    "visualizations": {
        "charts": [
            {"type": "curve", "title": "My Chart", "plot_data": "<base64-png>"}
        ],
        "tables": [
            {"type": "summary", "title": "Summary Table", "data": [{"metric": "A", "value": 123}]}
        ]
    },
    "insights": "Key findings and recommendations...",
    "context": {
        "analysis_type": "example",
        "variables_used": ["x", "y"],
        "column_map_confirmed": True
    }
}
```

---

## Step-by-Step Checklist
- [ ] Inherit from `MCPBase`.
- [ ] Set `self.name`, `self.required_columns`, and `self.description`.
- [ ] Implement `run()` as described above.
- [ ] Always propose and confirm variable mapping before running analysis.
- [ ] Return results in the required format.
- [ ] Test your MCP by simulating the chat flow (propose mapping, confirm, run analysis).

---

## Best Practices
- **Be explicit about required variables** and check for their presence in the mapping and data.
- **Never run analysis without user confirmation** of the mapping.
- **Return clear, actionable replies** for the chat agent (use LLM to polish if needed).
- **Log debug info** for easier troubleshooting.

---

## FAQ
**Q: Do I need to know anything about the rest of the system?**
A: No. If you follow this interface and conversational flow, your MCP will work out of the box.

**Q: How do I use an LLM to propose a mapping?**
A: You will receive a `chat_model` in `params`—call `chat_model.generate_reply(prompt)` with your prompt.

**Q: How do I generate a chart?**
A: Use matplotlib or your preferred library, save to a PNG in memory, and base64-encode it for the `plot_data` field.

---

## See Also
- Example: `van_westendorp.py` for a full-featured MCP
- Main architecture: `ARCHITECTURE.md`
- API/Chat flow: `README.md` 