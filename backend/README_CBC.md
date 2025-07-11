# Choice-Based Conjoint (CBC) MCP

## Overview

The Choice-Based Conjoint MCP provides advanced conjoint analysis capabilities using Hierarchical Bayes (HB) methodology. It automatically identifies conjoint data structure, runs HB analysis using PyTorch, and generates comprehensive visualizations and insights.

## Features

- **Automatic Data Structure Identification**: Intelligently identifies choice, task, alternative, and attribute columns
- **Hierarchical Bayes Analysis**: Uses FastHB_CBC implementation with PyTorch for efficient computation
- **LLM-Powered Variable Mapping**: Uses DeepSeek to propose optimal column mappings
- **Segmentation Support**: Run analysis by demographic or behavioral segments
- **Comprehensive Visualizations**: 
  - Attribute importance plots
  - Utility analysis charts
  - MCMC convergence plots
  - Parameter correlation matrices
- **Interactive Results**: Returns charts and tables for frontend display

## Requirements

```bash
torch>=2.1.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
arviz>=0.13.0
scikit-learn>=1.2.0
seaborn>=0.12.0
```

## Data Format

The MCP expects conjoint data with the following structure:

| Column | Description | Example |
|--------|-------------|---------|
| choice | Binary indicator (0/1) | 1 if selected, 0 otherwise |
| task_id | Task/question identifier | 1, 2, 3, ... |
| alternative_id | Alternative/profile identifier | 1, 2, 3, ... |
| attribute1 | First attribute | Brand A, Brand B, Brand C |
| attribute2 | Second attribute | $10, $15, $20 |
| ... | Additional attributes | ... |

## Usage

### Basic Usage
```
"Run choice-based conjoint analysis"
```

### With Segmentation
```
"Run conjoint analysis by age group"
"Analyze choice data by gender"
```

### Variable Mapping
The MCP will automatically propose column mappings and ask for confirmation:
```
Please confirm the variable mapping for CBC analysis:
Choice: choice_column
Task ID: task_id_column  
Alternative ID: alternative_id_column
Attributes: brand, price, color, size

Reply 'yes' to confirm or provide a new mapping.
```

## Output

### Visualizations
- **Attribute Importance Plot**: Shows relative importance of each attribute
- **Utility Analysis**: Displays utility values for attribute levels
- **MCMC Convergence**: Tracks parameter convergence during sampling
- **Correlation Matrix**: Shows parameter correlations

### Tables
- **Analysis Summary**: Key metrics and sample information
- **Attribute Rankings**: Ordered list of attribute importance

### Insights
- Attribute importance rankings
- Key findings and recommendations
- Sample size and analysis parameters

## Technical Details

### HB Analysis Parameters
- **MCMC Samples**: 1000 (configurable)
- **Burn-in**: 500 iterations (configurable)
- **Device**: Automatic GPU/CPU detection
- **Convergence**: Monitored and reported

### Data Processing
- **One-hot Encoding**: Automatic for categorical attributes
- **Missing Data**: Handled gracefully with warnings
- **Data Validation**: Checks for required structure

## Example Response

```json
{
  "reply": "I've completed the Choice-Based Conjoint analysis. The analysis shows attribute importance and utility values for each attribute level.",
  "visualizations": {
    "charts": [
      {
        "type": "cbc_analysis",
        "title": "Choice-Based Conjoint Analysis Results",
        "plot_data": "<base64-png>",
        "data": {
          "attribute_importance": [
            {"attribute": "price", "importance": 0.85},
            {"attribute": "brand", "importance": 0.72}
          ]
        }
      }
    ],
    "tables": [
      {
        "type": "cbc_summary", 
        "title": "Analysis Summary",
        "data": [
          {"metric": "Number of Respondents", "value": "50"},
          {"metric": "Number of Tasks", "value": "8"},
          {"metric": "Top Attribute", "value": "price"}
        ]
      }
    ]
  },
  "insights": "Choice-Based Conjoint Analysis Results:\n\nAttribute Importance Ranking:\n1. price: 0.850\n2. brand: 0.720\n\nThe most important attribute is 'price'."
}
```

## Troubleshooting

### Common Issues
1. **Data Structure**: Ensure data has choice, task_id, and alternative_id columns
2. **Memory**: Large datasets may require GPU or reduced MCMC samples
3. **Convergence**: Check MCMC convergence plots for parameter stability

### Error Messages
- "No conjoint structure found": Check column names and data format
- "Insufficient data": Ensure adequate sample size per task
- "MCMC failed to converge": Try increasing burn-in or samples

## Performance

- **Small datasets** (<1000 observations): ~30 seconds
- **Medium datasets** (1000-10000 observations): ~2-5 minutes  
- **Large datasets** (>10000 observations): ~10-30 minutes

GPU acceleration provides 2-5x speedup for large datasets. 