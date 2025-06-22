# Van Westendorp Price Sensitivity Analysis - Complete Formulas & Calculations

## Data Structure Requirements

Your dataset should contain four price variables for each respondent:
```
respondent_id | too_cheap | bargain | getting_expensive | too_expensive
1            | 10        | 15      | 25               | 35
2            | 8         | 12      | 20               | 30
...          | ...       | ...     | ...              | ...
n            | x         | y       | z                | w
```

## Step 1: Data Validation Formula

For each respondent, verify logical consistency:
```
too_cheap ≤ bargain ≤ getting_expensive ≤ too_expensive
```

**Validation Check:**
```
valid_response = (too_cheap[i] ≤ bargain[i]) AND 
                 (bargain[i] ≤ getting_expensive[i]) AND 
                 (getting_expensive[i] ≤ too_expensive[i])
```

## Step 2: Price Range Determination

**Minimum Price:**
```
P_min = min(too_cheap₁, too_cheap₂, ..., too_cheap_n, 
            bargain₁, bargain₂, ..., bargain_n,
            getting_expensive₁, getting_expensive₂, ..., getting_expensive_n,
            too_expensive₁, too_expensive₂, ..., too_expensive_n)
```

**Maximum Price:**
```
P_max = max(too_cheap₁, too_cheap₂, ..., too_cheap_n, 
            bargain₁, bargain₂, ..., bargain_n,
            getting_expensive₁, getting_expensive₂, ..., getting_expensive_n,
            too_expensive₁, too_expensive₂, ..., too_expensive_n)
```

**Extended Range (with 10% buffer):**
```
P_range_min = P_min × 0.9
P_range_max = P_max × 1.1
```

## Step 3: Cumulative Distribution Functions

For any price point P in the range [P_range_min, P_range_max]:

### Too Cheap Curve (TC)
```
TC(P) = (Number of respondents where too_cheap ≤ P) / N × 100

TC(P) = (∑[i=1 to N] I(too_cheap_i ≤ P)) / N × 100

where I(condition) = 1 if condition is true, 0 otherwise
```

### Bargain Curve (B)
```
B(P) = (Number of respondents where bargain ≤ P) / N × 100

B(P) = (∑[i=1 to N] I(bargain_i ≤ P)) / N × 100
```

### Getting Expensive Curve (GE)
```
GE(P) = (Number of respondents where getting_expensive ≥ P) / N × 100

GE(P) = (∑[i=1 to N] I(getting_expensive_i ≥ P)) / N × 100
```

### Too Expensive Curve (TE)
```
TE(P) = (Number of respondents where too_expensive ≥ P) / N × 100

TE(P) = (∑[i=1 to N] I(too_expensive_i ≥ P)) / N × 100
```

## Step 4: Intersection Point Calculations

### Point of Marginal Cheapness (PMC)
Find price P where TC(P) = TE(P)

**Numerical Solution:**
```
PMC = argmin|TC(P) - TE(P)|
```

**Linear Interpolation Method:**
```
If TC(P₁) < TE(P₁) and TC(P₂) > TE(P₂), then:

PMC = P₁ + (P₂ - P₁) × (TE(P₁) - TC(P₁)) / ((TE(P₁) - TC(P₁)) + (TC(P₂) - TE(P₂)))
```

### Point of Marginal Expensiveness (PME)
Find price P where B(P) = GE(P)

**Numerical Solution:**
```
PME = argmin|B(P) - GE(P)|
```

**Linear Interpolation Method:**
```
If B(P₁) > GE(P₁) and B(P₂) < GE(P₂), then:

PME = P₁ + (P₂ - P₁) × (B(P₁) - GE(P₁)) / ((B(P₁) - GE(P₁)) + (GE(P₂) - B(P₂)))
```

### Optimal Price Point (OPP)
Find price P where TC(P) = GE(P)

**Numerical Solution:**
```
OPP = argmin|TC(P) - GE(P)|
```

**Linear Interpolation Method:**
```
If TC(P₁) < GE(P₁) and TC(P₂) > GE(P₂), then:

OPP = P₁ + (P₂ - P₁) × (GE(P₁) - TC(P₁)) / ((GE(P₁) - TC(P₁)) + (TC(P₂) - GE(P₂)))
```

### Indifference Price Point (IPP)
Find price P where B(P) = TE(P)

**Numerical Solution:**
```
IPP = argmin|B(P) - TE(P)|
```

**Linear Interpolation Method:**
```
If B(P₁) > TE(P₁) and B(P₂) < TE(P₂), then:

IPP = P₁ + (P₂ - P₁) × (B(P₁) - TE(P₁)) / ((B(P₁) - TE(P₁)) + (TE(P₂) - B(P₂)))
```

## Step 5: Acceptable Price Range Calculation

**Range of Acceptable Prices:**
```
Acceptable_Range = [PMC, PME]
Range_Width = PME - PMC
```

**Stress Range (unacceptable prices):**
```
Stress_Range_Lower = [P_range_min, PMC]
Stress_Range_Upper = [PME, P_range_max]
```

## Step 6: Market Penetration Calculations

At any price P, calculate market penetration:

**Not Too Cheap:**
```
Not_Too_Cheap(P) = 100 - TC(P)
```

**Still a Bargain:**
```
Still_Bargain(P) = 100 - B(P)
```

**Not Yet Expensive:**
```
Not_Yet_Expensive(P) = 100 - GE(P)
```

**Not Too Expensive:**
```
Not_Too_Expensive(P) = 100 - TE(P)
```

**Overall Acceptance:**
```
Acceptance(P) = min(Not_Too_Cheap(P), Not_Too_Expensive(P))
```

## Step 7: Statistical Measures

### Mean Prices
```
Mean_Too_Cheap = (∑[i=1 to N] too_cheap_i) / N
Mean_Bargain = (∑[i=1 to N] bargain_i) / N
Mean_Getting_Expensive = (∑[i=1 to N] getting_expensive_i) / N
Mean_Too_Expensive = (∑[i=1 to N] too_expensive_i) / N
```

### Standard Deviations
```
SD_Too_Cheap = √[(∑[i=1 to N] (too_cheap_i - Mean_Too_Cheap)²) / (N-1)]
SD_Bargain = √[(∑[i=1 to N] (bargain_i - Mean_Bargain)²) / (N-1)]
SD_Getting_Expensive = √[(∑[i=1 to N] (getting_expensive_i - Mean_Getting_Expensive)²) / (N-1)]
SD_Too_Expensive = √[(∑[i=1 to N] (too_expensive_i - Mean_Too_Expensive)²) / (N-1)]
```

### Confidence Intervals (95%)
```
CI_Lower = Mean - (1.96 × SD / √N)
CI_Upper = Mean + (1.96 × SD / √N)
```

## Step 8: Price Sensitivity Index

**Newton-Miller-Smith Price Sensitivity Index:**
```
PSI = (Mean_Too_Expensive - Mean_Too_Cheap) / Mean_Too_Cheap × 100
```

**Interpretation:**
- PSI < 50%: Low price sensitivity
- 50% ≤ PSI < 100%: Moderate price sensitivity  
- PSI ≥ 100%: High price sensitivity

## Step 9: Sample Calculation Example

Given sample data:
```
N = 100 respondents
P = $20 (test price)
```

**Count at P = $20:**
- too_cheap ≤ $20: 75 respondents
- bargain ≤ $20: 45 respondents  
- getting_expensive ≥ $20: 60 respondents
- too_expensive ≥ $20: 25 respondents

**Calculations:**
```
TC(20) = 75/100 × 100 = 75%
B(20) = 45/100 × 100 = 45%
GE(20) = 60/100 × 100 = 60%
TE(20) = 25/100 × 100 = 25%
```

**Market Acceptance at $20:**
```
Acceptance(20) = min(100-75, 100-25) = min(25, 75) = 25%
```

## Step 10: Implementation Notes

**Discrete Price Points:**
Create price intervals (e.g., $0.50 or $1.00 increments) and calculate curves for each point.

**Smoothing:**
Apply moving averages or spline interpolation for smoother curves:
```
Smoothed_Value(P) = (Value(P-1) + Value(P) + Value(P+1)) / 3
```

**Weighted Analysis:**
If using demographic weights:
```
Weighted_TC(P) = (∑[i=1 to N] weight_i × I(too_cheap_i ≤ P)) / (∑[i=1 to N] weight_i) × 100
```

This comprehensive formula set provides all the mathematical foundations needed to implement Van Westendorp Price Sensitivity Analysis from raw survey data to final price recommendations.


# Van Westendorp Price Sensitivity Analysis - Complete Graph Plotting Guide

## Graph Setup and Configuration

### Basic Plot Structure
```
Chart Type: Line Chart
X-Axis: Price Range [P_range_min, P_range_max]
Y-Axis: Percentage of Respondents [0%, 100%]
Title: "Van Westendorp Price Sensitivity Analysis"
```

### Axis Configuration
**X-Axis (Price):**
```
Scale: Linear
Range: [P_range_min × 0.95, P_range_max × 1.05]
Tick Interval: (P_range_max - P_range_min) / 10
Format: Currency ($X.XX)
Label: "Price ($)"
```

**Y-Axis (Percentage):**
```
Scale: Linear
Range: [0, 100]
Tick Interval: 10
Format: Percentage (XX%)
Label: "Percentage of Respondents (%)"
Grid Lines: Major every 20%, Minor every 10%
```

## Step 1: Data Point Generation

### Create Price Array
```
Price_Step = (P_range_max - P_range_min) / 200  // 200 data points
Price_Array = [P_range_min + i × Price_Step for i in 0 to 200]
```

### Calculate Curve Values
For each price P in Price_Array:
```
TC_Values[i] = TC(Price_Array[i])
B_Values[i] = B(Price_Array[i])
GE_Values[i] = GE(Price_Array[i])
TE_Values[i] = TE(Price_Array[i])
```

## Step 2: Line Series Configuration

### Too Cheap Curve
```
Series Name: "Too Cheap"
Data Points: (Price_Array, TC_Values)
Line Style: Solid
Line Width: 2-3 pixels
Color: Red (#FF0000) or Dark Red (#CC0000)
Marker: None (smooth line)
Legend: "Too Cheap"
```

### Bargain Curve
```
Series Name: "Bargain"
Data Points: (Price_Array, B_Values)
Line Style: Solid
Line Width: 2-3 pixels
Color: Green (#00AA00) or Forest Green (#228B22)
Marker: None (smooth line)
Legend: "Bargain"
```

### Getting Expensive Curve
```
Series Name: "Getting Expensive"
Data Points: (Price_Array, GE_Values)
Line Style: Solid
Line Width: 2-3 pixels
Color: Orange (#FF8C00) or Gold (#FFD700)
Marker: None (smooth line)
Legend: "Getting Expensive"
```

### Too Expensive Curve
```
Series Name: "Too Expensive"
Data Points: (Price_Array, TE_Values)
Line Style: Solid
Line Width: 2-3 pixels
Color: Dark Red (#8B0000) or Crimson (#DC143C)
Marker: None (smooth line)
Legend: "Too Expensive"
```

## Step 3: Intersection Point Markers

### Point of Marginal Cheapness (PMC)
```
Marker Type: Circle
Size: 8-10 pixels
Color: Blue (#0066CC)
Border: 2 pixels, White
Position: (PMC, TC(PMC))
Label: "PMC ($X.XX)"
Label Position: Above point, offset +10 pixels
Label Font: Bold, 10pt
```

### Point of Marginal Expensiveness (PME)
```
Marker Type: Circle  
Size: 8-10 pixels
Color: Purple (#9966CC)
Border: 2 pixels, White
Position: (PME, B(PME))
Label: "PME ($X.XX)"
Label Position: Above point, offset +10 pixels
Label Font: Bold, 10pt
```

### Optimal Price Point (OPP)
```
Marker Type: Diamond
Size: 10-12 pixels
Color: Black (#000000)
Border: 2 pixels, White
Position: (OPP, TC(OPP))
Label: "OPP ($X.XX)"
Label Position: Above point, offset +15 pixels
Label Font: Bold, 12pt
```

### Indifference Price Point (IPP)
```
Marker Type: Square
Size: 8-10 pixels
Color: Gray (#666666)
Border: 2 pixels, White
Position: (IPP, B(IPP))
Label: "IPP ($X.XX)"
Label Position: Below point, offset -15 pixels
Label Font: Bold, 10pt
```

## Step 4: Range Highlighting

### Acceptable Price Range
```
Fill Type: Vertical Band
X-Range: [PMC, PME]
Y-Range: [0, 100]
Fill Color: Light Green (#90EE90) with 30% transparency
Border: None
Label: "Acceptable Range"
Label Position: Center of band, Y = 50%
Label Font: Bold, 11pt, Dark Green
```

### Stress Ranges
**Lower Stress Range:**
```
Fill Type: Vertical Band
X-Range: [P_range_min, PMC]
Y-Range: [0, 100]
Fill Color: Light Red (#FFB6C1) with 20% transparency
Border: None
```

**Upper Stress Range:**
```
Fill Type: Vertical Band
X-Range: [PME, P_range_max]
Y-Range: [0, 100]
Fill Color: Light Red (#FFB6C1) with 20% transparency
Border: None
```

## Step 5: Reference Lines and Annotations

### Horizontal Reference Lines
```
50% Line:
  Position: Y = 50
  Style: Dashed
  Color: Gray (#CCCCCC)
  Width: 1 pixel
  Label: "50%" (at right margin)
```

### Vertical Reference Lines (Optional)
```
Mean Price Lines:
  Positions: X = Mean_Too_Cheap, Mean_Bargain, Mean_Getting_Expensive, Mean_Too_Expensive
  Style: Dotted
  Color: Light Gray (#DDDDDD)
  Width: 1 pixel
  Labels: "Avg TC", "Avg B", "Avg GE", "Avg TE"
```

## Step 6: Legend Configuration

### Legend Properties
```
Position: Top Right or Bottom Right
Background: White with light border
Font Size: 10pt
Spacing: 5 pixels between items
Order: 
  1. Too Cheap (Red line)
  2. Bargain (Green line)
  3. Getting Expensive (Orange line)
  4. Too Expensive (Dark Red line)
  5. Key Points (symbols)
```

### Legend Items Format
```
Line Items: [Color Line] + [Text Label]
Point Items: [Marker Symbol] + [Point Name]
Range Items: [Color Rectangle] + [Range Name]
```

## Step 7: Title and Subtitle Configuration

### Main Title
```
Text: "Van Westendorp Price Sensitivity Analysis"
Font: Bold, 16pt
Position: Top Center
Color: Black (#000000)
Margin: 20 pixels from top
```

### Subtitle (Optional)
```
Text: "Product: [Product Name] | Sample Size: N=[XXX] | Date: [Date]"
Font: Regular, 12pt
Position: Below main title
Color: Dark Gray (#555555)
Margin: 10 pixels below main title
```

## Step 8: Annotations and Callouts

### Key Insights Box
```
Position: Bottom Left or Top Left (avoid line intersections)
Background: Light Blue (#E6F3FF) with border
Border: 1 pixel, Blue (#0066CC)
Padding: 10 pixels
Font: 9pt, Regular
Content:
  "Acceptable Range: $[PMC] - $[PME]"
  "Range Width: $[PME-PMC] ([Percentage]%)"
  "Optimal Price: $[OPP]"
  "Sample Size: [N] respondents"
```

### Directional Arrows (for curve interpretation)
```
Too Cheap Arrow:
  Start: (P_range_min + 10%, 20%)
  End: Point to steep part of TC curve
  Style: Blue arrow
  Label: "Quality concerns increase"

Too Expensive Arrow:
  Start: (P_range_max - 10%, 80%)
  End: Point to steep part of TE curve
  Style: Red arrow
  Label: "Purchase resistance increases"
```

## Step 9: Export and Display Settings

### Image Export Configuration
```
Format: PNG or SVG (for scalability)
Resolution: 300 DPI minimum
Dimensions: 1200 × 800 pixels (standard)
Background: White
Anti-aliasing: Enabled
```

### Interactive Plot Settings (if applicable)
```

```


