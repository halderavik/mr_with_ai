# backend/app/mcp/van_westendorp.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from app.services.mcp_base import MCPBase
from app.services.plotting import fig_to_base64
import matplotlib.pyplot as plt
from app.utils.common import filter_dataframe
import json
import io
import base64


class VanWestendorpMCP(MCPBase):
    """
    Van Westendorp's Price Sensitivity Meter (PSM) implementation.
    Analyzes price sensitivity using four key price points.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "van_westendorp"
        self.description = "Analyzes price sensitivity using Van Westendorp's Price Sensitivity Meter methodology"
        self.required_columns = ["too_cheap", "too_expensive", "bargain", "getting_expensive"]
        self.price_column = "price"
        self.analysis_type = "price_sensitivity"
        
    def _generate_plot(self, price_grid: np.ndarray, too_cheap: np.ndarray, 
                      too_expensive: np.ndarray, bargain: np.ndarray, 
                      getting_expensive: np.ndarray, pmc: float, pme: float, 
                      opp: float) -> str:
        """Generate the Van Westendorp plot and return as base64 encoded PNG."""
        plt.figure(figsize=(10, 6))
        
        # Plot the curves
        plt.plot(price_grid, too_cheap, 'r-', label='Too Cheap')
        plt.plot(price_grid, too_expensive, 'b-', label='Too Expensive')
        plt.plot(price_grid, bargain, 'g-', label='Bargain')
        plt.plot(price_grid, getting_expensive, 'y-', label='Getting Expensive')
        
        # Add vertical lines for key price points
        plt.axvline(x=pmc, color='r', linestyle='--', label=f'PMC: ${pmc:.2f}')
        plt.axvline(x=pme, color='b', linestyle='--', label=f'PME: ${pme:.2f}')
        plt.axvline(x=opp, color='g', linestyle='--', label=f'OPP: ${opp:.2f}')
        
        # Customize the plot
        plt.title('Van Westendorp Price Sensitivity Curves')
        plt.xlabel('Price ($)')
        plt.ylabel('Cumulative Percentage')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def handle_followup_question(self, question: str, analysis_result: Dict[str, Any], data: pd.DataFrame, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle follow-up questions about the Van Westendorp analysis.
        
        Args:
            question (str): The follow-up question from the user
            analysis_result (Dict[str, Any]): The previous analysis results
            data (pd.DataFrame): The original dataset
            conversation_context (Dict[str, Any]): Previous conversation context
            
        Returns:
            Dict[str, Any]: Answer and updated context
        """
        # Extract key metrics from analysis result
        price_points = analysis_result.get("visualizations", {}).get("tables", [{}])[0].get("data", [])
        metrics = {item["metric"]: item["value"] for item in price_points}
        
        # Initialize context if not provided
        if conversation_context is None:
            conversation_context = {
                "analysis_type": self.name,
                "variables_used": ["too_cheap", "bargain", "getting_expensive", "too_expensive"],
                "last_question": None,
                "last_answer": None,
                "message_history": []
            }
        
        # Add current question to message history
        conversation_context["message_history"] = conversation_context.get("message_history", [])[-4:] + [question]
        
        # Check for repeated questions
        if question.lower() == conversation_context.get("last_question", "").lower():
            return {
                "answer": "I've already answered this question. Would you like me to explain it differently?",
                "context": conversation_context
            }
        
        # Common question patterns and their answers
        question_patterns = {
            "variable": {
                "patterns": ["what variables", "which variables", "what columns", "which columns"],
                "answer": (
                    "The Van Westendorp analysis uses four key variables:\n"
                    "1. Too Cheap: Respondents' price point where the product is considered too cheap\n"
                    "2. Bargain: Respondents' price point where the product represents good value\n"
                    "3. Getting Expensive: Respondents' price point where the product starts to become expensive\n"
                    "4. Too Expensive: Respondents' price point where the product is considered too expensive"
                )
            },
            "interpretation": {
                "patterns": ["what does this mean", "how to interpret", "explain the results"],
                "answer": (
                    f"The analysis shows:\n"
                    f"• Point of Marginal Cheapness (PMC): {metrics.get('Point of Marginal Cheapness (PMC)', 'N/A')} - Below this price, too many customers consider it too cheap\n"
                    f"• Point of Marginal Expensiveness (PME): {metrics.get('Point of Marginal Expensiveness (PME)', 'N/A')} - Above this price, too many customers consider it too expensive\n"
                    f"• Optimal Price Point (OPP): {metrics.get('Optimal Price Point (OPP)', 'N/A')} - The recommended price point where acceptable equals expensive\n"
                    f"• Price Sensitivity: {metrics.get('Price Sensitivity', 'N/A')} - Indicates how sensitive customers are to price changes"
                )
            },
            "methodology": {
                "patterns": ["how does it work", "methodology", "how is it calculated"],
                "answer": (
                    "The Van Westendorp methodology works by:\n"
                    "1. Asking respondents four price-related questions\n"
                    "2. Creating cumulative distribution curves for each question\n"
                    "3. Finding key intersection points:\n"
                    "   - PMC: Where 'too cheap' equals 'bargain'\n"
                    "   - PME: Where 'too expensive' equals 'getting expensive'\n"
                    "   - OPP: Where 'acceptable' equals 'expensive'\n"
                    "4. Calculating price sensitivity as (PME - PMC) / PMC * 100"
                )
            },
            "recommendation": {
                "patterns": ["what price", "recommend", "suggest"],
                "answer": (
                    f"Based on the analysis, I recommend:\n"
                    f"• Optimal price point: {metrics.get('Optimal Price Point (OPP)', 'N/A')}\n"
                    f"• Acceptable price range: Between {metrics.get('Point of Marginal Cheapness (PMC)', 'N/A')} and {metrics.get('Point of Marginal Expensiveness (PME)', 'N/A')}\n"
                    f"• Price sensitivity of {metrics.get('Price Sensitivity', 'N/A')} indicates {'high' if float(metrics.get('Price Sensitivity', '0').replace('%', '')) > 50 else 'moderate'} sensitivity to price changes"
                )
            }
        }
        
        # Convert question to lowercase for matching
        question_lower = question.lower()
        
        # Check each category of questions
        for category, info in question_patterns.items():
            if any(pattern in question_lower for pattern in info["patterns"]):
                # Update conversation context
                conversation_context.update({
                    "last_question": question,
                    "last_answer": info["answer"]
                })
                return {
                    "answer": info["answer"],
                    "context": conversation_context
                }
        
        # If no specific pattern matches, provide a general response
        general_response = (
            "I can help explain:\n"
            "• The variables used in the analysis\n"
            "• How to interpret the results\n"
            "• The methodology behind the calculations\n"
            "• Price recommendations\n\n"
            "Please ask a specific question about any of these aspects."
        )
        
        # Update conversation context
        conversation_context.update({
            "last_question": question,
            "last_answer": general_response
        })
        
        return {
            "answer": general_response,
            "context": conversation_context
        }

    def generate_visualizations(self, price_grid, tc_cum, te_cum, ba_cum, ge_cum, pmc, pme, opp, price_sensitivity):
        """
        Generate visualizations for the Van Westendorp analysis.
        
        Args:
            price_grid: Array of price points
            tc_cum: Too Cheap cumulative distribution
            te_cum: Too Expensive cumulative distribution
            ba_cum: Bargain cumulative distribution
            ge_cum: Getting Expensive cumulative distribution
            pmc: Point of Marginal Cheapness
            pme: Point of Marginal Expensiveness
            opp: Optimal Price Point
            price_sensitivity: Price sensitivity percentage
            
        Returns:
            Dict[str, Any]: Visualization data
        """
        print("[DEBUG] Starting plot generation...")
        
        # Generate the plot
        plt.figure(figsize=(10, 6))
        
        # Plot the curves
        plt.plot(price_grid, tc_cum, 'r-', label='Too Cheap')
        plt.plot(price_grid, te_cum, 'b-', label='Too Expensive')
        plt.plot(price_grid, ba_cum, 'g-', label='Bargain')
        plt.plot(price_grid, ge_cum, 'y-', label='Getting Expensive')
        
        # Add vertical lines for key price points
        plt.axvline(x=pmc, color='r', linestyle='--', label=f'PMC: ${pmc:.2f}')
        plt.axvline(x=pme, color='b', linestyle='--', label=f'PME: ${pme:.2f}')
        plt.axvline(x=opp, color='g', linestyle='--', label=f'OPP: ${opp:.2f}')
        
        # Customize the plot
        plt.title('Van Westendorp Price Sensitivity Curves')
        plt.xlabel('Price ($)')
        plt.ylabel('Cumulative Percentage')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        print("[DEBUG] Plot created, converting to base64...")
        
        # Save plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        print(f"[DEBUG] Plot data generated, length: {len(plot_data)}")
        print(f"[DEBUG] Plot data preview: {plot_data[:50]}...")
        
        # Generate the chart data
        chart_data = {
            "type": "van_westendorp_curves",
            "title": "Van Westendorp Price Sensitivity Curves",
            "plot_data": plot_data,
            "data": {
                "price_grid": price_grid.tolist(),
                "too_cheap": tc_cum.tolist(),
                "too_expensive": te_cum.tolist(),
                "bargain": ba_cum.tolist(),
                "getting_expensive": ge_cum.tolist()
            },
            "annotations": {
                "pmc": float(pmc),
                "pme": float(pme),
                "opp": float(opp)
            }
        }
        
        print("[DEBUG] Chart data created with plot_data")
        
        # Generate the table data
        table_data = {
            "type": "van_westendorp_summary",
            "title": "Price Point Summary",
            "data": [
                {
                    "metric": "Point of Marginal Cheapness (PMC)",
                    "value": f"${pmc:.2f}"
                },
                {
                    "metric": "Point of Marginal Expensiveness (PME)",
                    "value": f"${pme:.2f}"
                },
                {
                    "metric": "Optimal Price Point (OPP)",
                    "value": f"${opp:.2f}"
                },
                {
                    "metric": "Price Sensitivity",
                    "value": f"{price_sensitivity:.1f}%"
                }
            ]
        }
        
        result = {
            "charts": [chart_data],
            "tables": [table_data]
        }
        
        print("[DEBUG] Final visualization result:", json.dumps(result, indent=2))
        return result

    def _process_user_request(self, question: str, metadata: Dict[str, Any], context: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process user request to extract analysis parameters and generate follow-up questions.
        
        Args:
            question (str): User's question
            metadata (Dict[str, Any]): Dataset metadata
            context (Optional[Dict[str, Any]]): Previous conversation context
            params (Optional[Dict[str, Any]]): Additional parameters including chat_model
            
        Returns:
            Dict containing:
            - analysis_plan: Dict with analysis parameters
            - followup_questions: List of questions to ask user
            - segmentation_groups: Dict mapping segment names to their values
        """
        try:
            print(f"[DEBUG] Processing question: {question}")
            
            # Get all available questions and their details
            questions_info = []
            column_labels = metadata.get('column_labels', {})
            
            for var, question_text in column_labels.items():
                if question_text:  # Only include non-empty questions
                    question_info = {
                        'variable': var,
                        'question': question_text,
                        'variable_label': metadata.get('variable_labels', {}).get(var, ''),
                        'value_labels': metadata.get('value_labels', {}).get(var, {}),
                        'measure': metadata.get('variable_measure', {}).get(var, ''),
                        'format': metadata.get('variable_formats', {}).get(var, '')
                    }
                    questions_info.append(question_info)
            
            print(f"[DEBUG] Available questions:")
            for info in questions_info:
                print(f"- {info['variable']}: {info['question']}")
            
            # Get segmentation from context or params
            segmentation = context.get('segmentation') if context else params.get('segmentation')
            
            # Step 1: Find matching variable by checking all available metadata
            if segmentation and segmentation not in metadata.get('columns', []):
                if not params or 'chat_model' not in params:
                    raise ValueError("chat_model not provided in params")
                
                # Create a focused prompt to find matching question using all available metadata
                prompt = (
                    "Given the following questions and their metadata from the survey, find the one that best matches the segmentation request.\n"
                    f"Segmentation requested: {segmentation}\n\n"
                    "Available questions with their metadata:\n" +
                    "\n".join(
                        f"- Variable: {info['variable']}\n"
                        f"  Question: {info['question']}\n"
                        f"  Variable Label: {info['variable_label']}\n"
                        f"  Value Labels: {info['value_labels']}\n"
                        f"  Measure: {info['measure']}\n"
                        f"  Format: {info['format']}"
                        for info in questions_info
                    ) +
                    "\n\nInstructions:\n"
                    "1. Look for questions that match the segmentation request semantically\n"
                    "2. Consider both the question text and variable labels\n"
                    "3. For age, look for questions asking about age or containing age-related terms\n"
                    "4. Also check the value labels for age ranges or age-related values\n"
                    "5. Reply with the EXACT variable name that best matches\n"
                    "6. If no match is found, reply with 'NO_MATCH'"
                )
                
                print(f"[DEBUG] Sending prompt to LLM for variable matching:\n{prompt}")
                
                # Get LLM's response
                response = params['chat_model'].generate_reply(prompt)
                matched_var = response.strip()
                
                print(f"[DEBUG] LLM matched variable: {matched_var}")
                
                if matched_var == 'NO_MATCH':
                    raise ValueError(f"Could not find matching variable for segmentation '{segmentation}'")
                
                if matched_var not in metadata.get('columns', []):
                    raise ValueError(f"LLM returned invalid variable name: {matched_var}")
                
                # Get the complete metadata for the matched variable
                matched_info = next((info for info in questions_info if info['variable'] == matched_var), None)
                if not matched_info:
                    raise ValueError(f"Could not find metadata for matched variable: {matched_var}")
                
                print(f"[DEBUG] Matched variable: {matched_info['variable']}")
                print(f"[DEBUG] Matched question: {matched_info['question']}")
                
                # Create segmentation groups based on value labels
                segmentation_groups = {}
                for value, label in matched_info['value_labels'].items():
                    segmentation_groups[label] = value
                
                print(f"[DEBUG] Created segmentation groups: {segmentation_groups}")
                
                return {
                    'filters': {},
                    'segmentation': matched_var,
                    'explanation': f"Running Van Westendorp analysis segmented by {matched_info['question']}",
                    'followup_questions': [],
                    'segmentation_groups': segmentation_groups
                }
            
            # If no segmentation specified, check if we need to ask for it
            if "by" in question.lower():
                # Extract potential segmentation variable from question
                parts = question.lower().split("by")
                if len(parts) > 1:
                    potential_seg = parts[1].strip()
                    
                    # Use LLM to find matching variable
                    if params and 'chat_model' in params:
                        prompt = (
                            "Given the following questions and their metadata from the survey, find the one that best matches the segmentation request.\n"
                            f"Segmentation requested: {potential_seg}\n\n"
                            "Available questions with their metadata:\n" +
                            "\n".join(
                                f"- Variable: {info['variable']}\n"
                                f"  Question: {info['question']}\n"
                                f"  Variable Label: {info['variable_label']}\n"
                                f"  Value Labels: {info['value_labels']}\n"
                                f"  Measure: {info['measure']}\n"
                                f"  Format: {info['format']}"
                                for info in questions_info
                            ) +
                            "\n\nInstructions:\n"
                            "1. Look for questions that match the segmentation request semantically\n"
                            "2. Consider both the question text and variable labels\n"
                            "3. For age, look for questions asking about age or containing age-related terms\n"
                            "4. Also check the value labels for age ranges or age-related values\n"
                            "5. Reply with the EXACT variable name that best matches\n"
                            "6. If no match is found, reply with 'NO_MATCH'"
                        )
                        
                        print(f"[DEBUG] Sending prompt to LLM for variable matching:\n{prompt}")
                        
                        response = params['chat_model'].generate_reply(prompt)
                        matched_var = response.strip()
                        
                        print(f"[DEBUG] LLM matched variable: {matched_var}")
                        
                        if matched_var != 'NO_MATCH' and matched_var in metadata.get('columns', []):
                            # Get the complete metadata for the matched variable
                            matched_info = next((info for info in questions_info if info['variable'] == matched_var), None)
                            if matched_info:
                                # Create segmentation groups based on value labels
                                segmentation_groups = {}
                                for value, label in matched_info['value_labels'].items():
                                    segmentation_groups[label] = value
                                
                                return {
                                    'filters': {},
                                    'segmentation': matched_var,
                                    'explanation': f"Running Van Westendorp analysis segmented by {matched_info['question']}",
                                    'followup_questions': [],
                                    'segmentation_groups': segmentation_groups
                                }
            
            # Default case: no segmentation
            return {
                'filters': {},
                'segmentation': None,
                'explanation': "Running standard Van Westendorp analysis",
                'followup_questions': [],
                'segmentation_groups': {}
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to process user request: {str(e)}")
            # Return a safe default analysis plan
            return {
                'filters': {},
                'segmentation': None,
                'explanation': "Error processing request. Please try again.",
                'followup_questions': ["Could you please rephrase your request?"],
                'segmentation_groups': {}
            }

    def _validate_data(self, data: pd.DataFrame, column_map: Dict[str, str]) -> None:
        """
        Validate that the data in the mapped columns is valid for analysis.
        
        Args:
            data (pd.DataFrame): Input data
            column_map (Dict[str, str]): Mapping of required columns to actual column names
            
        Raises:
            ValueError: If data validation fails
        """
        for field, column in column_map.items():
            # Check if column exists
            if column not in data.columns:
                raise ValueError(f"Column '{column}' (mapped to {field}) not found in dataset")
            
            # Check for non-numeric values
            if not pd.to_numeric(data[column], errors='coerce').notna().all():
                raise ValueError(f"Column '{column}' contains non-numeric values")
            
            # Check for negative values
            if (data[column] < 0).any():
                raise ValueError(f"Column '{column}' contains negative values")
            
            # Check for empty/null values
            if data[column].isna().any():
                raise ValueError(f"Column '{column}' contains null values")

    def _calculate_curves(self, too_cheap: pd.Series, too_expensive: pd.Series, 
                         bargain: pd.Series, getting_expensive: pd.Series) -> tuple:
        """
        Calculate the cumulative distribution curves for Van Westendorp analysis.
        Following the complete formula set from documentation.
        
        Args:
            too_cheap: Series of "too cheap" price points
            too_expensive: Series of "too expensive" price points
            bargain: Series of "bargain" price points
            getting_expensive: Series of "getting expensive" price points
            
        Returns:
            tuple: (price_grid, tc_cum, te_cum, ba_cum, ge_cum)
        """
        try:
            # Convert to numeric, coercing errors to NaN
            too_cheap = pd.to_numeric(too_cheap, errors='coerce')
            too_expensive = pd.to_numeric(too_expensive, errors='coerce')
            bargain = pd.to_numeric(bargain, errors='coerce')
            getting_expensive = pd.to_numeric(getting_expensive, errors='coerce')
            
            # Remove any NaN values
            too_cheap = too_cheap.dropna()
            too_expensive = too_expensive.dropna()
            bargain = bargain.dropna()
            getting_expensive = getting_expensive.dropna()
            
            if len(too_cheap) == 0 or len(too_expensive) == 0 or len(bargain) == 0 or len(getting_expensive) == 0:
                raise ValueError("One or more price columns contain no valid numeric data")
            
            # Calculate price range with 10% buffer
            min_price = min(
                too_cheap.min(),
                too_expensive.min(),
                bargain.min(),
                getting_expensive.min()
            )
            max_price = max(
                too_cheap.max(),
                too_expensive.max(),
                bargain.max(),
                getting_expensive.max()
            )
            
            if min_price >= max_price:
                raise ValueError("Invalid price range: min_price >= max_price")
            
            # Apply 10% buffer to range
            price_range_min = min_price * 0.9
            price_range_max = max_price * 1.1
            
            # Create price grid with 200 points for smooth curves
            price_grid = np.linspace(price_range_min, price_range_max, 200)
            
            # Calculate cumulative distributions following the formula
            # TC(P) = (Number of respondents where too_cheap ≤ P) / N × 100
            tc_cum = np.array([(too_cheap <= p).mean() * 100 for p in price_grid])
            
            # B(P) = (Number of respondents where bargain ≤ P) / N × 100
            ba_cum = np.array([(bargain <= p).mean() * 100 for p in price_grid])
            
            # GE(P) = (Number of respondents where getting_expensive ≥ P) / N × 100
            ge_cum = np.array([(getting_expensive >= p).mean() * 100 for p in price_grid])
            
            # TE(P) = (Number of respondents where too_expensive ≥ P) / N × 100
            te_cum = np.array([(too_expensive >= p).mean() * 100 for p in price_grid])
            
            return price_grid, tc_cum, te_cum, ba_cum, ge_cum
            
        except Exception as e:
            print(f"[ERROR] Failed to calculate curves: {str(e)}")
            raise ValueError(f"Failed to calculate price curves: {str(e)}")

    def _find_intersection(self, x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> float:
        """
        Find the x-value where two curves intersect using linear interpolation.
        Following the complete formula set from documentation.
        
        Args:
            x: Array of x values (price points)
            y1: Array of y values for first curve
            y2: Array of y values for second curve
            
        Returns:
            float: x-value of intersection or 0.0 if no intersection found
        """
        try:
            # Find where the curves cross
            idx = np.argwhere(np.diff(np.signbit(y1 - y2))).flatten()
            
            if len(idx) == 0:
                print("[WARNING] No intersection found between curves")
                return 0.0
            
            # Get the points before and after intersection
            i = idx[0]
            if i >= len(x) - 1:
                return 0.0
                
            x1, x2 = x[i], x[i + 1]
            y1_1, y1_2 = y1[i], y1[i + 1]
            y2_1, y2_2 = y2[i], y2[i + 1]
            
            # Linear interpolation formula
            # x = x1 + (x2 - x1) * (y2_1 - y1_1) / ((y2_1 - y1_1) + (y1_2 - y2_2))
            intersection_x = x1 + (x2 - x1) * (y2_1 - y1_1) / ((y2_1 - y1_1) + (y1_2 - y2_2))
            
            # Validate the intersection point
            if intersection_x <= 0:
                print("[WARNING] Invalid intersection point (<= 0)")
                return 0.0
                
            return float(intersection_x)
            
        except Exception as e:
            print(f"[ERROR] Failed to find intersection: {str(e)}")
            return 0.0

    def _clean_data(self, data: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """
        Clean the data by removing rows with missing or invalid responses.
        
        Args:
            data (pd.DataFrame): Input data
            column_map (Dict[str, str]): Mapping of required columns to actual column names
            
        Returns:
            pd.DataFrame: Cleaned data with only complete responses
        """
        print("[DEBUG] Starting data cleaning...")
        print(f"[DEBUG] Initial data shape: {data.shape}")
        
        # Get the actual column names from the mapping
        columns = list(column_map.values())
        
        # Convert all columns to numeric, coercing errors to NaN
        for col in columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove rows where any of the required columns have NaN values
        cleaned_data = data.dropna(subset=columns)
        
        # Remove rows with negative values
        for col in columns:
            cleaned_data = cleaned_data[cleaned_data[col] >= 0]
        
        print(f"[DEBUG] Data shape after cleaning: {cleaned_data.shape}")
        print(f"[DEBUG] Removed {data.shape[0] - cleaned_data.shape[0]} rows with incomplete or invalid responses")
        
        if cleaned_data.shape[0] == 0:
            raise ValueError("No valid responses found after cleaning. All responses were either incomplete or invalid.")
        
        return cleaned_data

    def _generate_insights(self, results: Dict[str, Any], chat_model: Any) -> str:
        """
        Generate insights from the Van Westendorp analysis results.
        Following the complete formula set from documentation.
        
        Args:
            results: Dictionary containing analysis results
            chat_model: LLM model for generating insights
            
        Returns:
            str: Formatted insights
        """
        try:
            insights = []
            
            for segment_name, segment_results in results.items():
                segment_insights = []
                
                # Extract key metrics
                pmc = segment_results["pmc"]
                pme = segment_results["pme"]
                opp = segment_results["opp"]
                price_sensitivity = segment_results["price_sensitivity"]
                sample_size = segment_results["sample_size"]
                
                # Calculate range width and percentage
                range_width = pme - pmc
                range_percentage = (range_width / pmc) * 100
                
                # Generate price sensitivity interpretation
                if price_sensitivity < 50:
                    sensitivity_level = "low"
                elif price_sensitivity < 100:
                    sensitivity_level = "moderate"
                else:
                    sensitivity_level = "high"
                
                # Build segment insights
                segment_insights.extend([
                    f"Price Range Analysis:",
                    f"• Acceptable Price Range: ${pmc:.2f} - ${pme:.2f}",
                    f"• Range Width: ${range_width:.2f} ({range_percentage:.1f}% of PMC)",
                    f"• Optimal Price Point: ${opp:.2f}",
                    f"• Price Sensitivity: {price_sensitivity:.1f}% ({sensitivity_level})",
                    f"• Sample Size: {sample_size} respondents"
                ])
                
                # Add market penetration insights
                segment_insights.extend([
                    f"\nMarket Penetration at Key Price Points:",
                    f"• At ${pmc:.2f} (PMC):",
                    f"  - Not Too Cheap: {100 - pmc:.1f}%",
                    f"  - Not Too Expensive: {100 - pme:.1f}%",
                    f"• At ${opp:.2f} (OPP):",
                    f"  - Overall Acceptance: {min(100 - pmc, 100 - pme):.1f}%"
                ])
                
                # Add strategic recommendations
                segment_insights.extend([
                    f"\nStrategic Recommendations:",
                    f"• Primary Target Price: ${opp:.2f}",
                    f"• Price Flexibility: {'High' if range_percentage > 50 else 'Moderate' if range_percentage > 25 else 'Low'}",
                    f"• Market Sensitivity: {sensitivity_level.capitalize()}",
                    f"• Pricing Strategy: {'Premium' if opp > (pmc + pme)/2 else 'Value' if opp < (pmc + pme)/2 else 'Balanced'}"
                ])
                
                # Add segment name if multiple segments
                if len(results) > 1:
                    insights.append(f"\n=== {segment_name} ===")
                insights.extend(segment_insights)
            
            return "\n".join(insights)
            
        except Exception as e:
            print(f"[ERROR] Failed to generate insights: {str(e)}")
            return "Unable to generate detailed insights due to an error in the analysis."

    def run(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process the data using Van Westendorp's Price Sensitivity Meter (PSM) methodology.
        Following the complete formula set from documentation.
        
        Args:
            data (pd.DataFrame): Input data
            params (Dict[str, Any]): Parameters including:
                - column_map: Optional mapping of question types to column names
                - chat_model: LLM model for column mapping
                - question: Optional follow-up question about the analysis
                - conversation_context: Optional previous conversation context
                - auto_map: Boolean indicating whether to handle mapping automatically
                - filters: Dict of filters to apply (e.g. {"gender": "male"})
                - segmentation: Optional segmentation parameter
                - metadata: Dataset metadata
                
        Returns:
            Dict[str, Any]: Analysis results
        """
        print("\n[DEBUG] ===== Van Westendorp MCP Processing =====")
        print(f"[DEBUG] Input data shape: {data.shape}")
        
        # Check if this is a follow-up question
        if params.get("question"):
            print(f"[DEBUG] Processing question: {params['question']}")
            # Get the previous analysis result from params
            previous_result = params.get("previous_result")
            if previous_result:
                print("[DEBUG] Found previous result, handling as follow-up question")
                result = self.handle_followup_question(
                    params["question"], 
                    previous_result, 
                    data,
                    params.get("conversation_context")
                )
                return {
                    "visualizations": previous_result.get("visualizations", {}),
                    "insights": result["answer"],
                    "context": result["context"]
                }
        
        # Process user request to create analysis plan
        if params.get("question"):
            print("[DEBUG] Creating analysis plan from user request")
            analysis_plan = self._process_user_request(
                params["question"],
                params.get("metadata", {}),
                params.get("conversation_context"),
                params  # Pass the entire params dict to access chat_model
            )
            
            # Check if we need to ask follow-up questions
            if analysis_plan.get("followup_questions"):
                return {
                    "reply": "I need some clarification before proceeding with the analysis.",
                    "followup_questions": analysis_plan["followup_questions"],
                    "context": {
                        "analysis_type": self.name,
                        "pending_plan": analysis_plan
                    }
                }
            
            # Update params with plan
            params["filters"] = analysis_plan["filters"]
            params["segmentation"] = analysis_plan["segmentation"]
            params["analysis_explanation"] = analysis_plan["explanation"]
            params["segmentation_groups"] = analysis_plan.get("segmentation_groups", [])
        
        # Apply any filters
        filters = params.get("filters", {})
        if filters:
            print(f"[DEBUG] Applying filters: {json.dumps(filters, indent=2)}")
        data = self._apply_filters(data, filters)
        
        # Handle segmentation if requested
        segmentation = params.get("segmentation")
        if segmentation:
            print(f"[DEBUG] Applying segmentation by: {segmentation}")
            # Get segmentation groups from params or calculate them
            segmentation_groups = params.get("segmentation_groups", [])
            if not segmentation_groups:
                segmentation_groups = data[segmentation].unique().tolist()
            
            # Create segments dictionary
            segments = {}
            for group in segmentation_groups:
                segment_data = data[data[segmentation] == group]
                if len(segment_data) > 0:  # Only include segments with data
                    segments[str(group)] = segment_data
        else:
            segments = {"Overall": data}
        
        print(f"[DEBUG] Processing {len(segments)} segments")
        
        # Get column labels from metadata
        column_labels = params.get("metadata", {})
        print(f"[DEBUG] MCP received metadata keys: {list(column_labels.keys()) if column_labels else 'None'}")
        print(f"[DEBUG] MCP received column_labels: {column_labels.get('column_labels', {}) if column_labels else 'None'}")
        
        if not column_labels:
            print("[WARNING] No metadata provided in params")
            # Try to get metadata from the data loader
            try:
                from ..services.data_loader import load_metadata
                user_id = params.get("user_id")
                dataset_id = params.get("dataset_id")
                if user_id and dataset_id:
                    print(f"[DEBUG] Attempting to load metadata for user_id: {user_id}, dataset_id: {dataset_id}")
                    column_labels = load_metadata(user_id, dataset_id)
                    print(f"[DEBUG] Loaded metadata keys: {list(column_labels.keys()) if column_labels else 'None'}")
            except Exception as e:
                print(f"[ERROR] Failed to load metadata: {e}")
                column_labels = {}
        
        if not column_labels:
            raise ValueError("No metadata available for column mapping")
        
        # Extract the actual column labels dictionary
        actual_column_labels = column_labels.get("column_labels", {})
        print(f"[DEBUG] Actual column_labels: {actual_column_labels}")
        if not actual_column_labels:
            raise ValueError("No column_labels found in metadata")
        
        # Use LLM to propose mapping
        prompt = (
            "Given these column labels, which columns should be used for the Van Westendorp analysis?\n"
            "We need columns for: too_cheap, bargain, getting_expensive, too_expensive\n\n"
            f"Available columns:\n{json.dumps(actual_column_labels, indent=2)}\n\n"
            "Reply with a JSON object mapping each required field to a column name. Do not include any markdown formatting or backticks in your response."
        )
        mapping_text = params["chat_model"].generate_reply(prompt)
        
        try:
            # Clean the response text to ensure it's valid JSON
            mapping_text = mapping_text.strip()
            if mapping_text.startswith('```json'):
                mapping_text = mapping_text[7:]
            if mapping_text.endswith('```'):
                mapping_text = mapping_text[:-3]
            mapping_text = mapping_text.strip()
            
            column_map = json.loads(mapping_text)
            
            # Validate the mapping
            required_fields = ["too_cheap", "bargain", "getting_expensive", "too_expensive"]
            for field in required_fields:
                if field not in column_map:
                    raise ValueError(f"Missing required field in mapping: {field}")
                if column_map[field] not in data.columns:
                    raise ValueError(f"Column {column_map[field]} not found in dataset")
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[ERROR] Failed to get valid column mapping: {str(e)}")
            raise ValueError(f"Could not determine correct column mapping: {str(e)}")
        
        # Run analysis for each segment
        results = {}
        visualizations = {"charts": [], "tables": []}
        
        for segment_name, segment_data in segments.items():
            print(f"[DEBUG] Processing segment: {segment_name}")
            
            try:
                # Clean the data first
                cleaned_data = self._clean_data(segment_data, column_map)
                
                # Extract price columns using the mapping
                tc_col = column_map["too_cheap"]
                ba_col = column_map["bargain"]
                ge_col = column_map["getting_expensive"]
                te_col = column_map["too_expensive"]
                
                # Calculate price points and curves
                price_grid, tc_cum, te_cum, ba_cum, ge_cum = self._calculate_curves(
                    cleaned_data[tc_col],
                    cleaned_data[te_col],
                    cleaned_data[ba_col],
                    cleaned_data[ge_col]
                )
                
                # Find intersection points following the formula
                # PMC: Where "too cheap" equals "getting expensive"
                pmc = self._find_intersection(price_grid, tc_cum, ge_cum)
                
                # PME: Where "bargain" equals "too expensive"
                pme = self._find_intersection(price_grid, ba_cum, te_cum)
                
                # OPP: Where "too cheap" equals "too expensive"
                opp = self._find_intersection(price_grid, tc_cum, te_cum)
                
                # Calculate price sensitivity using Newton-Miller-Smith formula
                if pmc > 0 and pme > 0:
                    price_sensitivity = (pme - pmc) / pmc * 100
                else:
                    price_sensitivity = 0.0
                    print("[WARNING] Could not calculate price sensitivity due to invalid intersection points")
                
                # Store results for this segment
                results[segment_name] = {
                    "pmc": float(pmc),
                    "pme": float(pme),
                    "opp": float(opp),
                    "price_sensitivity": float(price_sensitivity),
                    "sample_size": len(cleaned_data)
                }
                
                # Generate visualizations for this segment
                segment_viz = self.generate_visualizations(
                    price_grid, tc_cum, te_cum, ba_cum, ge_cum,
                    pmc, pme, opp, price_sensitivity
                )
                
                # Add segment name and sample size to visualizations
                for chart in segment_viz.get("charts", []):
                    chart["title"] = f"{chart['title']} - {segment_name} (n={len(cleaned_data)})"
                for table in segment_viz.get("tables", []):
                    table["title"] = f"{table['title']} - {segment_name} (n={len(cleaned_data)})"
                    # Add sample size to table data
                    table["data"].append({
                        "metric": "Sample Size",
                        "value": str(len(cleaned_data))
                    })
                
                # Add to overall visualizations
                visualizations["charts"].extend(segment_viz.get("charts", []))
                visualizations["tables"].extend(segment_viz.get("tables", []))
                
            except Exception as e:
                print(f"[ERROR] Failed to process segment {segment_name}: {str(e)}")
                raise ValueError(f"Failed to process segment {segment_name}: {str(e)}")
        
        # Generate insights using the enhanced method
        insights = self._generate_insights(results, params.get("chat_model"))
        
        print("[DEBUG] ======================================\n")
        
        # Create response
        return {
            "reply": (
                "I've completed the Van Westendorp analysis"
                + (f" for {len(segments)} segments" if len(segments) > 1 else "")
                + ". Here are the results and insights."
            ),
            "visualizations": visualizations,
            "insights": insights,
            "context": {
                "analysis_type": self.name,
                "variables_used": self.required_columns,
                "column_map": column_map,
                "filters": filters,
                "segmentation": segmentation,
                "segmentation_groups": params.get("segmentation_groups", []),
                "results": results
            }
        }

# Logic for Van Westendorp price sensitivity analysis 