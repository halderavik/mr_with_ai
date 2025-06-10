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

    def run(self, data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process the data using Van Westendorp's Price Sensitivity Meter (PSM) methodology.
        
        Args:
            data (pd.DataFrame): Input data
            params (Dict[str, Any]): Parameters including:
                - column_map: Optional mapping of question types to column names
                - chat_model: LLM model for column mapping
                - question: Optional follow-up question about the analysis
                - conversation_context: Optional previous conversation context
                
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Check if this is a follow-up question
        if params.get("question"):
            # Get the previous analysis result from params
            previous_result = params.get("previous_result")
            if previous_result:
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
        
        print("[DEBUG] Starting Van Westendorp analysis...")
        print("[DEBUG] Input data shape:", data.shape)
        print("[DEBUG] Available columns:", list(data.columns))
        
        # 1. Get column labels from metadata
        column_labels = params.get("metadata", {})
        if not column_labels:
            print("[WARNING] No metadata provided in params")
            # Try to get metadata from the data loader
            try:
                from ..services.data_loader import load_metadata
                user_id = params.get("user_id")
                dataset_id = params.get("dataset_id")
                if user_id and dataset_id:
                    column_labels = load_metadata(user_id, dataset_id)
                    print("[DEBUG] Loaded metadata from file:", column_labels)
            except Exception as e:
                print(f"[ERROR] Failed to load metadata: {e}")
                column_labels = {}
        
        if not column_labels:
            raise ValueError("No metadata available for column mapping")
            
        print("[DEBUG] Processing metadata for column mapping...")
        print("[DEBUG] Full metadata structure:", json.dumps(column_labels, indent=2))
        
        # Extract the actual column labels dictionary
        actual_column_labels = column_labels.get("column_labels", {})
        if not actual_column_labels:
            raise ValueError("No column_labels found in metadata")
            
        print("[DEBUG] Column labels dictionary:", json.dumps(actual_column_labels, indent=2))
        print("[DEBUG] Number of column labels:", len(actual_column_labels))
        
        # 2. Check if column mapping is provided in params
        if params.get("column_map"):
            print("[DEBUG] Using provided column mapping:", params["column_map"])
            col_map = params["column_map"]
            # If a mapping is provided, treat as confirmed
            column_map_confirmed = params.get("column_map_confirmed", False)
            if not column_map_confirmed:
                column_map_confirmed = True
        else:
            # 3. Use LLM to propose mapping
            prompt = (
                "You are an expert in survey data analysis. Your task is to analyze the SPSS column labels and find the exact question numbers "
                "that correspond to the Van Westendorp price sensitivity questions.\n\n"
                "Look through the values (survey questions) in the column labels dictionary and find questions that ask about:\n"
                "1. A price point that is 'Too Cheap' or 'Too Low' for the product\n"
                "2. A price point that represents a 'Bargain' or 'Good Value' for the product\n"
                "3. A price point where the product is 'Getting Expensive' or 'Starting to be Expensive'\n"
                "4. A price point that is 'Too Expensive' or 'Too High' for the product\n\n"
                "Here are the actual column labels from the survey:\n"
                f"{json.dumps(actual_column_labels, indent=2)}\n\n"
                "For each of the 4 Van Westendorp questions, return the EXACT key (question number) from the dictionary that contains the matching question.\n"
                "Return in this format:\n"
                "too_cheap: [exact_key_from_dict]\n"
                "bargain: [exact_key_from_dict]\n"
                "getting_expensive: [exact_key_from_dict]\n"
                "too_expensive: [exact_key_from_dict]\n\n"
                "Example: If you find a question like 'Q1: At what price would you consider this product to be too cheap?' with key 'Q1', "
                "you would return 'too_cheap: Q1'\n\n"
                "If you can't find a good match for any question, use 'None' for that key."
            )
            # Use chat_model from params instead of importing it
            chat_model = params.get("chat_model")
            if not chat_model:
                raise ValueError("chat_model not provided in params")
            llm_reply = chat_model.generate_reply(prompt)
            print("[DEBUG] Deepseek LLM reply for column mapping:", llm_reply)
            # Parse the LLM reply to extract the mapping
            col_map = {}
            for line in llm_reply.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key in ["too_cheap", "bargain", "getting_expensive", "too_expensive"]:
                        # Store the question number directly
                        col_map[key] = value
                        print(f"[DEBUG] Found match for {key}: {value}")
            print("[DEBUG] Parsed column mapping:", col_map)
            print("[DEBUG] Available DataFrame columns:", list(data.columns))
            column_map_confirmed = False

        # Require user confirmation of mapping before running analysis
        if not column_map_confirmed:
            # Compose a chat message for the user to confirm or edit the mapping
            mapping_str = '\n'.join([f"{k}: {v}" for k, v in col_map.items()])
            reply = (
                "Before running the Van Westendorp analysis, please confirm the variable mapping for your data columns. "
                "Here is the proposed mapping:\n\n"
                f"{mapping_str}\n\n"
                "If this mapping is correct, reply with 'yes' or 'confirm'. "
                "If you want to edit, reply with the correct mapping in the same format."
            )
            return {
                "reply": reply,
                "context": {
                    "analysis_type": self.name,
                    "proposed_column_map": col_map,
                    "variables_used": self.required_columns,
                    "column_map_confirmed": False
                }
            }

        # 4. Fallback to params or default mapping if LLM mapping is incomplete
        if params.get("column_map"):
            default_map = {k: v for k, v in params["column_map"].items() if k in ["too_cheap", "bargain", "getting_expensive", "too_expensive"]}
            default_map.update(col_map)
        else:
            default_map = {k: v for k, v in col_map.items() if k in ["too_cheap", "bargain", "getting_expensive", "too_expensive"]}

        # 5. Extract data using the question numbers
        print("[DEBUG] Extracting data using question numbers...")
        
        # First, create a mask for respondents who answered all questions
        valid_respondents = data[list(default_map.values())].notna().all(axis=1)
        print(f"\n[DEBUG] Total respondents: {len(data)}")
        print(f"[DEBUG] Respondents with all questions answered: {valid_respondents.sum()}")
        
        # Filter data for valid respondents
        filtered_data = data[valid_respondents]
        print(f"[DEBUG] Using {len(filtered_data)} respondents for analysis")
        
        price_data = {}
        for key, question_num in default_map.items():
            if question_num in filtered_data.columns:
                # Get the raw data for valid respondents
                raw_data = filtered_data[question_num]
                print(f"\n[DEBUG] Raw data for {key} ({question_num}):")
                print(f"First 20 rows: {raw_data.head(20).tolist()}")
                print(f"Data type: {raw_data.dtype}")
                print(f"Number of values: {len(raw_data)}")
                
                # Convert to float
                try:
                    price_data[key] = raw_data.astype(float).values
                except ValueError as e:
                    print(f"[WARNING] Direct conversion failed for {key}: {e}")
                    # Try cleaning the data first
                    cleaned_data = raw_data.str.replace('$', '').str.replace(',', '').astype(float)
                    price_data[key] = cleaned_data.values
                
                print(f"[DEBUG] Processed data for {key}:")
                print(f"Number of values: {len(price_data[key])}")
                print(f"First 10 values: {price_data[key][:10]}")
                print(f"Min value: {min(price_data[key])}")
                print(f"Max value: {max(price_data[key])}")
                print(f"Mean value: {np.mean(price_data[key])}")
            else:
                print(f"[ERROR] Question number {question_num} not found in DataFrame columns")
                raise ValueError(f"Required column '{question_num}' for '{key}' not found in data.")

        # 6. Run Van Westendorp analysis
        print("\n[DEBUG] Running Van Westendorp analysis...")
        p_tc = price_data["too_cheap"]
        p_ba = price_data["bargain"]
        p_ge = price_data["getting_expensive"]
        p_te = price_data["too_expensive"]

        print("\n[DEBUG] Price arrays summary:")
        print(f"Too Cheap: {len(p_tc)} values, range: [{min(p_tc)}, {max(p_tc)}]")
        print(f"Bargain: {len(p_ba)} values, range: [{min(p_ba)}, {max(p_ba)}]")
        print(f"Getting Expensive: {len(p_ge)} values, range: [{min(p_ge)}, {max(p_ge)}]")
        print(f"Too Expensive: {len(p_te)} values, range: [{min(p_te)}, {max(p_te)}]")

        # 7. Build Price Grid:
        print("\n[DEBUG] Building price grid...")
        all_prices = np.concatenate([p_tc, p_ba, p_ge, p_te])
        min_price = np.min(all_prices)
        max_price = np.max(all_prices)
        print(f"[DEBUG] Price range: {min_price} to {max_price}")
        
        # Create a sorted unique price grid
        price_step = (max_price - min_price) / 100  # Use 100 steps for better precision
        price_grid = np.arange(min_price, max_price + price_step, price_step)
        print(f"[DEBUG] Price grid size: {len(price_grid)}")
        print(f"[DEBUG] Price grid range: {price_grid[0]} to {price_grid[-1]}")
        
        # 8. Calculate cumulative distributions
        print("\n[DEBUG] Calculating cumulative distributions...")
        n = len(p_tc)  # Number of respondents
        print(f"[DEBUG] Number of respondents: {n}")
        
        # Too Cheap curve (increasing)
        tc_cum = np.array([np.sum(p_tc <= p) for p in price_grid]) / n * 100
        # Too Expensive curve (decreasing)
        te_cum = np.array([np.sum(p_te <= p) for p in price_grid]) / n * 100
        # Bargain curve (increasing)
        ba_cum = np.array([np.sum(p_ba <= p) for p in price_grid]) / n * 100
        # Getting Expensive curve (decreasing)
        ge_cum = np.array([np.sum(p_ge <= p) for p in price_grid]) / n * 100
        
        print("[DEBUG] Cumulative distribution ranges:")
        print(f"Too Cheap: {min(tc_cum):.1f}% to {max(tc_cum):.1f}%")
        print(f"Too Expensive: {min(te_cum):.1f}% to {max(te_cum):.1f}%")
        print(f"Bargain: {min(ba_cum):.1f}% to {max(ba_cum):.1f}%")
        print(f"Getting Expensive: {min(ge_cum):.1f}% to {max(ge_cum):.1f}%")
        
        # 9. Find key price points
        print("\n[DEBUG] Finding key price points...")
        # Point of Marginal Cheapness (PMC): where too cheap = bargain
        pmc_idx = np.argmin(np.abs(tc_cum - ba_cum))
        pmc = price_grid[pmc_idx]
        
        # Point of Marginal Expensiveness (PME): where too expensive = getting expensive
        pme_idx = np.argmin(np.abs(te_cum - ge_cum))
        pme = price_grid[pme_idx]
        
        # Optimal Price Point (OPP): where acceptable = expensive
        # Calculate acceptable range (between PMC and PME)
        acceptable_range = (price_grid >= pmc) & (price_grid <= pme)
        if np.any(acceptable_range):
            opp_idx = np.argmin(np.abs(ba_cum[acceptable_range] - ge_cum[acceptable_range]))
            opp = price_grid[acceptable_range][opp_idx]
        else:
            opp = (pmc + pme) / 2
            print("[WARNING] No acceptable price range found, using midpoint")
        
        print("[DEBUG] Key price points:")
        print(f"PMC: ${pmc:.2f}")
        print(f"PME: ${pme:.2f}")
        print(f"OPP: ${opp:.2f}")
        
        # 10. Calculate price sensitivity
        price_sensitivity = (pme - pmc) / pmc * 100
        
        # After calculating all the metrics, generate visualizations
        visualizations = self.generate_visualizations(
            price_grid=price_grid,
            tc_cum=tc_cum,
            te_cum=te_cum,
            ba_cum=ba_cum,
            ge_cum=ge_cum,
            pmc=pmc,
            pme=pme,
            opp=opp,
            price_sensitivity=price_sensitivity
        )
        
        # Build the final results
        results = {
            "visualizations": visualizations,
            "insights": (
                f"Using {n} respondents, the Van Westendorp analysis yields:\n"
                f"• Point of Marginal Cheapness (PMC): ${pmc:.2f}\n"
                f"• Point of Marginal Expensiveness (PME): ${pme:.2f}\n"
                f"• Optimal Price (where acceptable = expensive): ${opp:.2f}\n"
                f"The acceptable price range (between PMC and PME) suggests that most respondents "
                f"find prices between ${pmc:.2f} and ${pme:.2f} acceptable. "
                f"Above PME, more respondents consider the product too expensive."
            ),
            "context": {
                "analysis_type": self.name,
                "variables_used": self.required_columns
            }
        }
        
        # Polish the main reply using the chat model
        chat_model = params.get("chat_model")
        if chat_model:
            reply_prompt = (
                f"Given the following Van Westendorp analysis results, provide a clear, business-focused summary and next steps for a product manager:\n\n"
                f"{results['insights']}\n"
                "Be concise and actionable."
            )
            reply = chat_model.generate_reply(reply_prompt)
        else:
            reply = results['insights']
        results['reply'] = reply
        
        print("\n[DEBUG] Analysis complete")
        print("[DEBUG] Sending visualizations:", json.dumps(visualizations, indent=2))
        return results

# Logic for Van Westendorp price sensitivity analysis 