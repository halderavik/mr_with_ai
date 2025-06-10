# backend/app/mcp/van_westendorp.py

import pandas as pd
import numpy as np
from typing import Dict, Any
from app.services.mcp_base import MCPBase
from app.services.plotting import fig_to_base64
import matplotlib.pyplot as plt
from app.utils.common import filter_dataframe
import json


class VanWestendorpMCP(MCPBase):
    """
    Van Westendorp's Price Sensitivity Meter (PSM) implementation.
    Analyzes price sensitivity using four key price points.
    """
    
    def run(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the data using Van Westendorp's Price Sensitivity Meter (PSM) methodology.
        
        Args:
            data (pd.DataFrame): Input data
            params (Dict[str, Any]): Parameters including:
                - column_map: Optional mapping of question types to column names
                - chat_model: LLM model for column mapping
                
        Returns:
            Dict[str, Any]: Analysis results
        """
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
            
            # 11. Build results
            results = {
                "price_points": {
                    "pmc": float(pmc),
                    "pme": float(pme),
                    "opp": float(opp),
                    "price_sensitivity": float(price_sensitivity)
                },
                "curves": {
                    "price_grid": price_grid.tolist(),
                    "too_cheap": tc_cum.tolist(),
                    "too_expensive": te_cum.tolist(),
                    "bargain": ba_cum.tolist(),
                    "getting_expensive": ge_cum.tolist()
                },
                "insights": (
                    f"Using {n} respondents, the Van Westendorp analysis yields:\n"
                    f"• Point of Marginal Cheapness (PMC): ${pmc:.2f}\n"
                    f"• Point of Marginal Expensiveness (PME): ${pme:.2f}\n"
                    f"• Optimal Price (where acceptable = expensive): ${opp:.2f}\n"
                    f"The acceptable price range (between PMC and PME) suggests that most respondents "
                    f"find prices between ${pmc:.2f} and ${pme:.2f} acceptable. "
                    f"Above PME, more respondents consider the product too expensive."
                )
            }
            
            print("\n[DEBUG] Analysis complete")
            return results

# Logic for Van Westendorp price sensitivity analysis 