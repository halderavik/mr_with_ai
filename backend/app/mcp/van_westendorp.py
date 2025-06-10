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
                        # Verify the key exists in actual_column_labels
                        if value in actual_column_labels:
                            col_map[key] = value
                            print(f"[DEBUG] Found match for {key}: {value} -> {actual_column_labels[value]}")
                        else:
                            print(f"[WARNING] Key '{value}' not found in column labels")
                            col_map[key] = None
            
            print("[DEBUG] Parsed column mapping:", col_map)
            print("[DEBUG] Available column labels:", list(actual_column_labels.keys()))

            # 4. Fallback to params or default mapping if LLM mapping is incomplete
            if params.get("column_map"):
                default_map = {k: v for k, v in params["column_map"].items() if k in ["too_cheap", "bargain", "getting_expensive", "too_expensive"]}
                default_map.update(col_map)
            else:
                default_map = {k: v for k, v in col_map.items() if k in ["too_cheap", "bargain", "getting_expensive", "too_expensive"]}

        # 5. If not confirmed, return mapping for user confirmation
        if not params.get("confirmed"):
            return {
                "needs_confirmation": True,
                "proposed_mapping": default_map,
                "message": (
                    "I have analyzed your data and propose the following variable mapping for the Van Westendorp analysis. "
                    "Please confirm or edit the mapping before proceeding:",
                    default_map
                )
            }

        # 6. Proceed with analysis if confirmed
        # Ensure all required columns are present
        # Resolve col_map values to actual DataFrame columns if they are labels
        resolved_col_map = {}
        for k, v in default_map.items():
            if v in data.columns:
                resolved_col_map[k] = v
            elif column_labels and isinstance(column_labels, dict) and "column_labels" in column_labels:
                # Try to map label back to column name
                for col, label in column_labels["column_labels"].items():
                    if v == label and col in data.columns:
                        resolved_col_map[k] = col
                        break
                else:
                    resolved_col_map[k] = None
            else:
                resolved_col_map[k] = None

        print("[DEBUG] Final resolved column mapping for Van Westendorp:", resolved_col_map)

        # Ensure all required columns are present
        for key, colname in resolved_col_map.items():
            if not colname or colname not in data.columns:
                raise ValueError(f"Required column '{colname}' for '{key}' not found in data.")

        # Extract the four price arrays using resolved mapping
        p_tc = data[resolved_col_map["too_cheap"]].astype(float).dropna().values
        p_ba = data[resolved_col_map["bargain"]].astype(float).dropna().values
        p_ge = data[resolved_col_map["getting_expensive"]].astype(float).dropna().values
        p_te = data[resolved_col_map["too_expensive"]].astype(float).dropna().values

        # 7. Build Price Grid:
        all_prices = np.concatenate([p_tc, p_ba, p_ge, p_te])
        # Create a sorted unique price grid (e.g. from min to max, step = 0.01 of range):
        price_grid = np.linspace(all_prices.min(), all_prices.max(), 200)

        # 8. Compute cumulative distributions:
        # For each price in grid, compute:
        #   % respondents saying "too cheap" <= price (i.e. they think it's no longer "too cheap")
        #   % respondents saying "bargain" <= price (i.e. they consider price acceptable)
        #   % respondents saying "getting expensive" <= price
        #   % respondents saying "too expensive" <= price
        cum_tc = [np.mean(p_tc < x) for x in price_grid]
        cum_ba = [np.mean(p_ba < x) for x in price_grid]
        cum_ge = [np.mean(p_ge < x) for x in price_grid]
        cum_te = [np.mean(p_te < x) for x in price_grid]

        # Convert to numpy arrays:
        cum_tc = np.array(cum_tc)
        cum_ba = np.array(cum_ba)
        cum_ge = np.array(cum_ge)
        cum_te = np.array(cum_te)

        # 9. Find intersection points:
        #   - Point of Marginal Cheapness = intersection of "too_cheap" & "getting_expensive"
        #   - Point of Marginal Expensiveness = intersection of "bargain" & "too_expensive"
        #   - Optimal Price = intersection of "getting_expensive" & "bargain" (the P*).
        def find_intersection(x_vals, y1, y2):
            # Find x where |y1 - y2| is minimized
            idx = np.argmin(np.abs(y1 - y2))
            return x_vals[idx]

        try:
            pmc = find_intersection(price_grid, cum_tc, cum_ge)
            pme = find_intersection(price_grid, cum_ba, cum_te)
            p_opt = find_intersection(price_grid, cum_ba, cum_ge)
        except Exception:
            pmc = pme = p_opt = None

        # 10. Build a result table (for tabular output):
        table = []
        for i, price in enumerate(price_grid):
            table.append({
                "price": round(float(price), 2),
                "% Not Too Cheap": round(float(cum_tc[i]*100), 1),
                "% Acceptable (≤ Bargain)": round(float(cum_ba[i]*100), 1),
                "% Getting Expensive": round(float(cum_ge[i]*100), 1),
                "% Too Expensive": round(float(cum_te[i]*100), 1),
            })

        # 11. Create the Van Westendorp plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(price_grid, cum_tc * 100, label="% Not Too Cheap", color="blue")
        ax.plot(price_grid, cum_ba * 100, label="% Acceptable", color="green")
        ax.plot(price_grid, cum_ge * 100, label="% Getting Expensive", color="orange")
        ax.plot(price_grid, cum_te * 100, label="% Too Expensive", color="red")
        ax.axvline(p_opt, linestyle="--", color="black", label=f"Opt Price = {p_opt:.2f}")
        ax.axvline(pmc, linestyle=":", color="gray", label=f"PMC = {pmc:.2f}")
        ax.axvline(pme, linestyle=":", color="brown", label=f"PME = {pme:.2f}")
        ax.set_xlabel("Price")
        ax.set_ylabel("Percent of Respondents (%)")
        ax.set_title("Van Westendorp Price Sensitivity Curves")
        ax.legend(loc="best")
        fig.tight_layout()

        # Convert figure to base64 for embedding:
        chart_b64 = fig_to_base64(fig)

        # 12. Build "insights" text:
        n_resp = len(data)
        insights = (
            f"Using {n_resp} respondents, the Van Westendorp analysis yields:\n"
            f"• Point of Marginal Cheapness (PMC): ${pmc:.2f}\n"
            f"• Point of Marginal Expensiveness (PME): ${pme:.2f}\n"
            f"• Optimal Price (where acceptable = expensive): ${p_opt:.2f}\n"
            "The acceptable price range (between PMC and PME) suggests that most respondents\n"
            f"find prices between ${pmc:.2f} and ${pme:.2f} acceptable.  "
            "Above PME, more respondents consider the product too expensive."
        )
        print("[DEBUG] Van Westendorp Analysis Insights:", insights)

        return {
            "tables": {"van_westendorp_distribution": table},
            "charts": {"van_westendorp_curve": chart_b64},
            "insights": insights
        }

# Logic for Van Westendorp price sensitivity analysis 