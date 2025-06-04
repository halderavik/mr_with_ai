# backend/app/mcp/van_westendorp.py

import pandas as pd
import numpy as np
from ..services.mcp_base import MCPBase
from ..services.plotting import fig_to_base64
import matplotlib.pyplot as plt
from ..utils.common import filter_dataframe

class VanWestendorpMCP(MCPBase):
    """
    Implements Van Westendorp price sensitivity analysis.
    Assumes the DataFrame has columns:
      - "too_cheap"
      - "bargain"
      - "getting_expensive"
      - "too_expensive"
    All columns are numeric (price points) for each respondent.

    params may contain:
      - filter_column (str): column to filter on (e.g. "Gender")
      - filter_value (Any): value to select (e.g. "Male")
      - column_map (dict): override column names if survey used different names
        e.g. {"too_cheap": "Q1_cheap", "bargain": "Q2_bargain", ...}
    """
    def run(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Filter if requested:
        if params.get("filter_column") and params.get("filter_value") is not None:
            df = filter_dataframe(df, params["filter_column"], params["filter_value"])

        # 2. Determine column mapping:
        col_map = {
            "too_cheap": "too_cheap",
            "bargain": "bargain",
            "getting_expensive": "getting_expensive",
            "too_expensive": "too_expensive"
        }
        if params.get("column_map"):
            col_map.update(params["column_map"])

        # Ensure columns exist:
        for key, colname in col_map.items():
            if colname not in df.columns:
                raise ValueError(f"Required column '{colname}' not found in data.")

        # Extract the four price arrays:
        p_tc = df[col_map["too_cheap"]].astype(float).dropna().values
        p_ba = df[col_map["bargain"]].astype(float).dropna().values
        p_ge = df[col_map["getting_expensive"]].astype(float).dropna().values
        p_te = df[col_map["too_expensive"]].astype(float).dropna().values

        # 3. Build Price Grid:
        all_prices = np.concatenate([p_tc, p_ba, p_ge, p_te])
        # Create a sorted unique price grid (e.g. from min to max, step = 0.01 of range):
        price_grid = np.linspace(all_prices.min(), all_prices.max(), 200)

        # 4. Compute cumulative distributions:
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

        # 5. Find intersection points:
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

        # 6. Build a result table (for tabular output):
        table = []
        for i, price in enumerate(price_grid):
            table.append({
                "price": round(float(price), 2),
                "% Not Too Cheap": round(float(cum_tc[i]*100), 1),
                "% Acceptable (≤ Bargain)": round(float(cum_ba[i]*100), 1),
                "% Getting Expensive": round(float(cum_ge[i]*100), 1),
                "% Too Expensive": round(float(cum_te[i]*100), 1),
            })

        # 7. Create the Van Westendorp plot:
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

        # 8. Build "insights" text:
        n_resp = len(df)
        insights = (
            f"Using {n_resp} respondents, the Van Westendorp analysis yields:\n"
            f"• Point of Marginal Cheapness (PMC): ${pmc:.2f}\n"
            f"• Point of Marginal Expensiveness (PME): ${pme:.2f}\n"
            f"• Optimal Price (where acceptable = expensive): ${p_opt:.2f}\n"
            "The acceptable price range (between PMC and PME) suggests that most respondents\n"
            f"find prices between ${pmc:.2f} and ${pme:.2f} acceptable.  "
            "Above PME, more respondents consider the product too expensive."
        )

        return {
            "tables": {"van_westendorp_distribution": table},
            "charts": {"van_westendorp_curve": chart_b64},
            "insights": insights
        }

# Logic for Van Westendorp price sensitivity analysis 