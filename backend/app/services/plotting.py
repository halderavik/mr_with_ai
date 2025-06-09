import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import uuid

# directory where static charts will be saved
BASE_CHART_DIR = Path(__file__).parent.parent / "static" / "charts"
BASE_CHART_DIR.mkdir(parents=True, exist_ok=True)

def save_matplotlib_figure(fig, prefix: str = "chart") -> str:
    """
    Save a matplotlib figure to disk under a unique PNG name.
    Returns the relative URL (e.g. '/static/charts/xxx.png').
    """
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    filepath = BASE_CHART_DIR / filename
    fig.savefig(str(filepath), bbox_inches="tight")
    plt.close(fig)
    return f"/static/charts/{filename}"

def fig_to_base64(fig) -> str:
    """
    Convert a matplotlib figure into a base64‐encoded data URI.
    Frontend can embed this directly into an <img src="...">.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"
