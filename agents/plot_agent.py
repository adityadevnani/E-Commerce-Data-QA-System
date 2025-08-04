import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from langchain.schema import SystemMessage, HumanMessage
from agents.shared_llm import llm
import contextlib

def generate_plot_from_llm(df: pd.DataFrame, question: str):
    try:
        df = df.copy()
        df.columns = df.columns.str.lower().str.replace('[^0-9a-zA-Z_]', '', regex=True)

        if len(df) > 50 and len(df.columns) > 3:
            system_prompt = f"""
You are an expert Python data analyst. You have been given a raw pandas DataFrame named `df`.
Your task is to write a single Python script that first **aggregates** this raw data into a meaningful summary, and then **plots** that summary using seaborn/matplotlib.

The user's query is: "{question}"
The available columns in the raw `df` are: {', '.join(df.columns)}

RULES:
1. The script must perform a pandas aggregation (like .groupby()) to create a new, summarized DataFrame.
2. The script must then use this new summarized DataFrame to create a plot.
3. Use `plt.figure(figsize=(10, 6))`, set a title, and set axis labels.
4. Do NOT use `plt.show()`.
5. Return ONLY the complete, executable Python script.
"""
        else: 
            system_prompt = f"""
You are a Python data visualization expert. Your only job is to write a single, clean block of seaborn/matplotlib code to create a plot from the given (already aggregated) pandas DataFrame `df`.
The user's request is: "{question}"
The available columns in `df` are: {', '.join(df.columns)}

RULES:
- For time-series data, use `sns.lineplot()`. For categorical data, use `sns.barplot()`.
- Set a clear title and axis labels.
- If x-axis labels are long, rotate them with `plt.xticks(rotation=45, ha='right')`.
- Return ONLY the Python code.
"""
        
        messages = [SystemMessage(content=system_prompt)]
        response = llm.invoke(messages).content.strip().replace("```python", "").replace("```", "").strip()
        code = response

        plt.clf(); plt.close("all")
        local_vars = {"df": df, "plt": plt, "sns": sns, "np": np, "pd": pd}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(code, {}, local_vars)

        fig = plt.gcf()
        if not fig.axes or all(not ax.has_data() for ax in fig.axes):
            return "{\"error\":\"Generated code did not produce a plot.\"}"

        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        plt.close("all")
        return buf

    except Exception as e:
        plt.close("all")
        return f"{{\"error\":\"Could not generate valid plot code. Details: {e}\"}}"

def intelligent_table_selection(question: str, tables: dict):
    try:
        table_info = ""
        for name, df in tables.items():
            sample = df.head(2).to_string(index=False) if not df.empty else "Empty"
            table_info += f"{name}:\n  Columns: {', '.join(df.columns)}\n  Sample:\n{sample}\n\n"

        messages = [
            SystemMessage(content=(
                "You are a data analyst deciding which DataFrame(s) to use for a user's plot.\n"
                "Respond with:\n"
                "- PRIMARY_TABLE: [table]\n"
                "- JOIN_TABLES: [comma-separated or NONE]\n"
                "- JOIN_KEYS: [optional keys]\n"
                "- REASONING: [brief reason]\n\n"
                "Only return values in that format. Available tables:\n" + table_info
            )),
            HumanMessage(content=f"User question: {question}")
        ]

        response = llm.invoke(messages).content.strip()

        primary_table = None
        join_tables = []
        for line in response.splitlines():
            if line.startswith("PRIMARY_TABLE:"):
                primary_table = line.split(":", 1)[1].strip()
            elif line.startswith("JOIN_TABLES:"):
                val = line.split(":", 1)[1].strip()
                if val.upper() != "NONE":
                    join_tables = [t.strip() for t in val.split(",") if t.strip()]

        if not primary_table or primary_table not in tables:
            return None, None

        df = tables[primary_table].copy()

        for join_table in join_tables:
            if join_table not in tables:
                continue
            for key in ["order_id", "customer_id", "product_id"]:
                if key in df.columns and key in tables[join_table].columns:
                    df = df.merge(tables[join_table], on=key, how="left")
                    break
        return df, primary_table
    except Exception as e:
        print(f"Table selection failed: {e}")
        return None, None

def handle_plot_agent(question: str, tables: dict):
    """
    Main entry point for plotting: selects table(s) and returns a plot image or error message.
    """
    try:
    
        if 'data' in locals() and isinstance(locals()['data'], pd.DataFrame):
            selected_df = locals()['data']
        else:

            selected_df, _ = intelligent_table_selection(question, tables)
        
        if selected_df is None or selected_df.empty:
            return "Could not determine appropriate table for your question."

        plot_result = generate_plot_from_llm(selected_df, question)
        return plot_result
    except Exception as e:
        return f"Plot agent failed: {e}"