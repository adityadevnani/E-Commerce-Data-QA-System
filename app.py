import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import io
import traceback

from agents.graph_agent import run_agent_chain
from agents.plot_agent import generate_plot_from_llm

st.set_page_config(page_title="E-Commerce QA", layout="wide")

st.markdown("""
<style>
    body { background-color: #0e1117; color: #fafafa; }
    .stApp { background-color: #0e1117; }
    .stTextInput > div > div > input { color: #fafafa; background-color: #1c1f26; }
    .st-emotion-cache-1cpxqw2 { background-color: #1c1f26; }
    .stAlert { border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

load_dotenv()

@st.cache_data
def load_data():
    """Loads all e-commerce data from CSV files."""
    customers = pd.read_csv("data/customers.csv")
    order_items = pd.read_csv("data/order_items.csv")
    orders = pd.read_csv("data/orders.csv", parse_dates=['order_purchase_timestamp', 'order_approved_at', 'order_delivered_timestamp'])
    payments = pd.read_csv("data/payments.csv")
    products = pd.read_csv("data/products.csv")
    for df in [customers, order_items, orders, payments, products]:
        df.columns = df.columns.str.strip()
    return {"customers": customers, "orders": orders, "order_items": order_items, "payments": payments, "products": products}

def enrich_datetime_columns(df):
    """Adds year, month, etc., columns for plotting."""
    if not isinstance(df, pd.DataFrame): return df
    df = df.copy()
    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_date"] = df[col].dt.date
    return df

st.title("E-Commerce Data QA System")
st.markdown("Ask a question about orders, revenue, delivery, or customer behavior:")

question = st.text_input("Ask Your Question", key="user_question").strip()
tables = load_data()

if question:
    with st.spinner("Thinking..."):
        try:
            result_dict = run_agent_chain(question, tables)

            answer = result_dict.get("answer")
            data = result_dict.get("data")
            summary = result_dict.get("summary")
            show_plot = result_dict.get("plot", False)
            show_data = result_dict.get("show_data", False)

            is_simple_answer = not show_data and not show_plot

            if is_simple_answer:
                st.markdown("### Answer")
                st.success(answer)

            if show_plot:
                if isinstance(data, pd.DataFrame) and not data.empty:
                    st.markdown("### Chart")
                    enriched_data = enrich_datetime_columns(data)
                    plot_result = generate_plot_from_llm(enriched_data, question)
                    if isinstance(plot_result, io.BytesIO):
                        st.image(plot_result, use_container_width=True)
                    else:
                        st.warning(f"Plot generation failed: {plot_result}")
                else:
                    st.warning("Could not generate a plot as no data was found.")

            if show_data:
                if isinstance(data, pd.DataFrame) and not data.empty:
                    st.markdown("### Data Table")
                    st.markdown(f"**Total rows found: {len(data)}**")
                    if len(data) > 50:
                        st.markdown("_Showing top 50 rows_")
                    st.dataframe(data.head(50), use_container_width=True)
            
            if summary:
                st.markdown("### Summary")
                st.info(summary)

        except Exception as e:
            st.error("An unexpected error occurred during execution.")
            st.code(traceback.format_exc())