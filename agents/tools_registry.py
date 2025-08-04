import pandas as pd
import json
from langchain_core.tools import tool

from agents.customer_agent import handle_customer_query
from agents.order_agent import handle_order_query
from agents.payment_agent import handle_payment_query
from agents.product_agent import handle_product_query
from agents.logistics_agent import handle_logistics_query
from agents.shared_dataframe import store_dataframe, make_query_id

def get_tools(tables: dict):
    """Return a list of data-retrieval tools for the agent."""

    @tool
    def customer_query_tool(query: str) -> str:
        """
        Handle customer-related queries including demographics, locations, behavior analysis.
        Use for questions about customers, states, cities, or customer analysis.
        """
        answer, df = handle_customer_query(query, tables)
        query_id = None
        if isinstance(df, pd.DataFrame) and not df.empty:
            query_id = make_query_id("customer", query)
            store_dataframe(query_id, df)
        return json.dumps({"answer": answer, "query_id": query_id})

    @tool
    def order_query_tool(query: str) -> str:
        """
        Handle order-related queries including status, trends, revenue, and counts.
        Use for questions about orders, status, dates, values, or revenue.
        """
        answer, df = handle_order_query(query, tables)
        query_id = None
        if isinstance(df, pd.DataFrame) and not df.empty:
            query_id = make_query_id("order", query)
            store_dataframe(query_id, df)
        return json.dumps({"answer": answer, "query_id": query_id})

    @tool
    def payment_query_tool(query: str) -> str:
        """
        Handle payment-related queries including methods, values, and analysis.
        Use for questions about payments, types, amounts, or payment trends.
        """
        answer, df = handle_payment_query(query, tables)
        query_id = None
        if isinstance(df, pd.DataFrame) and not df.empty:
            query_id = make_query_id("payment", query)
            store_dataframe(query_id, df)
        return json.dumps({"answer": answer, "query_id": query_id})

    @tool
    def product_query_tool(query: str) -> str:
        """
        Handle product-related queries including categories, analysis, and popular products.
        Use for questions about products, categories, sales, or product performance.
        """
        answer, df = handle_product_query(query, tables)
        query_id = None
        if isinstance(df, pd.DataFrame) and not df.empty:
            query_id = make_query_id("product", query)
            store_dataframe(query_id, df)
        return json.dumps({"answer": answer, "query_id": query_id})

    @tool
    def logistics_query_tool(query: str) -> str:
        """

        Handle logistics and delivery queries including delivery times and fulfillment.
        Use for questions about delivery, shipping, logistics, or order fulfillment.
        """
        answer, df = handle_logistics_query(query, tables)
        query_id = None
        if isinstance(df, pd.DataFrame) and not df.empty:
            query_id = make_query_id("logistics", query)
            store_dataframe(query_id, df)
        return json.dumps({"answer": answer, "query_id": query_id})

    return [
        customer_query_tool,
        order_query_tool,
        payment_query_tool,
        product_query_tool,
        logistics_query_tool,
    ]