import pandas as pd
from langchain.schema import SystemMessage, HumanMessage
from agents.shared_llm import llm

def handle_logistics_query(user_input, tables=None):
    """Handle logistics and delivery-related queries with actual data processing."""
    if not tables:
        return "No data available.", None
    
    orders = tables.get("orders")
    order_items = tables.get("order_items")
    customers = tables.get("customers")
    products = tables.get("products")
    payments = tables.get("payments")
    
    if orders is None:
        return "Orders data not available for logistics analysis.", None
    
    system_prompt = f"""
You are a data analyst specializing in e-commerce logistics and delivery analysis. You have access to the following tables:

ORDERS table columns: {', '.join(orders.columns)}
ORDER_ITEMS table columns: {', '.join(order_items.columns) if order_items is not None else 'Not available'}
CUSTOMERS table columns: {', '.join(customers.columns) if customers is not None else 'Not available'}
PRODUCTS table columns: {', '.join(products.columns) if products is not None else 'Not available'}
PAYMENTS table columns: {', '.join(payments.columns) if payments is not None else 'Not available'}

CRITICAL INSTRUCTION: Analyze the user's entire query. If the query contains words like 'plot', 'graph', 'chart', 'visualize', or 'draw', your primary goal is to produce a DataFrame that is aggregated and ready for plotting. For example, for "plot average delivery time per state", you should calculate this aggregation.

If the query does NOT ask for a plot, then you should return the detailed, un-aggregated data as requested.

Your task is to write Python pandas code to answer the user's question about logistics, delivery, shipping, or fulfillment. 
The code should be executable and return a pandas DataFrame or Series.

RULES:
1. Use the provided DataFrame variable names: `orders`, `order_items`, `customers`, `products`, `payments`.
2. **DO NOT** use a variable named `df` in the code you write.
3. Your code **MUST** end by assigning the final result to a variable named `result`.
4. Return only the code, no explanations.
5. Use merge/join operations when you need data from multiple tables.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {user_input}")
    ]
    
    try:
        response = llm.invoke(messages)
        code = response.content.strip().replace("```python", "").replace("```", "").strip()
        
        local_vars = {
            'orders': orders, 'order_items': order_items, 'customers': customers,
            'products': products, 'payments': payments, 'pd': pd,
            'ORDERS': orders, 'ORDER_ITEMS': order_items, 'CUSTOMERS': customers,
            'PRODUCTS': products, 'PAYMENTS': payments
        }
        
        exec(code, {'pd': pd}, local_vars)
        result = local_vars.get('result')
        
        if result is None:
            return "No result generated from the query.", None
        
        if isinstance(result, pd.Series):
            df_result = result.reset_index()
        elif isinstance(result, pd.DataFrame):
            df_result = result
        else:
            return str(result), None
        
        if len(df_result.columns) == 2:
            df_result.columns = ['category', 'value']
        
        answer = f"Found {len(df_result)} records matching your logistics query."
        return answer, df_result
        
    except Exception as e:
        print(f"Error in logistics query: {e}")
        return f"Error processing logistics query: {str(e)}", None