import pandas as pd
from langchain.schema import SystemMessage, HumanMessage
from agents.shared_llm import llm

def handle_product_query(user_input, tables=None):
    """Handle product-related queries with actual data processing."""
    if not tables:
        return "No data available.", None
    
    products = tables.get("products")
    order_items = tables.get("order_items")
    orders = tables.get("orders")
    customers = tables.get("customers")
    payments = tables.get("payments")
    
    if products is None:
        return "Product data not available.", None
    
    system_prompt = f"""
You are a data analyst specializing in e-commerce product analysis. You have access to the following tables:

PRODUCTS table columns: {', '.join(products.columns)}
ORDER_ITEMS table columns: {', '.join(order_items.columns) if order_items is not None else 'Not available'}
ORDERS table columns: {', '.join(orders.columns) if orders is not None else 'Not available'}
CUSTOMERS table columns: {', '.join(customers.columns) if customers is not None else 'Not available'}
PAYMENTS table columns: {', '.join(payments.columns) if payments is not None else 'Not available'}

CRITICAL INSTRUCTION: Analyze the user's entire query. If the query contains words like 'plot', 'graph', 'chart', 'visualize', or 'draw', your primary goal is to produce a DataFrame that is aggregated and ready for plotting. For example, for "plot the top 5 product categories by sales", you should calculate sales for each category and show the top 5.

If the query does NOT ask for a plot, then you should return the detailed, un-aggregated data as requested.

Your task is to write Python pandas code to answer the user's question about products. 
The code should be executable and return a pandas DataFrame or Series.

RULES:
1. Use the provided DataFrame variable names: `products`, `order_items`, `orders`, `customers`, `payments`.
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
            'products': products, 'order_items': order_items, 'orders': orders,
            'customers': customers, 'payments': payments, 'pd': pd,
            'PRODUCTS': products, 'ORDER_ITEMS': order_items, 'ORDERS': orders,
            'CUSTOMERS': customers, 'PAYMENTS': payments
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
        
        answer = f"Found {len(df_result)} records matching your product query."
        return answer, df_result
        
    except Exception as e:
        print(f"Error in product query: {e}")
        return f"Error processing product query: {str(e)}", None