import pandas as pd
from langchain.schema import SystemMessage, HumanMessage
from agents.shared_llm import llm

def handle_customer_query(user_input, tables=None):
    """Handle customer-related queries with actual data processing."""
    if not tables:
        return "No data available.", None
    
    customers = tables.get("customers")
    orders = tables.get("orders")
    order_items = tables.get("order_items")
    payments = tables.get("payments")
    products = tables.get("products")
    
    if customers is None:
        return "Customer data not available.", None
    
    
    system_prompt = f"""
You are a data analyst specializing in e-commerce customer analysis. You have access to the following tables:

CUSTOMERS table columns: {', '.join(customers.columns)}
ORDERS table columns: {', '.join(orders.columns) if orders is not None else 'Not available'}
ORDER_ITEMS table columns: {', '.join(order_items.columns) if order_items is not None else 'Not available'}
PAYMENTS table columns: {', '.join(payments.columns) if payments is not None else 'Not available'}
PRODUCTS table columns: {', '.join(products.columns) if products is not None else 'Not available'}

CRITICAL INSTRUCTION: Analyze the user's entire query. If the query contains words like 'plot', 'graph', 'chart', 'visualize', or 'draw', your primary goal is to produce a DataFrame that is aggregated and ready for plotting. For example, for "plot customers by state", you should group by state and count the customers.

If the query does NOT ask for a plot, then you should return the detailed, un-aggregated data as requested.

Your task is to write Python pandas code to answer the user's question about customers. 
The code should be executable and return a pandas DataFrame or Series.

RULES:
1. Use the provided DataFrame variable names: `customers`, `orders`, `order_items`, `payments`, `products`.
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
            'customers': customers, 'orders': orders, 'order_items': order_items,
            'payments': payments, 'products': products, 'pd': pd,
            'CUSTOMERS': customers, 'ORDERS': orders, 'ORDER_ITEMS': order_items,
            'PAYMENTS': payments, 'PRODUCTS': products
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
        
        answer = f"Found {len(df_result)} records matching your customer query."
        return answer, df_result
        
    except Exception as e:
        print(f"Error in customer query: {e}")
        return f"Error processing customer query: {str(e)}", None