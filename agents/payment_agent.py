import pandas as pd
from langchain.schema import SystemMessage, HumanMessage
from agents.shared_llm import llm

def handle_payment_query(user_input, tables=None):
    """Handle payment-related queries with actual data processing."""
    if not tables:
        return "No data available.", None
    
    payments = tables.get("payments")
    orders = tables.get("orders")
    order_items = tables.get("order_items")
    customers = tables.get("customers")
    products = tables.get("products")
    
    if payments is None:
        return "Payment data not available.", None
    
    system_prompt = f"""
You are a data analyst specializing in e-commerce payment analysis. You have access to the following tables:

PAYMENTS table columns: {', '.join(payments.columns)}
ORDERS table columns: {', '.join(orders.columns) if orders is not None else 'Not available'}
ORDER_ITEMS table columns: {', '.join(order_items.columns) if order_items is not None else 'Not available'}
CUSTOMERS table columns: {', '.join(customers.columns) if customers is not None else 'Not available'}
PRODUCTS table columns: {', '.join(products.columns) if products is not None else 'Not available'}

CRITICAL INSTRUCTION: Analyze the user's entire query. If the query contains words like 'plot', 'graph', 'chart', 'visualize', or 'draw', your primary goal is to produce a DataFrame that is aggregated and ready for plotting. For example, for "plot total payment value by payment type", you should group by payment_type and sum the payment_value.

If the query does NOT ask for a plot, then you should return the detailed, un-aggregated data as requested.

Your task is to write Python pandas code to answer the user's question about payments. 
The code should be executable and return a pandas DataFrame or Series.

RULES:
1. Use the provided DataFrame variable names: `payments`, `orders`, `order_items`, `customers`, `products`.
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
            'payments': payments, 'orders': orders, 'order_items': order_items,
            'customers': customers, 'products': products, 'pd': pd,
            'PAYMENTS': payments, 'ORDERS': orders, 'ORDER_ITEMS': order_items,
            'CUSTOMERS': customers, 'PRODUCTS': products
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
        
        answer = f"Found {len(df_result)} records matching your payment query."
        return answer, df_result
        
    except Exception as e:
        print(f"Error in payment query: {e}")
        return f"Error processing payment query: {str(e)}", None
    