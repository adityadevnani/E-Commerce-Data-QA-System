import pandas as pd
import re

_stored_dataframes = {}

def store_dataframe(query_id: str, df: pd.DataFrame):
    """Store a dataframe for later retrieval."""
    global _stored_dataframes
    _stored_dataframes[query_id] = df

def get_stored_dataframe(query_id: str) -> pd.DataFrame:
    """Retrieve a stored dataframe."""
    global _stored_dataframes
    return _stored_dataframes.get(query_id)

def make_query_id(prefix: str, query: str) -> str:
    """Generate a stable, clean ID for a query (alphanumeric + underscores)."""
    clean_query = re.sub(r'\W+', '_', query.strip().lower())
    return f"{prefix}_{clean_query}"
