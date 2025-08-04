import pandas as pd
import json
import re
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

from agents.shared_llm import llm
from agents.tools_registry import get_tools
from agents.shared_dataframe import get_stored_dataframe

def run_agent_chain(question: str, tables: dict):
    """
    Runs a two-step agent chain:
    1. Intelligently classifies the user's display intent (plot, data, both).
    2. Retrieves the data and generates a summary.
    """

    show_plot_intent = False
    show_data_intent = True

    classification_prompt = f"""
You are an expert intent classifier. Your task is to analyze the user's query and determine which UI components to display based on its semantic meaning.

Analyze the user's query: "{question}"

- If the query's main purpose is to request a **visual plot** (e.g., "plot the orders..."), set "show_plot" to true and "show_data" to false.
- If the query asks for **both data and a plot** (e.g., "show me the orders and also a graph"), set both "show_plot" and "show_data" to true.
- If the query only asks for **data** (e.g., "give me the list of customers"), set "show_data" to true and "show_plot" to false.
- If the query is a **simple question** resulting in a single text answer (e.g., "how many customers?"), set both to false.
- If you are unsure, default to showing the data.

Return ONLY a valid JSON object with two keys: "show_data" (boolean) and "show_plot" (boolean).
"""
    try:
        intent_llm_call = llm.invoke([SystemMessage(content=classification_prompt)])
        cleaned_json = re.search(r'\{.*\}', intent_llm_call.content, re.DOTALL)
        if cleaned_json:
            parsed_intent = json.loads(cleaned_json.group(0))
            show_plot_intent = parsed_intent.get("show_plot", False)
            show_data_intent = parsed_intent.get("show_data", True)
    except Exception as e:
        print(f"Intent classification failed, falling back to default behavior. Error: {e}")
        show_plot_intent = False
        show_data_intent = True

    tools = get_tools(tables)
    data_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data retrieval assistant. Your ONLY job is to use a tool to get the data that answers the user's question. If the user asks for a plot or graph, focus on getting the necessary underlying data for it. Your final answer MUST be only the raw, unmodified JSON string from the tool."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm_with_tools = llm.bind_tools(tools)
    agent = ({"input": lambda x: x["input"], "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),} | data_prompt | llm_with_tools | OpenAIToolsAgentOutputParser())
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
    
    result = agent_executor.invoke({ "input": question })

    final_output_str = result.get("output", "Could not find an answer.")
    df, answer = None, final_output_str

    try:
        parsed_output = json.loads(final_output_str)
        if isinstance(parsed_output, dict):
            query_id = parsed_output.get("query_id")
            answer = parsed_output.get("answer", final_output_str)
            if query_id: df = get_stored_dataframe(query_id)
    except (json.JSONDecodeError, TypeError): pass

    final_summary = ""
    if df is not None:
        summary_prompt_text = f"A user asked: '{question}'\nIn response, a data table with {len(df)} rows was found. Here are the first 3 rows:\n{df.head(3).to_string()}\n\nWrite a concise, 2-3 line summary. IMPORTANT: Start by stating the total number of records found. Then, add a brief insight."
        final_summary = llm.invoke([SystemMessage(content=summary_prompt_text)]).content
    else:
        summary_prompt_text = f"A user asked: '{question}'\nThe direct answer is: {answer}\nRephrase this into a friendly, complete sentence."
        final_summary = llm.invoke([SystemMessage(content=summary_prompt_text)]).content

    return {
        "answer": answer,
        "data": df,
        "summary": final_summary,
        "show_data": show_data_intent and (df is not None),
        "plot": show_plot_intent and (df is not None),
    }