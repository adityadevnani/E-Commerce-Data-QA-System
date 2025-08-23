# E-Commerce Data QA System

An intelligent, conversational AI agent built with Streamlit and LangChain that allows users to ask questions about an e-commerce dataset in natural language. The agent can provide direct answers, display data tables, and generate dynamic plots and summaries.

## Dataset Source

This project uses a publicly available dataset from Kaggle:  
**[E-Commerce Order Dataset by Bytadit](https://www.kaggle.com/datasets/bytadit/ecommerce-order-dataset)**  
It includes multiple CSV files related to orders, order_items, customers, payments and products.


## Key Features

- **Natural Language Querying:** Ask complex questions about customers, orders, products, and payments in plain English.
- **Data Table Display:** View detailed, raw data tables for your queries.
- **Dynamic Chart Generation:** Request plots and graphs to visualize data. The system automatically prepares the data and generates charts.
- **AI-Generated Summaries:** Get concise, multi-line summaries for any data or plot you request.
- **Intelligent UI:** The interface intelligently decides what to show (a simple answer, a table, a plot, or a combination) based on the user's query.
- **Conversational Memory:** The agent remembers the context of the conversation, allowing for natural follow-up questions.

## How It Works

The application uses a powerful multi-agent architecture:

* **The User Interface (`app.py`):** The Streamlit frontend that you interact with. It manages the chat history and displays the final results.
* **The Manager (`graph_agent.py`):** The main "brain" of the application. It receives your question, decides which specialist is needed, and creates the final summary.
* **The Specialists (`order_agent.py`, `customer_agent.py`, `payment_agent.py`, `product_agent.py`, `logistics_agent.py`):** These are expert AI agents, each trained to handle a specific dataset (orders, order_items, customers, payment and products.). Their only job is to write and execute Python pandas code to find the exact data you need.
* **The Artist (`plot_agent.py`):** A specialist AI that takes the data prepared by other agents and writes Python matplotlib/seaborn code to draw the graphs and charts.

## Project Structure

ecommerce_agent_project/
│
├── agents/
│   ├── init.py
│   ├── customer_agent.py
│   ├── graph_agent.py
│   ├── logistics_agent.py
│   ├── order_agent.py
│   ├── payment_agent.py
│   ├── plot_agent.py
│   ├── product_agent.py
│   ├── shared_dataframe.py
│   ├── shared_llm.py
│   └── tools_registry.py
│
├── data/
│   ├── customers.csv
│   ├── order_items.csv
│   ├── orders.csv
│   ├── payments.csv
│   └── products.csv
│
├── venv/
│
├── .env
├── .gitignore
├── app.py
└── requirements.txt


## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd ecommerce_agent_project
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create Environment File:**
    Create a file named `.env` in the main project directory and add your Azure OpenAI API credentials.
    ```
    AZURE_OPENAI_API_KEY="your_api_key"
    AZURE_OPENAI_ENDPOINT="your_azure_endpoint"
    AZURE_OPENAI_DEPLOYMENT_NAME="your_deployment_name"
    AZURE_OPENAI_API_VERSION="your_api_version"
    ```

5.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## Example Usage

You can ask a variety of questions, such as:

* **Simple Data Request:** `give the orders that were canceled in 2018`
* **Simple Answer:** `how many unique customers are there`
* **Plot-Only Request:** `plot the orders delivered in 2017`
* **Plot and Data Request:** `show average product price per category and also show the graph`
