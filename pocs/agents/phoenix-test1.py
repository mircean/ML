import json
import os

import duckdb
import pandas as pd
import tqdm

# import phoenix as px
from phoenix-helper import get_openai_api_key, get_phoenix_endpoint
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.trace import StatusCode
from phoenix.otel import register
from pydantic import BaseModel, Field

# px.active_session()

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://127.0.0.1:6006/"

# initialize the OpenAI client
openai_api_key = get_openai_api_key()
client = OpenAI(api_key=openai_api_key)

MODEL = "gpt-4o-mini"

# define the path to the transactional data
TRANSACTION_DATA_FILE_PATH = "pocs/agents/data/Store_Sales_Price_Elasticity_Promotions_Data.parquet"

# PROJECT_NAME = "tracing-agent"
PROJECT_NAME = "evaluating-agent"
tracer_provider = register(project_name=PROJECT_NAME, endpoint=get_phoenix_endpoint() + "v1/traces")
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer(__name__)


# prompt template for step 2 of tool 1
SQL_GENERATION_PROMPT = """
Generate an SQL query based on a prompt. Do not reply with anything besides the SQL query.
The prompt is: {prompt}

The available columns are: {columns}
The table name is: {table_name}
"""


# code for step 2 of tool 1
def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    """Generate an SQL query based on a prompt"""
    formatted_prompt = SQL_GENERATION_PROMPT.format(prompt=prompt, columns=columns, table_name=table_name)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    assert isinstance(response.choices[0].message.content, str)
    return response.choices[0].message.content


# code for tool 1
@tracer.tool()
def lookup_sales_data(prompt: str) -> str:
    """Implementation of sales data lookup from parquet file using SQL"""
    try:
        # define the table name
        table_name = "sales"

        # step 1: read the parquet file into a DuckDB table
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")

        # step 2: generate the SQL code
        sql_query = generate_sql_query(prompt, df.columns, table_name)
        # clean the response to make sure it only includes the SQL code
        sql_query = sql_query.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "")

        with tracer.start_as_current_span("execute_sql_query", openinference_span_kind="chain") as span:
            span.set_input(sql_query)
            # step 3: execute the SQL query
            result = duckdb.sql(sql_query).df()
            span.set_output(value=str(result))
            span.set_status(StatusCode.OK)

        return result.to_string()
    except Exception as e:
        return f"Error accessing data: {str(e)}"


# example_data = lookup_sales_data("Show me all the sales for store 1320 on November 1st, 2021")
# print(example_data)

# Construct prompt based on analysis type and data subset
DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}
"""


# code for tool 2
@tracer.tool()
def analyze_sales_data(prompt: str, data: str) -> str:
    """Implementation of AI-powered sales data analysis"""
    formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    analysis = response.choices[0].message.content
    return analysis if analysis else "No analysis could be generated"


# print(analyze_sales_data(prompt="what trends do you see in this data",
#                          data=example_data))

# prompt template for step 1 of tool 3
CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
"""


# class defining the response format of step 1 of tool 3
class VisualizationConfig(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate")
    x_axis: str = Field(..., description="Name of the x-axis column")
    y_axis: str = Field(..., description="Name of the y-axis column")
    title: str = Field(..., description="Title of the chart")


# code for step 1 of tool 3
@tracer.chain()
def extract_chart_config(data: str, visualization_goal: str) -> dict:
    """Generate chart visualization configuration

    Args:
        data: String containing the data to visualize
        visualization_goal: Description of what the visualization should show

    Returns:
        Dictionary containing line chart configuration
    """
    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(data=data, visualization_goal=visualization_goal)

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
        response_format=VisualizationConfig,
    )

    try:
        # Extract axis and title info from response
        content = response.choices[0].message.content
        assert isinstance(content, str)
        content = json.loads(content)

        # Return structured chart config
        return {"chart_type": content["chart_type"], "x_axis": content["x_axis"], "y_axis": content["y_axis"], "title": content["title"], "data": data}
    except Exception:
        return {"chart_type": "line", "x_axis": "date", "y_axis": "value", "title": visualization_goal, "data": data}


# prompt template for step 2 of tool 3
CREATE_CHART_PROMPT = """
Write python code to create a chart based on the following configuration.
Only return the code, no other text.
config: {config}
"""


# code for step 2 of tool 3
@tracer.chain()
def create_chart(config: dict) -> str:
    """Create a chart based on the configuration"""
    formatted_prompt = CREATE_CHART_PROMPT.format(config=config)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    code = response.choices[0].message.content
    assert isinstance(code, str)
    code = code.replace("```python", "").replace("```", "")
    code = code.strip()

    return code


# code for tool 3
@tracer.tool()
def generate_visualization(data: str, visualization_goal: str) -> str:
    """Generate a visualization based on the data and goal"""
    config = extract_chart_config(data, visualization_goal)
    code = create_chart(config)
    return code


# code = generate_visualization(example_data, "A bar chart of sales by product SKU. Put the product SKU on the x-axis and the sales on the y-axis.")
# print(code)
# exec(code)

# Define tools/functions that can be called by the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_sales_data",
            "description": "Look up data from Store Sales Price Elasticity Promotions dataset",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string", "description": "The unchanged prompt that the user provided."}},
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sales_data",
            "description": "Analyze sales data to extract insights",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "The lookup_sales_data tool's output."},
                    "prompt": {"type": "string", "description": "The unchanged prompt that the user provided."},
                },
                "required": ["data", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_visualization",
            "description": "Generate Python code to create data visualizations",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "The lookup_sales_data tool's output."},
                    "visualization_goal": {"type": "string", "description": "The goal of the visualization."},
                },
                "required": ["data", "visualization_goal"],
            },
        },
    },
]

# Dictionary mapping function names to their implementations
tool_implementations = {"lookup_sales_data": lookup_sales_data, "analyze_sales_data": analyze_sales_data, "generate_visualization": generate_visualization}


# code for executing the tools returned in the model's response
@tracer.chain()
def handle_tool_calls(tool_calls, messages):
    for tool_call in tool_calls:
        print("Processing tool call:", tool_call.function.name)
        function = tool_implementations[tool_call.function.name]
        function_args = json.loads(tool_call.function.arguments)
        result = function(**function_args)
        messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})

    return messages


SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the Store Sales Price Elasticity Promotions dataset.
"""


def run_agent(messages):
    print("Running agent with messages:", messages)

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    # Check and add system prompt if needed
    if not any(isinstance(message, dict) and message.get("role") == "system" for message in messages):
        system_prompt = {"role": "system", "content": SYSTEM_PROMPT}
        messages.append(system_prompt)

    while True:
        # Router Span
        print("Starting router call span")
        with tracer.start_as_current_span(
            "router_call",
            openinference_span_kind="chain",
        ) as span:
            span.set_input(value=messages)

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
            )
            messages.append(response.choices[0].message)
            tool_calls = response.choices[0].message.tool_calls
            print("Received response with tool calls:", bool(tool_calls))
            span.set_status(StatusCode.OK)

            # if the model decides to call function(s), call handle_tool_calls
            if tool_calls:
                print("Processing tool calls")
                messages = handle_tool_calls(tool_calls, messages)
                span.set_output(value=tool_calls)
            else:
                print("No tool calls, returning final response")
                span.set_output(value=response.choices[0].message.content)
                return response.choices[0].message.content


# result = run_agent("Show me the code for graph of sales by store in Nov 2021, and tell me what trends you see.")
# print(result)
# Markdown(result)


def start_main_span(messages):
    print("Starting main span with messages:", messages)

    with tracer.start_as_current_span("AgentRun", openinference_span_kind="agent") as span:
        span.set_input(value=messages)
        ret = run_agent(messages)
        print("Main span completed with return value:", ret)
        span.set_output(value=ret)
        span.set_status(StatusCode.OK)
        return ret


if PROJECT_NAME == "tracing-agent":
    # run the agent
    result = start_main_span([{"role": "user", "content": "Which stores did the best in 2021?"}])
    print(result)

elif PROJECT_NAME == "evaluating-agent":
    agent_questions = [
        "What was the most popular product SKU?",
        "What was the total revenue across all stores?",
        "Which store had the highest sales volume?",
        "Create a bar chart showing total sales by store",
        "What percentage of items were sold on promotion?",
        "What was the average transaction value?",
    ]

    for question in tqdm.tqdm(agent_questions, desc="Processing questions"):
        ret = start_main_span([{"role": "user", "content": question}])

print("Done")
