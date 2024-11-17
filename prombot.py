import json
import os
import re
import uuid
from datetime import datetime
from textwrap import dedent
from typing import List, Optional

import pandas as pd
from crewai import Agent, Crew, Task, Process, LLM
from crewai_tools import tool
from dotenv import load_dotenv
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame, MetricSnapshotDataFrame
from pydantic import BaseModel
from tabulate import tabulate
import plotext as plt

from prombot.rag_manager import RAGManager

load_dotenv()

prom_urls = os.environ.get('PROMETHEUS_METRICS_URLS', "").split(',')
prom_server_url = os.environ.get('PROMETHEUS_SERVER_URL', None)
debug_mode = os.environ.get('DEBUG', '0').lower() == '1'
model = os.environ.get('LLM_MODEL', 'openai/gpt-4o')
ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')


# tool calls return a CachedResponse object - to prevent polluting the llm space, we store
# the actual response in the context, and return a CachedResponse object with the id of the response
# the summary contains anything that's actually relevant to the bot to understand.
def inject_context(ctx, raw) -> str:
    # find ids in the raw text and swap them out

    def replacement(match):
        key = match.group(1)  # Extract 'xxxx' from [id:xxxx]
        return "\n" + str(ctx.get(key, ""))

        # Find all blocks matching the pattern [id:xxxx]

    pattern = r"\[id:(\w+)\]"
    return re.sub(pattern, replacement, raw)

    return raw


if prom_server_url is not None:
    prom = PrometheusConnect(
        url=prom_server_url,
        disable_ssl=True,
    )

rag_manager = RAGManager(data_dir="runbooks", persist_dir="db")

print("Loading runbooks")
rag_manager.load_documents()

ctx = dict()


def format_timeseries_tables(df, step):
    # Reset index if timestamp is index
    df = df.reset_index()

    # Get start, end times and create step
    start = df['timestamp'].min()
    end = df['timestamp'].max()

    # Create timestamps column names, but limit to 6 slots
    timestamps = sorted(df['timestamp'].unique())
    max_slots = 10

    if len(timestamps) > max_slots:
        # Keep first 3 and last 3
        selected_timestamps = timestamps[:3] + timestamps[-3:]
        truncated = True
    else:
        selected_timestamps = timestamps
        truncated = False

    # Create t_number mapping
    timestamp_map = {}
    for i, t in enumerate(selected_timestamps):
        if truncated and i == 3:
            # Start counting from n-2 for the last 3 timestamps
            i = len(timestamps) - 3
        timestamp_map[t] = f't{i}'

    # Create row labels from all columns except timestamp and value
    label_cols = [col for col in df.columns if col not in ['timestamp', 'value']]
    df['row_label'] = df[label_cols].apply(
        lambda x: ', '.join([f"{col}={val}" for col, val in zip(label_cols, x) if not pd.isna(val)]),
        axis=1
    )

    # Header info
    header = (
        f"start: {start}\n"
        f"end: {end}\n"
        f"step: {step}\n"
    )

    output = [header]

    # Create a table for each unique row label
    for label in df['row_label'].unique():
        label_df = df[df['row_label'] == label]

        # Create a single row with values for each timestamp
        row_data = []
        headers = []

        # Add first 3 timestamps
        for i, ts in enumerate(selected_timestamps[:3]):
            headers.append(f't{i}')
            value = label_df[label_df['timestamp'] == ts]['value'].values
            row_data.append(value[0] if len(value) > 0 else '')

        if truncated:
            headers.append('...')
            row_data.append('')

            # Add last 3 timestamps
            for i, ts in enumerate(selected_timestamps[-3:]):
                headers.append(f't{len(timestamps) - 3 + i}')
                value = label_df[label_df['timestamp'] == ts]['value'].values
                row_data.append(value[0] if len(value) > 0 else '')

        # Create a single-row DataFrame for this label
        table_df = pd.DataFrame([row_data], columns=headers)

        # Add to output
        output.append(label)
        output.append(tabulate(table_df, headers='keys', tablefmt='psql', floatfmt='.2f', showindex=False))
        output.append("")  # empty line between tables

    return "\n".join(output)


def plot_timeseries_from_dataframe_with_unix_timestamps(df, x_col="timestamp", y_col="value", title=None):
    """
    Plots timeseries data from a DataFrame in the terminal using plotext.
    Aggregates timeseries by grouping all non-x/y columns and uses Unix timestamps for x-axis.

    Args:
        df (pd.DataFrame): The input DataFrame containing timeseries data.
        x_col (str): Column name for the x-axis (timestamp). Default is 'timestamp'.
        y_col (str): Column name for the y-axis (value). Default is 'value'.
        title (str): Title of the plot. Optional.

    Returns:
        str: String representation of the plot.
    """
    # If x_col is the index, reset it to make it a column
    if x_col == df.index.name:
        df = df.reset_index()

    # Ensure column names are stripped of extra spaces
    df = df.rename(columns=str.strip)

    # Validate x_col and y_col existence
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(
            f"Columns '{x_col}' or '{y_col}' not found in DataFrame. Available columns: {list(df.columns)}")

    # Convert the timestamp column to datetime for sorting and Unix timestamp conversion
    df = df.copy()
    df[x_col] = pd.to_datetime(df[x_col])
    df["plotext_time"] = df[x_col].astype(int) // 10 ** 9  # Convert to Unix timestamp (integer seconds)

    # Determine the grouping columns (all except x_col and y_col)
    group_columns = [col for col in df.columns if col not in {x_col, y_col, "plotext_time"}]

    # Group by the other columns to create individual timeseries
    grouped = df.groupby(group_columns)

    # Initialize plotext
    plt.clear_data()

    # Plot each timeseries
    for group, group_df in grouped:
        group_label = " | ".join(map(str, group)) if isinstance(group, tuple) else str(group)
        sorted_group_df = group_df.sort_values(x_col)
        x = sorted_group_df["plotext_time"].tolist()  # Use Unix timestamps for x-axis
        y = sorted_group_df[y_col].tolist()

        if len(x) == len(y):
            plt.plot(x, y, label=group_label)
        else:
            raise ValueError(f"Mismatch in lengths: x ({len(x)}) vs y ({len(y)}). Check data integrity.")

    # Set title and labels
    if title:
        plt.title(title)
    plt.xlabel("Unix Timestamp")
    plt.ylabel(y_col)

    # Build and return the plot as a string
    return plt.build()


def cache_response(summary: str, full_value) -> str:
    id = uuid.uuid4().hex.__str__()
    ctx[id] = full_value
    return "ResponsePointer\nid:{}\n---\nFOR YOUR EYES ONLY:\n{}".format(id, summary)


@tool
def retrieve_context(query: str) -> str:
    """
    Retrieve context from the documents

    :param query: Query to search in the documents
    :return: Context from the documents
    """
    results = rag_manager.search(query, k=3)
    return cache_response(
        "\n".join([doc.page_content for doc in results]),
        results
    )


retrieve_context.cache_function = lambda a, b: False


@tool
def list_metrics(substring: Optional[str] = None) -> str:
    """
    List all the metrics available in Prometheus. You can narrow down the list by providing a substring to match metric names.

    If you can't find any metrics containing the substring, consider listing without a substring!

    :param name_pattern: Optional pattern to filter the metrics by a part of the name
    :return: A summary of available metric names, as well as potential cases where the substring matches a metric label instead.
    """

    if (substring == "None" or substring == ""):
        substring = None

    # TODO figure out how to cache this?
    metrics = prom.all_metrics()
    if substring is not None:
        metrics = [m for m in metrics if substring in m]

    res = "Found metrics:\n" + ", ".join(metrics)

    if substring is not None:
        vals = rag_manager.search("metrics containing labels like: " + substring, k=5)
        found = [
            doc.page_content for doc in vals if doc.page_content.startswith("METRIC_VALUES")
        ]
        if len(found) > 0:
            res += "\n\nFound potential metric values:\n" + "\n".join(found)

    return res


def format_time(time: str) -> datetime:
    if time == "now":
        return datetime.now()
    return datetime.fromisoformat(time)


@tool
def metric_values(metric_name: str) -> str:
    """
    Get the current values of a metric.

    Any time you get the metric values, store the values in memory for future reference - you can ask retrieve_context to get you the "metric values for <metric_name>" document in order to get the correct names of each potential parameter.
    This is useful before you write a query, so that you don't write arbitrary queries that may not return any data.

    :param metric_name: Name of the metric
    :return: List of values. DO NOT modify keys or values of any element of the list, or you will break queries!
    """
    val = prom.get_current_metric_value(metric_name)

    # TODO store only labels?
    rag_manager.add_document(
        "METRIC_VALUES: " + metric_name,
        json.dumps(val),
    )

    metric_df = MetricSnapshotDataFrame(val)
    return cache_response(
        metric_df.head().__str__(),
        tabulate(metric_df, headers='keys', tablefmt='grid'),
    )


metric_values.cache_function = lambda a, b: False


@tool
def save_preferences(preferences: str):
    """
    Save user preferences based on the conversation context.

    Format preferences as a string, one per line. Format in a way you can easily understand. Don't use complex formats such as JSON or YAML.

    :param preferences: User preferences
    """
    rag_manager.add_document(
        "USER_PREFERENCES",
        preferences,
    )


def get_preferences() -> str:
    """
    Retrieve user preferences

    :return: User preferences
    """
    data = rag_manager.get_document_by_identifier("USER_PREFERENCES")
    if data is None:
        return ""

    return json.dumps(data['content'])


@tool
def plot_chart(promql: str, start_time: str, end_time: str, step: str) -> str:
    """
    Plot a range of values for metrics using PromQL in the form of a chart

    :param promql: PromQL query
    :param start_time: (str) specifies the query range start time. ALWAYS use ISO8601 format (eg. 2022-01-01T00:00:00).
    :param end_time: (str) specifies the query range end time. ALWAYS use ISO8601 format (eg. 2022-01-01T00:00:00).
    :param step: (str) Query resolution step width in duration format (eg. 15s, 1m, 1h)
    :return: Plot of the query result
    """

    val = prom.custom_query_range(
        query=promql,
        start_time=format_time(start_time),
        end_time=format_time(end_time),
        step=step,
    )

    df = MetricRangeDataFrame(val)

    return cache_response(
        df.head().__str__(),
        plot_timeseries_from_dataframe_with_unix_timestamps(df)
    )


@tool
def query_range(promql: str, start_time: str, end_time: str, step: str) -> str:
    """
    Query for a range of values for metrics using PromQL

    :param promql: PromQL query
    :param start_time: (str) specifies the query range start time. ALWAYS use ISO8601 format (eg. 2022-01-01T00:00:00).
    :param end_time: (str) specifies the query range end time. ALWAYS use ISO8601 format (eg. 2022-01-01T00:00:00).
    :param step: (str) Query resolution step width in duration format (eg. 15s, 1m, 1h)
    :return: Query result
    """

    val = prom.custom_query_range(
        query=promql,
        start_time=format_time(start_time),
        end_time=format_time(end_time),
        step=step,
    )

    df = MetricRangeDataFrame(val)

    return cache_response(
        df.head().__str__(),
        format_timeseries_tables(df, step)
    )


@tool
def query_value(promql: str) -> str:
    """
    Query values using Prometheus using PromQL

    :param promql: PromQL query
    :return: Query result
    """
    val = prom.custom_query(query=promql)
    metric_df = MetricSnapshotDataFrame(val)
    return cache_response(
        metric_df.head().__str__(),
        tabulate(metric_df, headers='keys', tablefmt='grid'),
    )


query_value.cache_function = lambda a, b: False

tools = [
    retrieve_context,
    save_preferences,
]

if prom_server_url is not None:
    print("Prometheus server is available at ", prom_server_url)
    tools.extend([
        list_metrics,
        metric_values,
        query_value,
        query_range,
        plot_chart,
    ])

if model.startswith('openai/'):
    print("Using OpenAI model: ", model)
    llm = LLM(
        model=model,
        api_key=os.environ['OPENAI_API_KEY'],
    )
elif model.startswith("ollama/"):
    print("Using OLLAMA model: ", model)
    llm = LLM(
        model=model,
        base_url=ollama_base_url
    )
else:
    raise ValueError("Unsupported model")

devops = Agent(
    llm=llm,
    role='Senior DevOps Engineer',
    backstory="""You are a highly skilled Senior DevOps Engineer with deep expertise in Prometheus and PromQL, experienced in designing, deploying, and optimizing complex monitoring systems. Your role is to provide accurate, efficient PromQL queries for diverse metrics requirements, leveraging advanced Prometheus functions and best practices. When given a request, you analyze the metric goals, select appropriate aggregations, and ensure syntax accuracy, delivering PromQL that is optimized for performance and clarity.""",
    goal="""provide accurate, efficient PromQL queries for diverse metrics requirements, leveraging advanced Prometheus functions and best practices.""",
)

chatbot = Agent(
    llm=llm,
    role='Senior DevOps Engineer',
    goal="""provide guidance, write efficient PromQL queries for diverse metrics requirements, leverage advanced Prometheus functions and best practices.""",
    backstory="""You are a highly skilled Senior DevOps Engineer with deep expertise in Prometheus and PromQL, experienced in designing, deploying, and optimizing complex monitoring systems. Your role is to provide accurate, efficient PromQL queries for diverse metrics requirements, leveraging advanced Prometheus functions and best practices. When given a request, you analyze the metric goals, select appropriate aggregations, and ensure syntax accuracy, delivering PromQL that is optimized for performance and clarity.""",
    tools=tools,
    allow_code_execution=True,
)

chat = Task(
    description=dedent("""
        Respond to user queries and perform actions using tools. Always take the user preferences in consideration when responding. 
        
        Preferences may include the user's preferred language, style, specific requirements around what it expects you to do, timeframes to prioritize when writing queries, etc.
        
        ***When you make any query, always include the promql query in the response to the user, so they know what you did***
        
        Don't alter the data you get from Prometheus. Always return the data as-is. Eg if you hae a metric price_assets with labels 'usd' and 'brl, each with a value (eg 100, 200),
        return the data as-is, don't change the labels or values. 
        
        Whenever you receive a ResponsePointer response, always include only the *id* in responses you print to the user, like so: [id:<id here>].
        
        Example:
        User: "what is the current value of price_assets?"
        
        Tool "metric_values('price_assets')" gives you the following response:
        ```
        ResponsePointer
        [id:559569e55d474fd8ba2eb430d828fc55]
        ---
        FOR YOUR EYES ONLY:
           __name__  ...  value
        0  price  ...   23.0
        1  price  ...  175.0
        2  price  ...    5.0
        3  price  ...   20.0
        4  price  ...   19.0
        [5 rows x 9 columns] 
        ```
        
        You should ALWAYS respond to the user with the id tag (eg [id:559569e55d474fd8ba2eb430d828fc55]).
        Notice you can include additional information in the response, but you should NEVER include the "FOR YOUR EYES ONLY" part as-is. 
        
        Here's an example response you can share with the user:
        ```
        <explain what you did. Be sure to include any promql queries you might have used should go here too!>
        
        [id:559569e55d474fd8ba2eb430d828fc55]
        
        <any additional information you might want to include>
        ``` 
        Don't mention anything about results being stored, "FOR YOUR EYES ONLY" blocks, etc - those are details the user isn't aware of.
        
        ATTENTION: never alter the id of the ResponsePointer, or the user won't be able to see the data!
                    
        
        Consider updating the preferences based on the user input and the context of the conversation.
        
        Current time is: {time}
       
        ---
        
        Current chat context:
        {context} 
        
        ---
        
        User preferences:
        {preferences}
        
        ---
        
        User input: 
        {user_input}
        """),
    expected_output="Appropriate responses or actions based on user input. Be brief and concise.",
    agent=chatbot,
)

crew = Crew(
    agents=[chatbot],
    tasks=[chat],
    process=Process.sequential,
    memory=False,
    verbose=debug_mode,
)

context = []

print("Preferences: ", get_preferences())

if __name__ == "__main__":


    print("Ask anything >")
    while True:
        question = input()
        res = crew.kickoff({
            'user_input': question,
            'preferences': get_preferences(),
            'context': '\n'.join(context)[-10000:],
            'time': datetime.now().isoformat(),
        })
        print(
            inject_context(
                ctx,
                res.raw
            )
        )
        context.extend([
            'User: ' + question,
            'Agent: ' + res.raw
        ])
