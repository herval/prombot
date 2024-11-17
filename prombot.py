import json
import os
from datetime import datetime
from textwrap import dedent
from typing import List, Optional

from crewai import Agent, Crew, Task, Process, LLM
from crewai_tools import tool
from dotenv import load_dotenv
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame, MetricSnapshotDataFrame
from tabulate import tabulate

from prombot.rag_manager import RAGManager

load_dotenv()

prom_urls = os.environ.get('PROMETHEUS_METRICS_URLS', "").split(',')
prom_server_url = os.environ.get('PROMETHEUS_SERVER_URL', None)
debug_mode = os.environ.get('DEBUG', '0').lower() == '1'
model = os.environ.get('LLM_MODEL', 'openai/gpt-4o')
ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')

if __name__ == "__main__":
    if prom_server_url is not None:
        prom = PrometheusConnect(
            url=prom_server_url,
            disable_ssl=True,
        )

    rag_manager = RAGManager(data_dir="runbooks", persist_dir="db")

    print("Loading runbooks")
    rag_manager.load_documents()

    @tool
    def retrieve_context(query: str) -> str:
        """
        Retrieve context from the documents

        :param query: Query to search in the documents
        :return: Context from the documents
        """
        results = rag_manager.search(query, k=3)
        return "\n".join([doc.page_content for doc in results])

    retrieve_context.cache_function = lambda a, b: False

    @tool
    def list_metrics(substring: Optional[str] = None) -> List[str]:
        """
        List all the metrics available in Prometheus. You can narrow down the list by providing a substring to match metric names.

        If you can't find any metrics containing the substring, consider listing without a substring!

        :param name_pattern: Optional pattern to filter the metrics by a part of the name
        :return: List of metric names
        """

        if(substring == "None"):
            substring = None

        metrics = prom.all_metrics()
        if substring is not None:
            metrics = [m for m in metrics if substring in m]

        return metrics


    @tool
    def metric_values(metric_name: str):
        """
        Get the current values of a metric.

        Any time you get the metric values, store the values in memory for future reference - you can ask retrieve_context to get you the "metric values for <metric_name>" document in order to get the correct names of each potential parameter.
        This is useful before you write a query, so that you don't write arbitrary queries that may not return any data.

        :param metric_name: Name of the metric
        :return: List of values. DO NOT modify keys or values of any element of the list, or you will break queries!
        """
        val = prom.get_current_metric_value(metric_name)

        rag_manager.add_document(
            "metric values for " + metric_name,
            json.dumps(val),
        )

        metric_df = MetricSnapshotDataFrame(val)
        return tabulate(metric_df, headers='keys', tablefmt='grid')


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
    def query_range(promql: str, start_time: str, end_time: str, step: str) -> str:
        """
        Query for a range of values for metrics using PromQL

        :param promql: PromQL query
        :param start_time: (str) specifies the query range start time. ALWAYS use ISO8601 format (eg. 2022-01-01T00:00:00). Use "now" for current time.
        :param end_time: (str) specifies the query range end time. ALWAYS use ISO8601 format (eg. 2022-01-01T00:00:00). Use "now" for current time.
        :param step: (str) Query resolution step width in duration format (eg. 15s, 1m, 1h)
        :return: Query result
        """

        def format_time(time: str) -> datetime:
            if time == "now":
                return datetime.now()
            return datetime.fromisoformat(time)

        def format_timeseries_tables(df):
            # Reset index to make timestamp a regular column
            df = df.reset_index()

            # Get all columns except timestamp and value
            group_columns = [col for col in df.columns if col not in ['timestamp', 'value']]

            # Create a pivot table
            # First create the aggregation column if it doesn't exist
            if 'aggregation' not in df.columns:
                df['aggregation'] = df[group_columns].apply(lambda x: ', '.join([f"{col}={val}" for col, val in zip(group_columns, x)]), axis=1)

            # Pivot (using actual timestamps)
            pivot_df = df.pivot(index='aggregation', columns='timestamp', values='value')

            # Reset index to make 'aggregation' a regular column
            pivot_df.reset_index(inplace=True)

            return tabulate(pivot_df, headers='keys', tablefmt='psql')

        val = prom.custom_query_range(
            query=promql,
            start_time=format_time(start_time),
            end_time=format_time(end_time),
            step=step,
        )

        df = MetricRangeDataFrame(val)

        return format_timeseries_tables(df)





    @tool
    def query_value(promql: str):
        """
        Query discrete values using Prometheus using PromQL

        :param promql: PromQL query
        :return: Query result
        """
        val = prom.custom_query(query=promql)
        return val
        # metric_df = MetricSnapshotDataFrame(val)
        # return tabulate(metric_df, headers='keys', tablefmt='grid')


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
        
        Whenever you get data from Prometheus, always include the data in a well-formatted table as part of the response too.
        
        Don't alter the data you get from Prometheus. Always return the data as-is. Eg if you hae a metric price_assets with labels 'usd' and 'brl, each with a value (eg 100, 200),
        return the data as-is, don't change the labels or values. This is what the table could look like:
        
        +----------------------------------+----------------------------------+
        | price_assets{{currency=\"usd\"}} | price_assets{{currency=\"brl\"}} |
        |----------------------------------|----------------------------------|
        | 100                              | 200                              |
        +----------------------------------+----------------------------------+
        
        or:
        
        +----------|--------------+
        | currency | value        |
        |----------|--------------|
        | usd      | 100          |
        | brl      | 200          |
        +----------|--------------+
        
        You can also output the data in a different format, as long as it's clear and easy to understand. This is an example of a range of values over time:
        
        __name__=temperature_celsius
        +----------------------+---------------------------------------------------+
        |  timestamp           | building_type=commercial | building_type=cultural |
        |----------------------+---------------------------------------------------|
        |  2024-11-16 22:44:50 |                       19 |                     24 |
        |  2024-11-16 22:45:50 |                       24 |                     24 |
        +----------------------+--------------------------+------------------------+

        
        
        
        If your response includes too many metrics, consider summarizing the data or providing a high-level overview, instead of outputting the entire list!

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

    print("Ask anything >")
    while True:
        question = input()
        res = crew.kickoff({
            'user_input': question,
            'preferences': get_preferences(),
            'context': '\n'.join(context)[-10000:],
            'time': datetime.now().isoformat(),
        })
        print(res.raw)
        context.extend([
            'User: ' + question,
            'Agent: ' + res.raw
        ])
