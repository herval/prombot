# PromBot

A PromQL assistant.

PromBot helps you write PromQL queries by providing LLM-assisted autocompletion, validating queries and answering questions.

## Operational Memory

PromBot can use any free-form runbooks or documentation to answer questions. To load files, just add them to the `runbooks` directory.


## Usage

Interactive session:

```bash
$ OPENAI_API_KEY=xxx prombot
```

To enable query validation, execution and more, set the `PROMETHEUS_URL` environment variable:

```bash
$ PROMETHEUS_SERVER_URL=http://localhost:9090 OPENAI_API_KEY=xxx prombot
```

This will allow the bot to retrieve the list of available metrics and validate queries against them, as well as execute queries in runtime.

