# ruff: noqa: E501
"""
Sample documentation for testing the RAG MVP.

These are excerpts from official documentation, used for testing
the ingestion and retrieval pipeline without requiring network access.

In production, these would be fetched from actual documentation URLs.
For the MVP, we use these static samples to validate the pipeline
works correctly.

NOTE: The long lines in the documentation text below are intentional -
they represent realistic documentation content and are excluded from
the line length check.
"""

SAMPLE_DOCUMENTS: list[dict[str, str]] = [
    # LangChain samples
    {
        "source": "langchain",
        "url": "https://python.langchain.com/docs/concepts/agents/",
        "title": "Agents - LangChain",
        "text": """
# Agents

Agents are systems that use LLMs as reasoning engines to determine which actions to take and the inputs necessary to perform the action. After executing actions, the results can be fed back into the LLM to determine whether more actions are needed, or whether it is okay to finish.

## When to use agents

Agents are great when you need flexibility in the execution path of your application. They excel at:
- Complex multi-step reasoning
- Dynamic tool selection based on context
- Handling ambiguous user requests

## Key components

An agent consists of:
1. **LLM**: The reasoning engine that decides what to do
2. **Tools**: Functions the agent can call
3. **Agent Executor**: Orchestrates the agent loop

## Creating an agent

To create a basic agent with tools:

```python
from langchain.agents import create_react_agent
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-70b-versatile")
agent = create_react_agent(llm, tools, prompt)
```
""",
    },
    {
        "source": "langchain",
        "url": "https://python.langchain.com/docs/concepts/tool_calling/",
        "title": "Tool Calling - LangChain",
        "text": """
# Tool Calling

Tool calling allows LLMs to interact with external tools and APIs. The model generates structured output specifying which tool to call and with what arguments.

## How it works

1. Define tools with clear descriptions and schemas
2. Bind tools to the model using `model.bind_tools(tools)`
3. The model decides when to call tools based on the conversation
4. Your application executes the tool and returns results

## Defining tools

Tools can be defined as Python functions with type hints:

```python
from langchain_core.tools import tool

@tool
def search_documentation(query: str) -> str:
    '''Search the documentation for relevant information.'''
    return search_index(query)
```

## Binding tools to models

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-70b-versatile")
llm_with_tools = llm.bind_tools([search_documentation])
```

The model will now include tool calls in its response when appropriate.
""",
    },
    # FastAPI samples
    {
        "source": "fastapi",
        "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
        "title": "First Steps - FastAPI",
        "text": """
# First Steps

The simplest FastAPI file could look like this:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

## Step by step

### Step 1: Import FastAPI

`FastAPI` is a Python class that provides all the functionality for your API.

### Step 2: Create a FastAPI instance

Here the `app` variable will be an instance of the class `FastAPI`.
This will be the main point of interaction to create your API.

### Step 3: Define a path operation

A "path" here refers to the last part of the URL starting from the first `/`.
So, in a URL like `https://example.com/items/foo`, the path would be `/items/foo`.

### Step 4: Define the path operation function

This is our "path operation function":
- `path`: is `/`
- `operation`: is `get`
- `function`: is the function below the decorator

### Running the server

```bash
uvicorn main:app --reload
```
""",
    },
    {
        "source": "fastapi",
        "url": "https://fastapi.tiangolo.com/tutorial/body/",
        "title": "Request Body - FastAPI",
        "text": """
# Request Body

When you need to send data from a client to your API, you send it as a request body.

A request body is data sent by the client to your API. A response body is the data your API sends to the client.

## Use Pydantic models

To declare a request body, you use Pydantic models:

```python
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

app = FastAPI()

@app.post("/items/")
async def create_item(item: Item):
    return item
```

## Benefits

With Pydantic models you get:
- Automatic data validation
- Automatic documentation
- Editor support (completion, type checks)
- Automatic JSON parsing
""",
    },
    # Pydantic samples
    {
        "source": "pydantic",
        "url": "https://docs.pydantic.dev/latest/concepts/models/",
        "title": "Models - Pydantic",
        "text": """
# Models

The primary way to define objects in Pydantic is via models. Models are classes that inherit from `BaseModel`.

## Basic model usage

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
```

You can create instances by passing data as keyword arguments:

```python
user = User(id=1, name="John", email="john@example.com")
```

## Validation

Pydantic validates data automatically:

```python
user = User(id="not an int", name="John", email="john@example.com")
# ValidationError: id must be an integer
```

## Key methods

- `model_validate()`: Validate a dict and return a model instance
- `model_dump()`: Convert a model to a dict
- `model_json_schema()`: Get the JSON Schema for the model
""",
    },
    # Qdrant samples
    {
        "source": "qdrant",
        "url": "https://qdrant.tech/documentation/concepts/collections/",
        "title": "Collections - Qdrant",
        "text": """
# Collections

Collections in Qdrant are named sets of points (vectors with payloads).

## Creating a collection

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(":memory:")  # In-memory for testing

client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
```

## Vector configuration

- `size`: Dimensionality of vectors (must match your embedding model)
- `distance`: Similarity metric (COSINE, EUCLID, or DOT)

## Payloads

Each point can store arbitrary metadata as a payload:

```python
{
    "text": "Document content...",
    "source": "langchain",
    "url": "https://..."
}
```

Payloads can be indexed for efficient filtering.
""",
    },
    {
        "source": "qdrant",
        "url": "https://qdrant.tech/documentation/concepts/search/",
        "title": "Search - Qdrant",
        "text": """
# Search

Qdrant provides several ways to search for similar vectors.

## Basic search

```python
results = client.search(
    collection_name="my_collection",
    query_vector=[0.1, 0.2, ...],  # Your query embedding
    limit=5  # Number of results
)
```

## Filtering

You can filter results by payload:

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="my_collection",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value="langchain")
            )
        ]
    ),
    limit=5
)
```

## Score threshold

You can set a minimum similarity threshold:

```python
results = client.search(
    collection_name="my_collection",
    query_vector=query_embedding,
    score_threshold=0.7,  # Only return results with score >= 0.7
    limit=5
)
```
""",
    },
    # Groq samples
    {
        "source": "groq",
        "url": "https://console.groq.com/docs/quickstart",
        "title": "Quickstart - Groq",
        "text": """
# Groq Quickstart

Groq provides ultra-fast inference for open-source LLMs.

## Installation

```bash
pip install groq
```

## Basic usage

```python
from groq import Groq

client = Groq(api_key="your-api-key")

response = client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

## With LangChain

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    groq_api_key="your-api-key"
)

response = llm.invoke("Hello!")
```

## Available models

- llama-3.1-70b-versatile: Best quality
- llama-3.1-8b-instant: Faster, smaller
- mixtral-8x7b-32768: Good for long contexts
""",
    },
    # Docker samples
    {
        "source": "docker",
        "url": "https://docs.docker.com/get-started/overview/",
        "title": "Docker Overview",
        "text": """
# Docker Overview

Docker is a platform for developing, shipping, and running applications in containers.

## What is a container?

A container is a standard unit of software that packages up code and all its dependencies so the application runs quickly and reliably across environments.

## Key concepts

### Images
A Docker image is a read-only template with instructions for creating a Docker container. Images are built from Dockerfiles.

### Containers
A container is a runnable instance of an image. You can create, start, stop, move, or delete a container.

### Dockerfile
A text file containing instructions to build a Docker image:

```dockerfile
FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### Docker Compose
A tool for defining and running multi-container Docker applications:

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
```
""",
    },
]


def get_sample_documents() -> list[dict[str, str]]:
    """
    Get sample documents for testing.

    Returns
    -------
    list[dict[str, str]]
        List of document dictionaries with text, source, url, title.
    """
    return SAMPLE_DOCUMENTS


def get_documents_by_source(source: str) -> list[dict[str, str]]:
    """
    Get sample documents filtered by source.

    Parameters
    ----------
    source
        Source identifier (e.g., "langchain").

    Returns
    -------
    list[dict[str, str]]
        Filtered document list.
    """
    return [doc for doc in SAMPLE_DOCUMENTS if doc["source"] == source]
