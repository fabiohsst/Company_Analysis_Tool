# Technical Question Explainer Tool

This notebook implements a tool that uses OpenAI's API and Ollama to provide explanations for technical questions. The tool is designed to be used as a learning resource throughout the course.

## Setup and Imports

```python
# imports
import os
import requests
from dotenv import load_dotenv
from IPython.display import Markdown, display
from openai import OpenAI
```

This section imports the necessary libraries:
- `os`: For environment variable handling
- `requests`: For making HTTP requests
- `dotenv`: For loading environment variables from a .env file
- `IPython.display`: For displaying markdown in Jupyter notebooks
- `openai`: The OpenAI API client

## Environment Setup and API Key Validation

```python
# set up environment
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# validate API key
if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not api_key.startswith("sk-proj-"):
    print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif api_key.strip() != api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("API key found and looks good so far!")
```

This section:
1. Loads environment variables from the .env file
2. Retrieves the OpenAI API key
3. Validates the API key format and presence
4. Provides helpful error messages for common issues

## Constants and Client Initialization

```python
# constants
MODEL_GPT = 'gpt-4o-mini'
openai = OpenAI()
```

This section:
1. Defines the GPT model to be used
2. Initializes the OpenAI client

## System Prompt Definition

```python
system_prompt = """You're experienced programmer, specialized in python code \
I going to give to you a piece of code and you need to explain it to me."""
```

This defines the system prompt that sets the context for the GPT model. The prompt instructs the model to act as an experienced Python programmer who will explain code.

## Sample Code for Analysis

```python
code = """yield from {book.get("author") for book in books if book.get("author")}"""
```

This section defines the sample Python code that will be explained by the model.

## Message Construction

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": code}
]
```

This constructs the message array for the OpenAI API call, including:
1. The system message that sets the context
2. The user message containing the code to be explained

## Basic API Call Example

```python
# To give you a preview -- calling OpenAI with system and user messages:
response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(response.choices[0].message.content)
```

This demonstrates a basic API call to OpenAI:
1. Creates a chat completion request
2. Prints the response content from the first choice

## Streaming Response Implementation

```python
# Create a streaming response generator
response_stream = openai.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": code}
    ],
    stream=True  # Enable streaming
)

accumulated_response = ""

# Read streaming chunks
for chunk in response_stream:
    if chunk.choices:
        content = chunk.choices[0].delta.content
        if content:
            accumulated_response += content

display(Markdown(accumulated_response))
```

This section implements streaming responses:
1. Creates a streaming chat completion request
2. Accumulates the response chunks
3. Displays the final result as markdown

## Placeholder for Llama Implementation

```python
# Get Llama 3.2 to answer
```

This is a placeholder for future implementation of Llama 3.2 model integration.

