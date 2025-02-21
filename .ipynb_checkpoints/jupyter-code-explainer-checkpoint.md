# AI Code Explanation Tool
### Building an AI-Powered Code Understanding Assistant

This notebook demonstrates how to build a tool that uses AI (OpenAI's GPT and Ollama) to explain code snippets. We'll break down the implementation into logical steps and explain each component in detail.

## 1. Setting Up the Environment

First, let's import our required libraries and set up our environment. This section handles all the necessary imports and configurations.

```python
import os
from typing import Generator, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

### Why these imports?
- `typing`: For type hints, making our code more maintainable
- `dataclasses`: For creating clean configuration objects
- `logging`: For proper debugging and monitoring
- `dotenv`: For secure API key management
- `OpenAI`: For interfacing with GPT models
- `IPython.display`: For rich output in Jupyter

## 2. Configuration Management

We'll use a dataclass to manage our configuration. This makes it easy to modify settings and maintain default values.

```python
@dataclass
class Config:
    """Configuration settings for the AI models."""
    openai_model: str = "gpt-4"
    system_prompt: str = """
    You're an experienced programmer, specialized in Python code.
    Please explain the provided code clearly and concisely, including:
    1. Overall purpose
    2. Key components and their functions
    3. Potential improvements or best practices
    4. Example usage where appropriate
    """
```

### Why use a dataclass?
- Clean organization of configuration parameters
- Type checking for configuration values
- Easy modification of default values
- Clear documentation of available settings

## 3. Main Explainer Class

Now let's implement our main class that handles code explanation. We'll break it down into methods with clear responsibilities.

```python
class AICodeExplainer:
    """A class to handle code explanation using AI models."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize with optional custom configuration."""
        self.config = config or Config()
        self._setup_environment()
        self.client = OpenAI()

    def _setup_environment(self) -> None:
        """Set up and validate environment variables."""
        load_dotenv(override=True)
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("No OpenAI API key found in environment variables")
        
        if not api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
        
        api_key = api_key.strip()
        os.environ['OPENAI_API_KEY'] = api_key
        logger.info("Environment setup completed successfully")
```

### Key Components Explained:
1. **Initialization**: 
   - Takes optional configuration
   - Sets up environment
   - Initializes OpenAI client

2. **Environment Setup**:
   - Loads environment variables
   - Validates API key
   - Ensures secure configuration

## 4. Explanation Methods

Here are the core methods that handle getting explanations from the AI model.

```python
class AICodeExplainer:
    # ... (previous methods) ...

    def explain_code(self, code: str, stream: bool = False) -> str:
        """Get an explanation for the provided code."""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": code}
        ]

        try:
            if stream:
                return self._stream_explanation(messages)
            return self._get_explanation(messages)
        except Exception as e:
            logger.error(f"Error getting explanation: {str(e)}")
            raise

    def _get_explanation(self, messages: list) -> str:
        """Get a complete explanation in one response."""
        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages
        )
        return response.choices[0].message.content

    def _stream_explanation(self, messages: list) -> str:
        """Stream the explanation and accumulate the response."""
        response_stream = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            stream=True
        )

        accumulated_response = ""
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                accumulated_response += chunk.choices[0].delta.content

        return accumulated_response

    def display_markdown_explanation(self, explanation: str) -> None:
        """Display the explanation as formatted Markdown."""
        display(Markdown(explanation))
```

### Explanation Methods Breakdown:
1. **explain_code**:
   - Main interface for getting explanations
   - Supports both streaming and non-streaming modes
   - Handles error logging and propagation

2. **_get_explanation**:
   - Gets complete response in one call
   - Suitable for shorter explanations
   - More efficient for quick queries

3. **_stream_explanation**:
   - Streams response in chunks
   - Better for longer explanations
   - Provides immediate feedback

4. **display_markdown_explanation**:
   - Formats output as Markdown
   - Enhances readability in Jupyter

## 5. Usage Example

Let's see how to use our tool with a real code example.

```python
def main():
    """Demonstrate usage of the AICodeExplainer."""
    # Example code to explain
    sample_code = """
    yield from {book.get("author") for book in books if book.get("author")}
    """
    
    try:
        # Initialize explainer
        explainer = AICodeExplainer()
        
        # Get streaming explanation
        print("Getting streaming explanation...")
        streaming_explanation = explainer.explain_code(sample_code, stream=True)
        explainer.display_markdown_explanation(streaming_explanation)
        
        # Get regular explanation
        print("\nGetting regular explanation...")
        regular_explanation = explainer.explain_code(sample_code, stream=False)
        explainer.display_markdown_explanation(regular_explanation)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
```

### Testing the Tool

Now you can test the tool with different code snippets:

```python
# Create an instance
explainer = AICodeExplainer()

# Test with a simple code snippet
code_to_explain = """
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
"""

# Get and display explanation
explanation = explainer.explain_code(code_to_explain)
explainer.display_markdown_explanation(explanation)
```

## Next Steps

You can extend this tool by:
1. Adding support for different programming languages
2. Implementing caching for common explanations
3. Adding code formatting capabilities
4. Integrating with version control systems
5. Adding support for code complexity analysis

Would you like to implement any of these extensions or explore other features?
