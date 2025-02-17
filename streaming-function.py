from IPython.display import Markdown, display
from openai import OpenAI
from typing import Optional

def get_streaming_response(
    prompt: str,
    system_prompt: str,
    model: str = "gpt-4o-mini",
    client: Optional[OpenAI] = None
) -> str:
    """
    Generate a streaming response from OpenAI's API and display it as Markdown.
    
    Args:
        prompt (str): The user's prompt/question to be sent to the API
        system_prompt (str): The system prompt that sets the context for the model
        model (str, optional): The OpenAI model to use. Defaults to "gpt-4"
        client (Optional[OpenAI], optional): An OpenAI client instance. If None, creates a new one.
    
    Returns:
        str: The accumulated response from the API
        
    Example:
        >>> system_prompt = "You're an experienced programmer."
        >>> user_prompt = "Explain what this code does: print('Hello, World!')"
        >>> response = get_streaming_response(user_prompt, system_prompt)
    """
    # Initialize OpenAI client if not provided
    if client is None:
        client = OpenAI()
    
    # Create messages array
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Create streaming response
    response_stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )
    
    # Accumulate and display response
    accumulated_response = ""
    
    for chunk in response_stream:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                accumulated_response += content
    
    # Display the markdown response
    display(Markdown(accumulated_response))
    
    return accumulated_response
