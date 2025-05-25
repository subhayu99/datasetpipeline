import os
from time import sleep
from typing import Callable, Optional, Dict, Any
import openai
from .utils import get_openai_rate_limit_seconds
from .logger import LOGGER


# Configuration for OpenAI API | TODO: Need to make it configurable in the future
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4o",
    "max_tokens": 4096,
    "timeout": 30,
}

def get_openai_client():
    """
    Initialize and return OpenAI client with proper configuration.
    
    Returns:
        openai.OpenAI: Configured OpenAI client
        
    Raises:
        ValueError: If API key is not found
    """
    api_key = OPENAI_CONFIG["api_key"]
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        )
    
    return openai.OpenAI(
        api_key=api_key,
        timeout=OPENAI_CONFIG["timeout"]
    )

def call_openai_api(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    n: int = 1,
    logger: Callable[[str, str], None] = LOGGER.log,
    max_retries: int = 3,
    **kwargs
) -> Optional[Any]:
    """
    Call the OpenAI API to generate text based on the given messages.
    
    Args:
        messages (list[dict[str, str]]): A list of dictionaries representing the conversation messages.
        model (str, optional): The model to use. Defaults to configured model.
        temperature (float, optional): The temperature parameter for text generation. Defaults to 0.7.
        max_tokens (int, optional): Maximum tokens to generate. Defaults to configured max_tokens.
        n (int, optional): The number of text generation samples to generate. Defaults to 1.
        logger (Callable[[str, str], None], optional): A logger function. Defaults to LOGGER.log.
        max_retries (int, optional): Maximum number of retries for rate limiting. Defaults to 3.
        **kwargs: Additional parameters to pass to the OpenAI API.
    
    Returns:
        The response from the OpenAI API, or None if filtered/failed.
        
    Raises:
        Exception: If the OpenAI API call fails after all retries.
        
    Example:
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris"},
        ]
        response = call_openai_api(messages)
        if response:
            print(response.choices[0].message.content)
    """
    
    # Use provided parameters or fall back to defaults
    model = model or OPENAI_CONFIG["model"]
    max_tokens = max_tokens or OPENAI_CONFIG["max_tokens"]
    
    client = get_openai_client()
    
    # Prepare API parameters
    api_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,
        **kwargs
    }
    
    for attempt in range(max_retries + 1):
        try:
            logger("DEBUG", f"Making OpenAI API call (attempt {attempt + 1}/{max_retries + 1})")
            response = client.chat.completions.create(**api_params)
            logger("DEBUG", "OpenAI API call successful")
            return response
            
        except openai.APIError as e:
            # Handle content filtering
            if "response was filtered" in str(e) or "content_filter" in str(e).lower():
                logger("DEBUG", "Prompt was filtered by OpenAI content policy.")
                return None
                
            # Handle rate limiting
            if e.status_code == 429:  # Rate limit error
                if attempt < max_retries:
                    rate_limit_seconds = get_openai_rate_limit_seconds(str(e))
                    seconds_to_sleep = rate_limit_seconds + 10  # Add buffer
                    logger("DEBUG", f"Rate limited: {e}")
                    logger("INFO", f"OpenAI API rate limited, sleeping for {seconds_to_sleep} seconds...")
                    sleep(seconds_to_sleep)
                    continue
                else:
                    logger("ERROR", f"Max retries exceeded for rate limiting: {e}")
                    raise
            
            # Handle other API errors
            logger("ERROR", f"OpenAI API error: {e}")
            if attempt < max_retries:
                sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                raise
                
        except Exception as e:
            logger("ERROR", f"Unexpected error: {e}")
            if attempt < max_retries:
                sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                raise

def call_openai_api_simple(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Optional[str]:
    """
    Simplified function to call OpenAI API with a single prompt.
    
    Args:
        prompt (str): The prompt to send to the API.
        model (str, optional): The model to use.
        temperature (float, optional): The temperature parameter.
        max_tokens (int, optional): Maximum tokens to generate.
        
    Returns:
        str: The generated text response, or None if failed.
    """
    messages = [{"role": "user", "content": prompt}]
    response = call_openai_api(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1
    )
    
    if response and response.choices:
        return response.choices[0].message.content
    return None

# Configuration helper functions
def set_openai_config(**kwargs):
    """
    Update OpenAI configuration.
    
    Args:
        **kwargs: Configuration parameters to update (api_key, model, max_tokens, timeout)
    """
    global OPENAI_CONFIG
    for key, value in kwargs.items():
        if key in OPENAI_CONFIG:
            OPENAI_CONFIG[key] = value

def get_openai_config() -> Dict[str, Any]:
    """
    Get current OpenAI configuration.
    
    Returns:
        dict: Current configuration
    """
    return OPENAI_CONFIG.copy()

if __name__ == "__main__":
    # Example usage
    try:
        response = call_openai_api_simple(
            prompt="Only give the answer. 1+2=",
            temperature=0,
        )
        if response:
            print(response)
        else:
            print("No response received")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your OPENAI_API_KEY environment variable")