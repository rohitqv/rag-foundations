from openai import OpenAI


def _get_client():
    """Return OpenAI client (uses OPENAI_API_KEY env var)."""
    return OpenAI()


def generate_with_single_input(
    prompt: str,
    max_tokens: int = 256,
    model: str = "gpt-4o-mini",
    role: str = "user"
) -> dict:
    """
    Generate text from a language model based on a single input prompt.

    Parameters:
        prompt (str): The input text prompt to send to the language model.
        max_tokens (int): Maximum tokens to generate in the response.
        model (str): The model to use for generation.
        role (str): The role of the message ('user', 'assistant', or 'system').

    Returns:
        dict: A dictionary with 'role' and 'content' keys from the model's response.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": role, "content": prompt}],
        max_tokens=max_tokens,
    )
    message = response.choices[0].message
    return {"role": message.role, "content": message.content}


def generate_with_multiple_input(
    messages: list[dict],
    max_tokens: int = 256,
    model: str = "gpt-4o-mini"
) -> dict:
    """
    Generate text from a language model with multiple input messages (conversation).

    Parameters:
        messages (List[Dict]): A list of dicts with 'role' and 'content' for each message.
        max_tokens (int): Maximum tokens to generate in the response.
        model (str): The model to use for generation.

    Returns:
        dict: A dictionary with 'role' and 'content' keys from the model's response.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    message = response.choices[0].message
    return {"role": message.role, "content": message.content}