#!/usr/bin/env python3
"""
tool_calling_agent.py

Extended Tool-Calling Agent with Debug Logging and Improved Parsing.
"""

import os
import json
import re
import wikipedia
from openai import OpenAI

# 1. GLOBAL DEFINITIONS AND SETUP
# --------------------------------------------------------------------
# Here, we set up a basic OpenAI client pointing to a custom base URL,
# with an API key specified as 'none' (e.g., for internal testing).
client = OpenAI(base_url='http://10.246.250.226:12300/v1', api_key='none')

# 2. TOOL (FUNCTION) SCHEMA DEFINITIONS
# --------------------------------------------------------------------
# The `functions` list defines the schema for each tool we expose to the LLM.
# Each entry details the function name, what it does, and the JSON schema
# for its parameters. This allows the LLM to call the tools properly.
functions = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for pages related to a query and return a list of page titles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search term for Wikipedia."}
                },
                "required": ["query"],
                "additionalProperties": False
            }
        },
        "strict": True
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_wikipedia_page",
            "description": "Fetches Wikipedia page summary for a given title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The title of the Wikipedia page."}
                },
                "required": ["title"],
                "additionalProperties": False
            }
        },
        "strict": True
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_assist",
            "description": (
                "Assists with Wikipedia queries, including suggesting corrections (mode='suggest') "
                "and finding pages near coordinates (mode='geosearch')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["suggest", "geosearch"], "description": "The type of assistance needed."},
                    "query": {"type": "string", "description": "The query for suggest mode."},
                    "latitude": {"type": "number", "description": "Latitude for geosearch mode."},
                    "longitude": {"type": "number", "description": "Longitude for geosearch mode."}
                },
                "required": ["mode"],
                "additionalProperties": False
            }
        },
        "strict": True
    }
]

# 3. TOOL FUNCTION IMPLEMENTATIONS
# --------------------------------------------------------------------
# The actual Python implementations for the above-defined tools. The agent
# may call these functions to fulfill the user's query.

def search_wikipedia(query: str):
    """
    Searches Wikipedia for the given query.

    Args:
        query (str): The search term used in Wikipedia.

    Returns:
        list: A list of up to 5 Wikipedia page titles that match the query.
              In case of an error, returns a list containing an error message.

    Error Handling:
        Catches general exceptions and returns a list with the error message.
    """
    print(f"Agent Tools: Calling search_wikipedia('{query}')...")
    try:
        results = wikipedia.search(query, results=5)
        print(f"Agent Tools: search_wikipedia('{query}') -> {results}")
        return results
    except Exception as e:
        error_msg = f"Error in search_wikipedia: {str(e)}"
        print(error_msg)
        return [error_msg]

def fetch_wikipedia_page(title: str):
    """
    Fetches a short (2-sentence) summary of a Wikipedia page by title.

    Args:
        title (str): The exact title of the Wikipedia page.

    Returns:
        dict: A dictionary containing the 'title' and 'summary' keys if successful.
              If the page is not found, returns a dict with 'error' key.
              If there's a disambiguation error, returns a dict with 'error' and 'options'.

    Error Handling:
        - PageError: returns a dict with 'error' stating that the page wasn't found.
        - DisambiguationError: returns a dict with the disambiguation options.
        - General exceptions are caught and returned as an error message.
    """
    print(f"Agent Tools: Calling fetch_wikipedia_page('{title}')...")
    try:
        summary = wikipedia.summary(title, sentences=2)
        result = {"title": title, "summary": summary}
        print(f"Agent Tools: fetch_wikipedia_page('{title}') -> {result}")
        return result
    except wikipedia.exceptions.PageError:
        error = {"error": f"No Wikipedia page found for '{title}'."}
        print(f"Agent Tools: {error}")
        return error
    except wikipedia.exceptions.DisambiguationError as e:
        error = {"error": "Disambiguation error", "options": e.options}
        print(f"Agent Tools: {error}")
        return error
    except Exception as e:
        error = {"error": f"Error: {str(e)}"}
        print(f"Agent Tools: {error}")
        return error

def wikipedia_assist(mode: str, **kwargs):
    """
    Assists with specialized Wikipedia queries:
      - mode='suggest': tries to suggest spelling corrections.
      - mode='geosearch': finds pages near given coordinates.

    Args:
        mode (str): Either 'suggest' or 'geosearch'.
        kwargs: Additional parameters such as 'query', 'latitude', or 'longitude'
                depending on the mode.

    Returns:
        Varies: 
            - For 'suggest': a suggestion if found; otherwise a best-match title or error message.
            - For 'geosearch': a dict containing the latitude, longitude, and a list of nearby pages.

    Error Handling:
        Catches all exceptions and returns an error message string.
        Validates presence of lat/lon for 'geosearch' mode.
    """
    print(f"Agent Tools: Calling wikipedia_assist(mode='{mode}', kwargs={kwargs})...")
    try:
        if mode == "suggest":
            query = kwargs.get("query", "")
            suggestion = wikipedia.suggest(query)
            print(f"Agent Tools: suggest('{query}') -> {suggestion}")
            if suggestion and suggestion.lower() != query.lower():
                return suggestion
            else:
                fallback_results = wikipedia.search(query, results=5)
                return fallback_results[0] if fallback_results else "No suggestion found."
        elif mode == "geosearch":
            lat = kwargs.get("latitude")
            lon = kwargs.get("longitude")
            if lat is None or lon is None:
                return "Error: No coordinates provided."
            geo_results = wikipedia.geosearch(lat, lon, results=20)
            return {"latitude": lat, "longitude": lon, "results": geo_results}
        else:
            return f"Error: Invalid mode '{mode}'"
    except Exception as e:
        error_msg = f"Error in wikipedia_assist: {str(e)}"
        print(error_msg)
        return error_msg

# 4. CONVERSATION LOOP WITH DEBUG LOGGING AND IMPROVED PARSING
# --------------------------------------------------------------------
# The main logic that interacts with the LLM. We repeatedly query the LLM
# until it provides a final user-facing answer or until we reach a maximum
# iteration count. The LLM can request function calls, which we handle here.

def conversation_loop(system_prompt: str, user_query: str):
    """
    Handles a multi-turn conversation with the LLM, allowing it to call functions
    defined in the 'functions' list until a final answer is generated or the
    maximum iteration count is reached.

    Args:
        system_prompt (str): The system-level prompt to guide the LLM's behavior.
        user_query (str): The user's question or command.

    Returns:
        str: The final answer from the LLM, or a default message if no answer is produced.
    
    Error Handling:
        If the LLM attempts to call the same function with the same arguments
        repeatedly, subsequent calls are skipped. If the conversation exceeds
        the maximum iterations, returns "(No final answer)".
    """
    print("=======================================================")
    print(f"Query: {user_query}")
    print("Thought Process: The agent will parse the query and decide if/when to call a Wikipedia function.")
    print("Agent Thought (Step 1): Initializing conversation and calling LLM for the first time...")

    # Initialize conversation with system and user messages.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    max_iterations = 10  # Increased to allow more function calls for complex queries
    iteration = 0
    function_calls_made = set()  # Track function calls to prevent redundancy
    processed_pages = set()      # Track pages already processed for summaries

    # Continue iterating until we reach a final answer or hit max_iterations.
    while iteration < max_iterations:
        iteration += 1
        print(f"\nAgent Thought (Step {iteration+1}): Sending messages to LLM to get next step...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=functions,
            function_call="auto"
        )
        msg = response.choices[0].message

        # The LLM's textual content (if any).
        print(f"LLM Thought (Step {iteration+1}): {msg.content if msg.content else '(No direct text)'}")
        messages.append({"role": "assistant", "content": msg.content or ""})

        # First, check for a structured function call attribute.
        if hasattr(msg, "function_call") and msg.function_call:
            func_name = msg.function_call["name"]
            print(f"Agent Thought (Step {iteration+1}): LLM wants to call function '{func_name}' via function_call...")
            try:
                # Parse the arguments from JSON if present.
                args = json.loads(msg.function_call["arguments"])
            except json.JSONDecodeError:
                args = {}
                print("Agent Thought: Could not parse function arguments as JSON.")
        else:
            # If not present, attempt to parse a function call from the text content using regex.
            pattern = r"(\w+)\((.*?)\)"
            match = re.search(pattern, msg.content or "")
            if match:
                func_name = match.group(1)
                args_str = match.group(2)
                args = {}
                # Split arguments on commas.
                args_list = [arg.strip() for arg in args_str.split(',')]
                if func_name == "wikipedia_assist":
                    # If the first argument isn't key=value, assume it's the mode.
                    if args_list:
                        first_arg = args_list[0]
                        if '=' not in first_arg:
                            mode = first_arg.strip().strip("'").strip('"')
                            args['mode'] = mode
                            args_list = args_list[1:]
                elif func_name in ["search_wikipedia", "fetch_wikipedia_page"]:
                    # For these, there's typically just one main argument: 'query' or 'title'.
                    expected_param = "query" if func_name == "search_wikipedia" else "title"
                    if args_list and '=' not in args_list[0]:
                        value = args_list[0].strip().strip("'").strip('"')
                        args[expected_param] = value
                        args_list = args_list[1:]
                
                # Parse additional 'key=value' style arguments.
                for arg in args_list:
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Handle string values in quotes:
                        if value.startswith('"') and value.endswith('"'):
                            args[key] = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            args[key] = value[1:-1]
                        else:
                            # Attempt to parse numeric arguments into floats.
                            try:
                                args[key] = float(value)
                            except ValueError:
                                args[key] = value
                print(f"Agent Thought (Step {iteration+1}): Parsed function call from text: '{func_name}' with args {args}")
            else:
                # No function call detected, treat it as the final answer.
                final_answer = msg.content.strip() or "(No response)"
                print("\nAgent Final Answer:", final_answer)
                print("=======================================================\n")
                return final_answer

        # Prevent repeated calls to the same function with identical arguments.
        func_call_key = f"{func_name}{json.dumps(args, sort_keys=True)}"
        if func_call_key in function_calls_made:
            print(f"Agent Thought (Step {iteration+1}): Function '{func_name}' with args {args} already called. Skipping...")
            messages.append({
                "role": "function",
                "name": func_name,
                "content": json.dumps({"output": "Function already called, please proceed."})
            })
            continue
        function_calls_made.add(func_call_key)

        # Execute the function requested by the LLM.
        if func_name == "search_wikipedia":
            result = search_wikipedia(args.get("query", ""))
        elif func_name == "fetch_wikipedia_page":
            title = args.get("title", "")
            # Avoid fetching the same page multiple times.
            if title in processed_pages:
                print(f"Agent Thought (Step {iteration+1}): Page '{title}' already processed. Skipping...")
                result = "Page already processed"
            else:
                result = fetch_wikipedia_page(title)
                processed_pages.add(title)
        elif func_name == "wikipedia_assist":
            mode = args.get("mode")
            other_args = {k: v for k, v in args.items() if k != "mode"}
            result = wikipedia_assist(mode, **other_args)
        else:
            result = "Unknown function"
            print(f"Agent Thought (Step {iteration+1}): Unknown function '{func_name}' requested.")

        # Format the function result as JSON and add it back into the conversation.
        function_response = json.dumps({"output": result})
        print(f"Agent Tools (Step {iteration+1}): Returning function result to LLM -> {function_response}")
        messages.append({
            "role": "function",
            "name": func_name,
            "content": function_response
        })

    # If we reach this point, we have not arrived at a final answer within the allowed iterations.
    print("Max iterations reached without a final user-facing answer.")
    return "(No final answer)"

# 5. MAIN FUNCTION WITH SAMPLE QUERIES
# --------------------------------------------------------------------
# The entry point for running the script. It defines a system prompt and
# then calls `conversation_loop` for each sample query in `queries`.

def main():
    """
    The main entry point: sets a system prompt describing the available Wikipedia
    assistant functions, and runs a series of sample user queries through
    the conversation loop.
    """
    system_prompt = (
        "You are a Wikipedia assistant. You have access to the following functions:\n"
        "1) search_wikipedia(query): Return a list of page titles related to the query.\n"
        "2) fetch_wikipedia_page(title): Return a short summary of the Wikipedia page.\n"
        "3) wikipedia_assist(mode, query/latitude/longitude): For suggestions (mode='suggest') or geosearch (mode='geosearch').\n\n"
        "Use these functions to gather information and answer the user's query. "
        "If a query seems misspelled, use wikipedia_assist with mode='suggest' to get corrections. "
        "If a query asks for pages near a location, use wikipedia_assist with mode='geosearch'. "
        "When calling wikipedia_assist, specify the mode as the first argument, e.g., wikipedia_assist('suggest', query='Tmo Cruse'). "
        "Once you have sufficient information, provide a concise final answer, citing the Wikipedia pages url. "
        "You must base your answers strictly on the information returned by the functions. Do not use any prior knowledge or assumptions."
    )

    # A set of example user queries to demonstrate how the conversation loop handles them.
    queries = [
        "Fetch the summary for the page 'Times Square'.",
        "What pages talk about Greece?",
        "Who is Tmo Cruise and when was he born?",
        "Find interesting Wikipedia pages near 40.7580, -73.9855 that mention tourist attractions.",
        "I want to find pages within a 1 km radius that might be interesting around the 'Louvre Museum' in Paris.",
        "Fetch the summary for the page 'Adam Glchrst'",
        "Who is the Prime Minister of Indai?",
        "Who is Eln Msuk?",
        "What is  Artificail Inteligence.",
        "What are the leading universities in Untd Stats of Amrica ?"
    ]

    # Execute the conversation loop for each sample query.
    for user_query in queries:
        conversation_loop(system_prompt, user_query)

if __name__ == "__main__":
    main()
