from Tool_Calling_Agent.logger import logger

# List of function definitions that the LLM can call via tool calling.
functions = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for pages related to a query and return a list of page titles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search term for Wikipedia.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        "strict": True,
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_wikipedia_page",
            "description": "Fetches Wikipedia page summary for a given title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the Wikipedia page.",
                    }
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        },
        "strict": True,
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
                    "mode": {
                        "type": "string",
                        "enum": ["suggest", "geosearch"],
                        "description": "The type of assistance needed.",
                    },
                    "query": {
                        "type": "string",
                        "description": "The query for suggest mode.",
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Latitude for geosearch mode.",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude for geosearch mode.",
                    },
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
        },
        "strict": True,
    },
]

# Log the names of all tool functions registered for use by the LLM.
logger.info(
    f"Tool functions registered for LLM: {[f['function']['name'] for f in functions]}"
)
