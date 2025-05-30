import json
from Tool_Calling_Agent.config import OpenAIClientManager
from Tool_Calling_Agent.tool_definitions import functions
from Tool_Calling_Agent.tool_implementations import WikipediaToolHandler
from Tool_Calling_Agent.utils import FunctionCallParser
from Tool_Calling_Agent.logger import logger


class WikipediaAgent:
    """Agent that interacts with Wikipedia using function-calling LLM."""

    def __init__(self, system_prompt: str):
        """Initialize the agent with prompt, tools, and client."""
        self.system_prompt = system_prompt
        self.client = OpenAIClientManager.get_client()
        self.functions = functions
        self.tool_handler = WikipediaToolHandler()
        self.max_iterations = 10
        logger.info("WikipediaAgent initialized")

    def run_conversation(self, user_query: str) -> str:
        """Run a multi-turn conversation loop with tool-calling."""
        logger.info(f"Starting new conversation for query: '{user_query}'")

        # Start conversation with system prompt and user query
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        function_calls_made = set()
        processed_pages = set()

        for iteration in range(self.max_iterations):
            logger.debug(f"Iteration {iteration + 1}: Sending to LLM")

            # Send current messages to LLM with tool-calling support
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                functions=self.functions,
                function_call="auto",
            )
            msg = response.choices[0].message
            logger.debug(f"LLM responded with: {msg.content}")
            messages.append({"role": "assistant", "content": msg.content or ""})

            func_name = None
            args = {}

            # Extract function call if LLM provided one
            if hasattr(msg, "function_call") and msg.function_call:
                func_name = msg.function_call["name"]
                args = json.loads(msg.function_call["arguments"])
                logger.info(f"Function call requested: {func_name} with args: {args}")
            else:
                # Parse function call from message text if not explicitly formatted
                func_name, args = FunctionCallParser.parse(msg.content or "")
                if not func_name:
                    final_response = msg.content.strip()
                    logger.info(
                        f"Final user-facing response (no function needed): {final_response}"
                    )
                    return final_response

            # Avoid redundant function calls
            call_key = f"{func_name}{json.dumps(args, sort_keys=True)}"
            if call_key in function_calls_made:
                logger.warning(
                    f"Repeated function call: {func_name} with same args. Skipping."
                )
                messages.append(
                    {"role": "function", "name": func_name, "content": "Already called"}
                )
                continue
            function_calls_made.add(call_key)

            # Route function call to appropriate implementation
            if func_name == "search_wikipedia":
                result = self.tool_handler.search(args.get("query", ""))
            elif func_name == "fetch_wikipedia_page":
                title = args.get("title", "")
                if title in processed_pages:
                    result = "Page already processed"
                    logger.warning(f"Skipping previously fetched page: {title}")
                else:
                    result = self.tool_handler.fetch_page(title)
                    processed_pages.add(title)
            elif func_name == "wikipedia_assist":
                mode = args.pop("mode", None)
                result = self.tool_handler.assist(mode, **args)
            else:
                result = "Unknown function"
                logger.error(f"Unknown function '{func_name}' requested.")

            logger.info(f"Result of function '{func_name}': {result}")

            # Append tool response back into conversation context
            messages.append(
                {
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps({"output": result}),
                }
            )

        # If the loop ends without a direct answer
        logger.warning("Reached maximum iterations without a final answer.")
        return "(No final answer)"
