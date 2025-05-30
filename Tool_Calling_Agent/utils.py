import json
import re
from Tool_Calling_Agent.logger import logger


class FunctionCallParser:
    """
    Utility class to extract function name and keyword arguments from a raw text string.
    Used to parse unstructured LLM responses into structured function calls.
    """

    @staticmethod
    def parse(text):
        """
        Parses a string like 'function_name(arg1="val", arg2=42)' into:
        - function name (str)
        - arguments (dict)
        """
        logger.debug(f"Attempting to parse function call from text: {text}")
        pattern = r"(\w+)\((.*?)\)"
        match = re.search(pattern, text)
        if not match:
            logger.warning("No function call pattern matched.")
            return None, {}

        func_name = match.group(1)
        raw_args = match.group(2)
        args = {}

        logger.debug(f"Matched function: {func_name}, raw args: {raw_args}")
        args_list = [arg.strip() for arg in raw_args.split(",")]

        for arg in args_list:
            if "=" in arg:
                try:
                    key, value = arg.split("=", 1)
                    args[key.strip()] = value.strip().strip('"').strip("'")
                except ValueError as ve:
                    logger.warning(f"Failed to parse argument: {arg} -> {ve}")
            else:
                logger.debug(f"Ignored non-keyword argument: {arg}")

        logger.info(f"Parsed function: {func_name} with args: {args}")
        return func_name, args
