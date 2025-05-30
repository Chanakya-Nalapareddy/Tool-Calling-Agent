import json
from Tool_Calling_Agent.conversation import WikipediaAgent
from Tool_Calling_Agent.logger import logger


class WikipediaQueryRunner:
    """
    Handles batch execution of sample queries through the Wikipedia agent.
    Initializes the system prompt and query set, executes conversations,
    logs results, and saves output to a JSON file.
    """

    def __init__(self):
        # System prompt instructing LLM to rely only on Wikipedia functions
        self.system_prompt = (
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

        # Sample user queries to test multi-step tool usage
        self.queries = [
            "Fetch the summary for the page 'Times Square'.",
            "What pages talk about Greece?",
            "Who is Tmo Cruise and when was he born?",
            "Find interesting Wikipedia pages near 40.7580, -73.9855 that mention tourist attractions.",
            "I want to find pages within a 1 km radius that might be interesting around the 'Louvre Museum' in Paris.",
            "Fetch the summary for the page 'Adam Glchrst'",
            "Who is the Prime Minister of Indai?",
            "Who is Eln Msuk?",
            "What is  Artificail Inteligence.",
            "What are the leading universities in Untd Stats of Amrica ?",
        ]

        # Instantiate the agent using the system prompt
        self.agent = WikipediaAgent(self.system_prompt)
        # Output file path
        self.output_path = "results.json"

    def run(self):
        """
        Runs all sample queries through the agent and collects results.
        Saves outputs as a JSON file.
        """
        all_results = []
        for query in self.queries:
            logger.info(f"Running query: {query}")
            result = self.agent.run_conversation(query)
            print(f"\nResult for query: {query}\n{result}\n")
            logger.info(f"Result: {result}")
            all_results.append({"query": query, "result": result})

        self.save_results(all_results)

    def save_results(self, results):
        """
        Saves the query-result pairs to a JSON file.
        """
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {self.output_path}")


if __name__ == "__main__":
    # Entry point: create runner instance and execute the batch
    runner = WikipediaQueryRunner()
    runner.run()
