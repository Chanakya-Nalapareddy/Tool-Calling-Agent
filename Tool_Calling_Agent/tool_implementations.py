import wikipedia
from Tool_Calling_Agent.logger import logger


class WikipediaToolHandler:
    """
    Handles Wikipedia-related functionality: search, page summary fetch, spelling suggestions, and geosearch.
    """

    def __init__(self):
        """
        Initializes the Wikipedia tool handler and logs its creation.
        """
        logger.info("WikipediaToolHandler initialized")

    def search(self, query: str):
        """
        Searches Wikipedia for a given query string.
        Returns up to 5 page titles or an error message.
        """
        logger.info(f"Calling search_wikipedia with query: '{query}'")
        try:
            results = wikipedia.search(query, results=5)
            logger.debug(f"search_wikipedia results: {results}")
            return results
        except Exception as e:
            logger.error(f"search_wikipedia error: {e}")
            return [f"Error: {str(e)}"]

    def fetch_page(self, title: str):
        """
        Retrieves a short summary (2 sentences) for a Wikipedia page by its title.
        Handles PageError and DisambiguationError gracefully.
        """
        logger.info(f"Calling fetch_wikipedia_page with title: '{title}'")
        try:
            summary = wikipedia.summary(title, sentences=2)
            result = {"title": title, "summary": summary}
            logger.debug(f"fetch_wikipedia_page result: {result}")
            return result
        except wikipedia.exceptions.PageError:
            error = {"error": f"No Wikipedia page found for '{title}'."}
            logger.warning(error["error"])
            return error
        except wikipedia.exceptions.DisambiguationError as e:
            error = {"error": "Disambiguation error", "options": e.options}
            logger.warning(f"Disambiguation for '{title}': Options: {e.options}")
            return error
        except Exception as e:
            error = {"error": f"Error: {str(e)}"}
            logger.error(f"Unexpected error in fetch_wikipedia_page: {e}")
            return error

    def assist(self, mode: str, **kwargs):
        """
        Provides assistance for either:
        - 'suggest': to correct spelling of Wikipedia titles.
        - 'geosearch': to find pages near specified coordinates.
        Returns suggestions, geosearch results, or errors.
        """
        logger.info(f"Calling wikipedia_assist with mode='{mode}' and args={kwargs}")
        try:
            if mode == "suggest":
                query = kwargs.get("query", "")
                suggestion = wikipedia.suggest(query)
                logger.debug(f"suggest('{query}') -> {suggestion}")
                return suggestion or "No suggestion found"
            elif mode == "geosearch":
                lat = kwargs.get("latitude")
                lon = kwargs.get("longitude")
                if lat is None or lon is None:
                    logger.warning("geosearch called without coordinates.")
                    return "Error: No coordinates provided."
                geo_results = wikipedia.geosearch(lat, lon)
                logger.debug(f"geosearch({lat}, {lon}) -> {geo_results}")
                return {"latitude": lat, "longitude": lon, "results": geo_results}
            logger.warning(f"Invalid mode passed to wikipedia_assist: '{mode}'")
            return "Invalid mode"
        except Exception as e:
            logger.error(f"wikipedia_assist error: {e}")
            return f"Error: {str(e)}"
