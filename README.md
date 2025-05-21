
# Tool-Calling Agent

This project is an extended **Tool-Calling Agent** with **debug logging** and **improved parsing**. It integrates with the OpenAI API and Wikipedia to allow for interactive querying of Wikipedia content. The agent can search Wikipedia, fetch page summaries, suggest corrections for misspelled queries, and perform geosearch based on geographic coordinates.

## Features
- **Wikipedia Search**: Search for Wikipedia pages related to a given query.
- **Page Summary Fetching**: Fetch a short summary (2 sentences) of a Wikipedia page.
- **Wikipedia Assistance**:
  - **Suggest Mode**: Provides spelling suggestions for misspelled queries.
  - **Geosearch Mode**: Finds Wikipedia pages related to geographic coordinates.

## Prerequisites
Before running the project, ensure you have the following installed:

- Python 3.x
- `pip` (Python package installer)

Additionally, you will need an OpenAI API key for communication with the OpenAI model.

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Chanakya-Nalapareddy/Tool-Calling-Agent.git
cd Tool-Calling-Agent
```

### Step 2: Install Dependencies
Install required libraries using `pip`:

```bash
pip install -r requirements.txt
```

### Step 3: API Key Setup
The script uses the OpenAI API for interaction. Set up your OpenAI API key by replacing `'none'` in the `OpenAI` client initialization in `tool_calling_agent.py`:

```python
client = OpenAI(base_url='http://10.246.250.226:12300/v1', api_key='your-api-key')
```

You can get your API key from OpenAI's platform if you don't already have one.

## Usage

To start the tool-calling agent, run the script `tool_calling_agent.py`:

```bash
python tool_calling_agent.py
```

### Example Queries:
The agent will attempt to answer several predefined queries about Wikipedia:

1. Fetch the summary for the page 'Times Square'.
2. What pages talk about Greece?
3. Who is Tmo Cruise and when was he born?
4. Find interesting Wikipedia pages near 40.7580, -73.9855 that mention tourist attractions.

You can modify the `queries` list in the `main()` function to test other queries as needed.

## Functions
The following tool functions are available for the agent to call:
1. **`search_wikipedia(query)`**: Searches Wikipedia for a given query and returns a list of page titles.
2. **`fetch_wikipedia_page(title)`**: Fetches a short (2-sentence) summary for a given Wikipedia page title.
3. **`wikipedia_assist(mode, **kwargs)`**: 
   - Mode `'suggest'`: Suggests spelling corrections for a query.
   - Mode `'geosearch'`: Finds pages related to specific geographic coordinates (latitude and longitude).

## Debug Logging
The agent logs each step of the process to the console, allowing you to track the internal decision-making process. This includes calls to the Wikipedia API, responses, and any errors that may occur.

## Error Handling
Each tool function has error handling for:
- Page not found errors (for `fetch_wikipedia_page`).
- Disambiguation errors (when a query matches multiple pages).
- Invalid input or other exceptions.

## Contributing
Feel free to fork the repository, make improvements, and create a pull request. If you have any bugs or feature requests, please open an issue.

## License
This project is open-source and available under the [MIT License](LICENSE).
