# **News Collector Agent: Architecture & Requirements**

Below is a **unified design document** in **markdown** format that can be copied into your agentic development system (like Cursor). This document merges previously discussed **agent requirements** and the **specifics** of the **News Collector Agent** use case, including a note on **OpenAI credentials** for LLM usage.

---

## **Table of Contents**

1. Overall Architecture  
2. BaseAgent Requirements  
3. NewsCollectorAgent Requirements  
4. Configuration & Credentials  
5. User Stories  
6. Utilities Outline  
7. Future Directions

---

## **1\. Overall Architecture**

┌───────────────────────────┐  
│         User             │  
│ (provides feeds, etc.)   │  
└───────────┬──────────────┘  
            │  (input)  
            ▼  
┌───────────────────────────┐  
│   NewsCollectorAgent      │  
│ \- Inherits BaseAgent      │  
│ \- Coordinates utilities   │  
│ \- Performs dedup & LLM    │  
│   summarization           │  
└───────────┬──────────────┘  
            │ calls  
            ▼  
┌─────────────────────────────────┐  
│   Utilities (RSS, scraping,    │  
│   aggregator, semantic dedup,  │  
│   LLM-based summarizer)        │  
└─────────────────────────────────┘  
            │ returns  
            ▼  
┌───────────────────────────┐  
│ Final Digest (HTML)       │  
│ \+ Error Log (if needed)   │  
└───────────────────────────┘

1. **User**: Provides a list of **RSS feeds** and **aggregator** URLs, along with **interests** in natural language.  
2. **NewsCollectorAgent**:  
   * Fetches raw news data via **scraping utilities**.  
   * Filters out irrelevant articles (LLM-based).  
   * Deduplicates similar stories (semantic matching).  
   * Summarizes main trends (LLM-based).  
   * Produces an **HTML digest**.  
3. **Utilities**:  
   * RSS fetch (e.g., using `feedparser`).  
   * Generic aggregator scrape (e.g., `requests + BeautifulSoup`).  
   * Semantic dedup (could be embeddings or LLM-based).  
   * Summarization (LLM).  
4. **Output**:  
   * A nicely formatted **HTML** digest with bullet points or short paragraphs summarizing top news.  
   * Error log for any failed feeds.

---

## **2\. BaseAgent Requirements**

### **2.1 Attributes**

* **`agent_name`**: A string identifier (e.g., "NewsCollectorAgent").  
* **`config`**: A dictionary storing runtime parameters (e.g., LLM model info, prompts, thresholds).  
* **`tools`**: A dictionary of utility references. For example, `{"scrape_rss": scrape_rss_fn, "deduplicate": deduplicate_fn}`, etc.

### **2.2 Methods**

1. **`initialize(context: Dict[str, Any]) -> None`**

   * Called once at agent startup.  
   * Preps LLM usage, loads partial config, etc.  
2. **`run_step(input_data: Dict[str, Any]) -> Dict[str, Any]`**

   * The core agent action.  
   * Consumes user or chain data (e.g., feeds, aggregator links, interests).  
   * Returns a result (e.g., `{"html_digest": "...", "error_log": {...}}`).  
3. **`shutdown() -> None`**

   * Final cleanup.  
   * Not much needed for ephemeral runs (but included for future expansions).

### **2.3 General Agent Principles**

* **LLM-Agnostic**: The base agent design does not assume a single LLM; the config can switch between models.  
* **Ephemeral Execution**: No persistent memory yet.  
* **Error Handling**: If a tool or network call fails, the agent can retry once and then log the error without crashing.

---

## **3\. NewsCollectorAgent Requirements**

### **3.1 Key Responsibilities**

1. **Fetch News Items**

   * From RSS feeds and aggregator pages.  
   * Combine into a single list of `(headline, link, snippet, etc.)`.  
2. **Relevance Filter**

   * LLM-based approach, providing `(article_text, user_interests)` → `relevant or not`.  
   * Discard irrelevant items.  
3. **Deduplicate**

   * Cluster articles covering the same story via semantic similarity.  
   * Merge duplicates into one news item with **multiple links**.  
4. **Summarize**

   * LLM-based summarization of final relevant items.  
   * Output bullet points or short paragraphs.  
5. **Generate HTML Digest**

   * Produce a nicely formatted HTML string (inspired by “brutalist report”).  
   * List any **failed** sources under “unavailable sources.”

### **3.2 Minimal Flow in `run_step()`**

1\. fetch\_all\_news():  
   \- For each feed/aggregator  
     \- Attempt to scrape  
     \- Retry once on network failure  
     \- Collect or log errors

2\. filter\_by\_relevance():  
   \- For each item, call an LLM-based “check\_relevance” utility  
   \- Drop items below threshold

3\. deduplicate():  
   \- Use an embedding or LLM-based similarity check  
   \- Merge duplicates

4\. summarize():  
   \- LLM call to produce bullet points/sections

5\. generate\_html():  
   \- Insert articles \+ summary into an HTML template  
   \- Add error info at bottom

return {  
  "html\_digest": \<string\>,  
  "error\_log": \<dict of feed \-\> error\>  
}

---

## **4\. Configuration & Credentials**

Because we’re using **OpenAI** for LLM calls, we’ll include a **config file** to store credentials and relevant parameters.

### **4.1 Example `config.yaml`**

llm:  
  provider: "openai"  
  openai\_api\_key: "\<Your-OpenAI-API-Key\>"  
  model: "gpt-3.5-turbo"  
  temperature: 0.7  
  max\_tokens: 2048

agent:  
  name: "NewsCollectorAgent"  
  relevance\_threshold: 0.5  
  summary\_prompt: "Write a concise, bullet-point summary of the following news articles..."  
  \# Additional parameters as needed

* **`llm.provider`**: `"openai"` indicates the type of LLM.  
* **`llm.openai_api_key`**: The API key for OpenAI; in practice, you might store it in a `.env` file or environment variable instead of plain YAML.  
* **`llm.model`, `temperature`, `max_tokens`**: Basic LLM usage settings.  
* **`agent.name`**: Identifies the agent in logs.  
* **`agent.relevance_threshold`**: For filtering news.  
* **`agent.summary_prompt`**: A basic prompt template for summarization.

### **4.2 Handling Sensitive Data**

* **Use Environment Variables**: Typically, you’d do `OPENAI_API_KEY=“...”` in your environment.  
* **Local `.env` File**: If using Python, a library like `python-dotenv` can load environment variables at runtime.  
* **Separate `secrets.env`**: For sensitive credentials, you might have a separate file not tracked by version control.

---

## **5\. User Stories**

1. **Basic Collection**

   * “As a user, I provide several RSS feed URLs and aggregator links, plus my interests (e.g., ‘AI research, climate news’), and the agent returns a summarized HTML digest.”  
2. **Failure Handling**

   * “As a user, if one feed times out, I want to see a note that it was unavailable, but I still want the rest of the news in the digest.”  
3. **Dedup & Summaries**

   * “As a user, I don’t want repeated versions of the same story. The agent merges duplicates and then shows one combined headline.”  
4. **Styled Output**

   * “As a user, I want a well-formatted HTML digest with bullet points and sections, so I can read it easily in my browser or email.”

---

## **6\. Utilities Outline**

1. **RSS Fetch Utility**

Pseudocode:  
 def fetch\_rss(url: str) \-\> List\[Dict\]:  
  \# use feedparser or requests  
  \# return list of articles

*   
2. **Aggregator Scrape Utility**

Pseudocode:  
 def scrape\_aggregator(url: str) \-\> List\[Dict\]:  
  \# use requests \+ BeautifulSoup  
  \# parse headlines, links  
  \# return articles

*   
3. **Relevance Checker**

LLM-based function:  
 def check\_relevance(item\_text: str, user\_interests: str) \-\> bool:  
  \# prompt the LLM  
  \# return True if relevant

*   
4. **Deduplication**

Possibly embedding-based:  
 def deduplicate\_articles(articles: List\[Dict\]) \-\> List\[Dict\]:  
  \# compute embeddings for each article  
  \# cluster or group by similarity  
  \# merge duplicates  
  return deduped\_articles

*   
5. **Summarizer**

Another LLM call:  
 def summarize\_articles(articles: List\[Dict\]) \-\> str:  
  \# prompt the LLM with bullet-point instructions  
  return summary\_text

* 

---

## **7\. Future Directions**

* **Scheduling & Orchestrator**: Let a global orchestrator run this agent daily or hourly.  
* **User Feedback**: Thumbs up/down to refine the relevance model.  
* **Memory & Logging**: Persist daily summaries, handle advanced historical queries.  
* **Architect Agent**: Another agent could propose modifications or expansions to the system code.

---

## **Final Remarks**

This document captures both the **general “Agent” requirements** and the **specifics** of the **News Collector Agent**, including:

1. **How the agent uses LLM calls** (OpenAI-based, stored in config).  
2. **Key steps** (fetch → filter → dedup → summarize → HTML).  
3. **Output** (HTML digest, error log).  
4. **Minimal config** for storing OpenAI credentials and model settings.

You can now **copy** this **markdown file** into your agentic development tool (e.g., Cursor) to guide the **implementation** of the `BaseAgent`, `NewsCollectorAgent`, and associated utilities.

