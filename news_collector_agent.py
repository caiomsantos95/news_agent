#!/usr/bin/env python3
import feedparser
import requests
from bs4 import BeautifulSoup
import yaml
import os
import openai
import json
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass
import enum
import argparse
from pathlib import Path


def load_config(config_file="config.yaml"):
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    else:
        # Default config if the file doesn't exist
        return {
            "llm": {
                "provider": "openai",
                "openai_api_key": "YOUR_OPENAI_API_KEY",
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 2048
            },
            "agent": {
                "name": "NewsCollectorAgent",
                "relevance_threshold": 0.5,
                "relevance_prompt": "Determine if the following article is relevant given the user interests. Article: \"{article_text}\" User interests: \"{interests}\". Respond with a JSON object in the format {\"relevant\": true} or {\"relevant\": false}.",
                "deduplication_prompt": "You are given a list of articles as a JSON array. Deduplicate the articles by merging ones that are similar. Return a JSON array of unique articles using the same format as provided. Articles: {articles}",
                "summary_prompt": "Given the following list of articles (in JSON format), provide a concise, bullet-point summary of the news. Articles: {articles}",
                "planning_prompt": """You are an expert news collection system planner. Given the following input parameters and current state,
                    create a detailed plan for collecting and processing news articles.

                    Input Parameters:
                    {input_params}

                    Current State:
                    {current_state}

                    Create a plan considering:
                    1. Source reliability and priority
                    2. Processing order optimization
                    3. Resource allocation
                    4. Error handling strategies

                    Respond with a JSON object:
                    {
                      "steps": [
                        {
                          "id": "step_number",
                          "action": "action_name",
                          "params": {},
                          "expected_outcome": "what should happen",
                          "fallback": "what to do if fails"
                        }
                      ],
                      "reasoning": "explanation of the plan"
                    }""",
                "step_evaluation_prompt": """You are an expert system evaluator. Analyze the results of the last step and determine the next best action.

                    Original Plan:
                    {original_plan}

                    Current Step:
                    {current_step}

                    Step Results:
                    {step_results}

                    Evaluate and respond with a JSON object:
                    {
                      "success": true/false,
                      "analysis": "what happened",
                      "next_action": "what to do next",
                      "adjustments": ["any changes needed to the plan"]
                    }"""
            }
        }


def fetch_rss(url):
    parsed_feed = feedparser.parse(url)
    if parsed_feed.bozo:
        raise Exception(f"Error parsing RSS feed: {url}")
    articles = []
    for entry in parsed_feed.entries:
        article = {
            "title": entry.get("title", "No Title"),
            "link": entry.get("link", ""),
            "summary": entry.get("summary", "")
        }
        articles.append(article)
    return articles


def scrape_aggregator(url):
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch aggregator page, status code: {response.status_code}")
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = []
    # Example heuristic: find h2 elements that contain links
    for h2 in soup.find_all('h2'):
        a = h2.find('a')
        if a:
            title = a.get_text().strip()
            link = a.get('href', '')
            articles.append({"title": title, "link": link, "summary": ""})
    if not articles:
        # Fallback: try all links in the page as articles
        for a in soup.find_all('a'):
            title = a.get_text().strip()
            link = a.get('href', '')
            if title and link:
                articles.append({"title": title, "link": link, "summary": ""})
    return articles


def generate_html_digest(articles, summary, error_log):
    html = "<html><head><title>News Digest</title></head><body>"
    html += "<h1>News Digest</h1>"
    html += "<h2>Articles</h2>"
    html += summary
    if error_log:
        html += "<h2>Errors</h2><ul>"
        for source, error in error_log.items():
            html += f"<li>{source}: {error}</li>"
        html += "</ul>"
    html += "</body></html>"
    return html


async def async_llm_call(prompt: str, config: dict) -> str:
    """Async version of LLM call with retry logic"""
    max_retries = 3
    base_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            client = openai.AsyncOpenAI(api_key=config["llm"]["openai_api_key"])
            response = await client.chat.completions.create(
                model=config["llm"]["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=config["llm"]["temperature"],
                max_tokens=config["llm"]["max_tokens"]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" in str(e).lower():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # exponential backoff
                    print(f"Rate limit reached. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
            print(f"Error in LLM call: {e}")
            return ""
    return ""


async def check_relevance_batch(articles: List[Dict], interests: str, config: dict) -> List[Dict]:
    """Process a batch of articles for relevance in parallel with rate limiting"""
    async def process_single_article(article: Dict) -> Dict:
        article_text = f"{article.get('title', '')}\n{article.get('summary', '')}"
        prompt = config["agent"]["relevance_prompt"].format(
            article_text=article_text,
            interests=interests
        )
        try:
            response_text = await async_llm_call(prompt, config)
            if response_text:
                try:
                    result = json.loads(response_text)
                    article['relevant'] = result.get('relevant', False)
                    article['relevance_confidence'] = result.get('confidence', 0.0)
                except json.JSONDecodeError:
                    print(f"Error parsing LLM response: {response_text}")
                    article['relevant'] = False
                    article['relevance_confidence'] = 0.0
            else:
                article['relevant'] = False
                article['relevance_confidence'] = 0.0
            return article
        except Exception as e:
            print(f"Error in relevance check: {e}")
            article['relevant'] = False
            article['relevance_confidence'] = 0.0
            return article

    # Process articles with rate limiting
    results = []
    batch_size = 5  # Process 5 articles at a time to avoid rate limits
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[process_single_article(article) for article in batch]
        )
        results.extend(batch_results)
        if i + batch_size < len(articles):
            await asyncio.sleep(1)  # Add delay between batches
    
    # Filter based on relevance and confidence
    threshold = config["agent"].get("relevance_threshold", 0.5)
    return [
        article for article in results 
        if article.get('relevant', False) and 
        article.get('relevance_confidence', 0.0) >= threshold
    ]


async def deduplicate_articles_llm(articles: List[Dict], config: dict) -> List[Dict]:
    """Deduplicate articles using LLM"""
    template = config["agent"].get("deduplication_prompt", "")
    articles_json = json_serialize(articles)
    prompt = template.format(articles=articles_json)
    
    try:
        response_text = await async_llm_call(prompt, config)
        if response_text:
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                print(f"Error parsing deduplication response: {response_text}")
    except Exception as e:
        print(f"Error in deduplication: {e}")
    
    return articles


async def summarize_articles_llm(articles: List[Dict], config: dict) -> str:
    """Summarize articles using LLM"""
    template = config["agent"].get("summary_prompt", "")
    articles_json = json_serialize(articles)
    prompt = template.format(articles=articles_json)
    
    try:
        return await async_llm_call(prompt, config)
    except Exception as e:
        print(f"Error in summarization: {e}")
        return "Error generating summary."


class BaseAgent:
    def __init__(self):
        self.agent_name = ""
        self.config = {}
        self.tools = {}

    def initialize(self, context: dict):
        self.config = context.get("config", {})
        self.agent_name = self.config.get("agent", {}).get("name", "BaseAgent")
        print(f"[{self.agent_name}] Initialized with config.")

    def run_step(self, input_data: dict) -> dict:
        raise NotImplementedError("Subclasses must implement run_step")

    def shutdown(self):
        print(f"[{self.agent_name}] Shutting down.")


class StepAction(enum.Enum):
    FETCH_SOURCES = "fetch_sources"
    CHECK_RELEVANCE = "check_relevance"
    DEDUPLICATE = "deduplicate"
    SUMMARIZE = "summarize"
    GENERATE_DIGEST = "generate_digest"

@dataclass
class PlanStep:
    id: int
    action: StepAction
    params: Dict[str, Any]
    expected_outcome: str
    fallback: str

@dataclass
class ExecutionPlan:
    steps: List[PlanStep]
    reasoning: str

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (PlanStep, ExecutionPlan)):
            return obj.__dict__
        if isinstance(obj, StepAction):
            return obj.value
        if isinstance(obj, enum.Enum):
            return obj.value
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return super().default(obj)

class NewsCollectorAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.error_log = {}
        self.current_plan: Optional[ExecutionPlan] = None
        self.execution_state = {
            "articles": [],
            "filtered_articles": [],
            "summary": "",
            "step_results": {},
            "processed_urls": set()  # Use a set for processed URLs
        }

    def _make_hashable(self, obj):
        """Convert lists to tuples for hashing"""
        if isinstance(obj, dict):
            return frozenset((k, self._make_hashable(v)) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            return tuple(self._make_hashable(x) for x in obj)
        elif isinstance(obj, set):
            return frozenset(self._make_hashable(x) for x in obj)
        return obj

    async def create_plan(self, input_data: dict) -> ExecutionPlan:
        """Create a plan using LLM reasoning"""
        current_state = {
            "error_log": self.error_log,
            "execution_state": self.execution_state
        }
        
        prompt = self.config["agent"]["planning_prompt"].format(
            input_params=json_serialize(input_data),
            current_state=json_serialize(current_state)
        )
        
        try:
            response = await async_llm_call(prompt, self.config)
            if response:
                try:
                    plan_data = json.loads(response)
                    steps = [
                        PlanStep(
                            id=step["id"],
                            action=StepAction(step["action"]),
                            params=step["params"],
                            expected_outcome=step["expected_outcome"],
                            fallback=step["fallback"]
                        )
                        for step in plan_data["steps"]
                    ]
                    return ExecutionPlan(steps=steps, reasoning=plan_data["reasoning"])
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Error parsing plan: {e}")
        except Exception as e:
            print(f"Error creating plan: {e}")
        
        # Fallback to default plan
        return self.create_default_plan(input_data)

    def create_default_plan(self, input_data: dict) -> ExecutionPlan:
        """Create a default plan when LLM planning fails"""
        default_steps = [
            PlanStep(
                id=1,
                action=StepAction.FETCH_SOURCES,
                params={"rss_feeds": input_data.get("rss_feeds", []),
                       "aggregator_urls": input_data.get("aggregator_urls", [])},
                expected_outcome="Fetched articles from all sources",
                fallback="Continue with successfully fetched sources"
            ),
            PlanStep(
                id=2,
                action=StepAction.CHECK_RELEVANCE,
                params={"interests": input_data.get("interests", "")},
                expected_outcome="Filtered relevant articles",
                fallback="Use all articles if filtering fails"
            ),
            # ... other default steps ...
        ]
        return ExecutionPlan(steps=default_steps, reasoning="Default sequential plan")

    async def evaluate_step(self, step: PlanStep, result: Any) -> dict:
        """Evaluate step results and determine next action"""
        prompt = self.config["agent"]["step_evaluation_prompt"].format(
            original_plan=json_serialize(self.current_plan),
            current_step=json_serialize(step),
            step_results=json_serialize(result)
        )
        
        try:
            response = await async_llm_call(prompt, self.config)
            if response:
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    print(f"Error parsing evaluation response: {response}")
        except Exception as e:
            print(f"Error evaluating step: {e}")
        
        return {
            "success": True,
            "analysis": "Evaluation failed, continuing with plan",
            "next_action": "continue",
            "adjustments": []
        }

    async def execute_step(self, step: PlanStep) -> Any:
        """Execute a single step of the plan"""
        try:
            if step.action == StepAction.FETCH_SOURCES:
                articles = []
                for url in step.params.get("rss_feeds", []):
                    if url not in self.execution_state["processed_urls"]:
                        articles.extend(await asyncio.to_thread(fetch_rss, url))
                        self.execution_state["processed_urls"].add(url)
                for url in step.params.get("aggregator_urls", []):
                    if url not in self.execution_state["processed_urls"]:
                        articles.extend(await asyncio.to_thread(scrape_aggregator, url))
                        self.execution_state["processed_urls"].add(url)
                self.execution_state["articles"].extend(articles)
                return {"articles_count": len(articles)}
            elif step.action == StepAction.CHECK_RELEVANCE:
                filtered = await check_relevance_batch(
                    self.execution_state["articles"],
                    step.params["interests"],
                    self.config
                )
                self.execution_state["filtered_articles"] = filtered
                return {"filtered_count": len(filtered)}
            # ... implement other actions ...
        except Exception as e:
            print(f"Error executing step {step.id}: {e}")
            return {"error": str(e)}

    async def async_run_step(self, input_data: dict) -> dict:
        """Main execution loop with planning and reasoning"""
        # Create initial plan
        self.current_plan = await self.create_plan(input_data)
        print(f"Created plan with {len(self.current_plan.steps)} steps")
        print(f"Reasoning: {self.current_plan.reasoning}")
        
        # Execute each step with evaluation
        for step in self.current_plan.steps:
            print(f"\nExecuting step {step.id}: {step.action.value}")
            
            # Execute step
            result = await self.execute_step(step)
            
            # Evaluate results
            evaluation = await self.evaluate_step(step, result)
            
            # Store results in execution state
            self.execution_state[step.action.value] = result
            
            # Handle evaluation results
            if not evaluation["success"]:
                print(f"Step {step.id} failed: {evaluation['analysis']}")
                if step.fallback:
                    print(f"Applying fallback: {step.fallback}")
                    # Implement fallback logic
            
            # Apply any adjustments to the plan
            for adjustment in evaluation.get("adjustments", []):
                print(f"Applying adjustment: {adjustment}")
                # Implement plan adjustment logic
        
        # Generate final digest
        return {
            "html_digest": generate_html_digest(
                self.execution_state.get("articles", []),
                self.execution_state.get("summary", ""),
                self.error_log
            ),
            "error_log": self.error_log,
            "execution_state": self.execution_state
        }

    async def fetch_all_sources(self, params: dict) -> List[Dict]:
        """Fetch articles from all sources"""
        articles = []
        rss_tasks = [
            fetch_rss(url) for url in params.get("rss_feeds", [])
        ]
        aggregator_tasks = [
            scrape_aggregator(url) for url in params.get("aggregator_urls", [])
        ]
        
        all_results = await asyncio.gather(
            *rss_tasks, *aggregator_tasks, 
            return_exceptions=True
        )
        
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                source = (
                    params["rss_feeds"][i] 
                    if i < len(params["rss_feeds"]) 
                    else params["aggregator_urls"][i - len(params["rss_feeds"])]
                )
                self.error_log[source] = str(result)
            else:
                articles.extend(result)
        
        return articles


def parse_arguments():
    parser = argparse.ArgumentParser(description='News Collector Agent')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test', action='store_true',
                       help='Run with test configuration')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode to input preferences')
    return parser.parse_args()

def setup_user_preferences(interactive: bool = False) -> dict:
    """Set up user preferences either interactively or from config"""
    if interactive:
        print("\n=== User Preferences Setup ===")
        
        print("\nEnter your topics of interest (comma-separated):")
        topics = [t.strip() for t in input().split(",") if t.strip()]
        
        print("\nHow do you prefer your content? Select multiple (comma-separated numbers):")
        print("1. In-depth analysis")
        print("2. Technical details")
        print("3. Practical implications")
        print("4. Brief summaries")
        print("5. High-level overviews")
        content_choices = input().split(",")
        content_preferences = []
        preference_map = {
            "1": "Prefer in-depth analysis over news briefs",
            "2": "Include technical details when available",
            "3": "Focus on practical implications",
            "4": "Prefer brief summaries",
            "5": "Focus on high-level overviews"
        }
        for choice in content_choices:
            if choice.strip() in preference_map:
                content_preferences.append(preference_map[choice.strip()])
        
        print("\nEnter topics to exclude (comma-separated):")
        excluded = [t.strip() for t in input().split(",") if t.strip()]
        
        return {
            "topics_of_interest": topics,
            "content_preferences": content_preferences,
            "excluded_topics": excluded
        }
    return {}

def setup_news_sources(interactive: bool = False, config: dict = None) -> dict:
    """Set up news sources either interactively or from config"""
    if interactive:
        print("\n=== News Sources Setup ===")
        
        print("\nEnter RSS feed URLs (comma-separated) or press Enter to use defaults:")
        rss_input = input().strip()
        rss_feeds = ([url.strip() for url in rss_input.split(",") if url.strip()] 
                    if rss_input 
                    else config.get("agent", {}).get("default_sources", {}).get("rss_feeds", []))
        
        print("\nEnter aggregator URLs (comma-separated) or press Enter to use defaults:")
        agg_input = input().strip()
        aggregator_urls = ([url.strip() for url in agg_input.split(",") if url.strip()]
                         if agg_input
                         else config.get("agent", {}).get("default_sources", {}).get("aggregator_urls", []))
        
        return {
            "rss_feeds": rss_feeds,
            "aggregator_urls": aggregator_urls
        }
    return {
        "rss_feeds": config.get("agent", {}).get("default_sources", {}).get("rss_feeds", []),
        "aggregator_urls": config.get("agent", {}).get("default_sources", {}).get("aggregator_urls", [])
    }

async def run_test(config: dict):
    """Run a test with sample configuration"""
    agent = NewsCollectorAgent()
    agent.initialize({"config": config})
    
    # Use test data
    input_data = {
        "rss_feeds": config["agent"]["default_sources"]["rss_feeds"],
        "aggregator_urls": config["agent"]["default_sources"]["aggregator_urls"],
        "user_preferences": config["agent"]["user_preferences"]
    }
    
    print("\n=== Running Test with Configuration ===")
    print(f"RSS Feeds: {input_data['rss_feeds']}")
    print(f"Aggregators: {input_data['aggregator_urls']}")
    print(f"User Preferences: {input_data['user_preferences']}")
    
    result = await agent.async_run_step(input_data)
    
    # Save the digest to a file
    output_file = "news_digest.html"
    with open(output_file, "w") as f:
        f.write(result["html_digest"])
    
    print(f"\nNews digest has been saved to {output_file}")
    if result["error_log"]:
        print("\n=== Errors ===")
        for source, error in result["error_log"].items():
            print(f"{source}: {error}")

async def main():
    args = parse_arguments()
    
    # Load configuration
    if args.test:
        config_file = "test_config.yaml"
    else:
        config_file = args.config
    
    config = load_config(config_file)
    
    if args.test:
        await run_test(config)
    else:
        # Interactive or normal mode
        agent = NewsCollectorAgent()
        agent.initialize({"config": config})
        
        # Get user preferences and sources
        user_preferences = setup_user_preferences(args.interactive)
        news_sources = setup_news_sources(args.interactive, config)
        
        input_data = {
            **news_sources,
            "user_preferences": user_preferences
        }
        
        result = await agent.async_run_step(input_data)
        
        # Save the digest to a file
        output_file = "news_digest.html"
        with open(output_file, "w") as f:
            f.write(result["html_digest"])
        
        print(f"\nNews digest has been saved to {output_file}")
        if result["error_log"]:
            print("\n=== Errors ===")
            for source, error in result["error_log"].items():
                print(f"{source}: {error}")

# Add this helper function
def json_serialize(obj):
    return json.dumps(obj, cls=EnhancedJSONEncoder)

if __name__ == "__main__":
    asyncio.run(main()) 