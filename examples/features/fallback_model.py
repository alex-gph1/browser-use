"""
Example: Using a fallback LLM model.

When the primary LLM fails with rate limits (429), authentication errors (401),
payment/credit errors (402), or server errors (500, 502, 503, 504), the agent
retries that specific request with the fallback model and then continues using
the primary model for the next request.

Note: The primary LLM will first exhaust its own retry logic (typically 5 attempts
with exponential backoff) before the fallback is triggered. This means transient errors
are handled by the provider's built-in retries, and the fallback only kicks in when
the provider truly can't recover.

This is useful for:
- High availability: Keep your agent running even when one provider has issues
- Cost optimization: Use a cheaper model as fallback when the primary is rate limited
- Multi-provider resilience: Switch between OpenAI, Anthropic, Google, etc.

@dev You need to add OPENAI_API_KEY and ANTHROPIC_API_KEY to your environment variables.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from browser_use import Agent
from browser_use.llm import ChatAnthropic, ChatOpenAI

llm = ChatAnthropic(model='claude-sonnet-4-0')
fallback_llm = ChatOpenAI(model='gpt-4o')

agent = Agent(
	task='Go to github.com and find the browser-use repository',
	llm=llm,
	fallback_llm=fallback_llm,
)


async def main():
	result = await agent.run()
	print(result)

	if agent.fallback_retry_count > 0:
		print(f'Note: Fallback LLM was used for {agent.fallback_retry_count} request(s) during execution')


if __name__ == '__main__':
	asyncio.run(main())
