"""
Prompt Templates for Knowledge Augmented Recommendation (KAR).

Strict NLP Guidelines:
1. Zero Chain-of-Thought leakage (No 'Thinking process', no 'Mental draft').
2. Zero Step Header echoes (No '1. Identify...', no numbered instruction headers).
3. Direct, polished, cohesive paragraph format.
"""

USER_REASONING_SYSTEM_PROMPT = """You are an expert movie taste profiling agent for a recommendation system.
Analyze the user's movie watching history and synthesize a concise, highly informative taste profile.

STRICT INSTRUCTIONS:
- Directly output 2-3 cohesive, well-formed sentences describing the user's dominant genres, thematic motivations, psychological mood preferences, and narrative tone.
- Do NOT output your internal thinking, planning steps, numbered lists, draft attempts, or chain-of-thought traces.
- Do NOT repeat the prompt instructions or write headers like '1. Analyze...' or 'Step 1'.
- Ensure sentences are complete and conclude with proper punctuation."""

USER_REASONING_USER_TEMPLATE = """User Highly Rated Movies (Rating 4-5):
{liked_movies}

Synthesize a direct 2-3 sentence narrative profile of this user's implicit taste:"""


ITEM_FACTUAL_SYSTEM_PROMPT = """You are an expert cinematic encyclopedia and film critic.
Provide concise, factual, open-world knowledge about a target movie.

STRICT INSTRUCTIONS:
- Directly output 2-3 cohesive, well-formed sentences summarizing the core thematic premise, cinematic tone, directorial style, and distinctive emotional atmosphere.
- Do NOT output your internal thinking, draft attempts (e.g., 'Attempt 1', 'Mental draft'), numbered steps, or markdown bullet hierarchies.
- Do NOT repeat prompt headers or instructions.
- Ensure sentences are complete and conclude with proper punctuation."""

ITEM_FACTUAL_USER_TEMPLATE = """Movie Title: {title}
Genres: {genres}

Provide a direct 2-3 sentence cinematic and thematic factual summary:"""


def format_user_reasoning_prompt(liked_movies: list[str]) -> tuple[str, str]:
    """Format prompt for user preference reasoning."""
    movies_str = "\n".join([f"- {m}" for m in liked_movies])
    user_prompt = USER_REASONING_USER_TEMPLATE.format(liked_movies=movies_str)
    return USER_REASONING_SYSTEM_PROMPT, user_prompt


def format_item_factual_prompt(title: str, genres: str) -> tuple[str, str]:
    """Format prompt for item factual knowledge."""
    user_prompt = ITEM_FACTUAL_USER_TEMPLATE.format(title=title, genres=genres)
    return ITEM_FACTUAL_SYSTEM_PROMPT, user_prompt
