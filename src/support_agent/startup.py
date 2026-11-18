"""Startup validation: is the app configured well enough to answer questions?

Run once when the server starts (see the lifespan in api.py). Every check
reuses the validator the running app already uses, so the rules live in one
place:

    COHERE_API_KEY                   agent.api_key_from_env (present, not blank)
    COHERE_RERANK_*                  rerank.config_from_env
    COHERE_CHAT_TIMEOUT_SECONDS,
    COHERE_EMBED_TIMEOUT_SECONDS     cohere_calls.chat_timeout / embed_timeout
    CHAT_RATE_LIMIT_PER_MINUTE/DAY   rate_limit.limiter_from_env
    LOG_LEVEL                        log_config.log_level_from_env
    data/kb.json                     kb.load_articles, plus "not empty"

Nothing here imports `cohere`, creates a client or uses the network: an API
key's *presence* is checked, not whether Cohere accepts it. Error messages
name the setting and the rule, never a setting's value.
"""

import os
from collections.abc import Mapping
from pathlib import Path

from dotenv import load_dotenv

from support_agent import kb
from support_agent.agent import ConfigurationError, api_key_from_env
from support_agent.cohere_calls import chat_timeout, embed_timeout
from support_agent.log_config import log_level_from_env
from support_agent.rate_limit import limiter_from_env
from support_agent.rerank import config_from_env


class StartupError(RuntimeError):
    """The app is misconfigured. The message lists every problem found and
    never contains a setting's value."""


def _knowledge_base_problem(path: Path | None) -> str | None:
    shown = kb.KB_PATH if path is None else path
    try:
        articles = kb.load_articles(path)
    except OSError as error:  # Missing file, permissions, ...
        return f"Knowledge base {shown} could not be read ({type(error).__name__})"
    except ValueError as error:  # Not JSON, or an invalid article (kb's own rules).
        return str(error)
    if not articles:
        # Loads fine, but every question would be handed off (no_articles).
        return f"Knowledge base {shown} contains no articles"
    return None


def find_problems(environ: Mapping[str, str] | None = None, kb_path: Path | None = None) -> list[str]:
    """Run every check and return the problems found (empty list = ready).

    Each check runs even if an earlier one failed, so one deploy shows every
    problem at once.
    """
    environ = os.environ if environ is None else environ
    checks = (
        lambda: api_key_from_env(environ),
        lambda: config_from_env(environ),
        lambda: chat_timeout(environ),
        lambda: embed_timeout(environ),
        lambda: limiter_from_env(environ),
        lambda: log_level_from_env(environ),
    )

    problems = []
    for check in checks:
        try:
            check()
        except (ConfigurationError, ValueError) as error:
            problems.append(str(error))  # These messages name the setting only.
        except Exception as error:  # Unexpected: report the type, not its text.
            problems.append(f"Unexpected {type(error).__name__} while checking settings")

    kb_problem = _knowledge_base_problem(kb_path)
    if kb_problem:
        problems.append(kb_problem)
    return problems


def check_startup() -> None:
    """Load .env (local development only; real environment variables win, as
    in the running app) and raise StartupError if anything is invalid."""
    load_dotenv()
    problems = find_problems()
    if problems:
        raise StartupError(
            "The support agent cannot start: " + "; ".join(problems)
        )
