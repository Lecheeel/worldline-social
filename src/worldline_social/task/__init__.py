"""Task workflow: goal -> research -> world design -> cast -> configure -> run.

The task module turns a user goal into a simulated worldline. It is
headless (usable from the CLI) and drives the studio task UI.
"""

from .agents import DecomposeAgent, WorldDesignAgent, render_event_text
from .models import (
    STAGE_ORDER,
    Decomposition,
    ResearchNote,
    ResearchQuestion,
    SearchConfig,
    TaskRecord,
    WorldDesign,
)
from .runner import TaskBudgetExceeded, TaskRunner
from .search import SearchError, SearchResult, web_search, web_search_sync
from .store import TaskError, TaskStore

__all__ = [
    "DecomposeAgent",
    "Decomposition",
    "ResearchNote",
    "ResearchQuestion",
    "STAGE_ORDER",
    "SearchConfig",
    "SearchError",
    "SearchResult",
    "TaskBudgetExceeded",
    "TaskError",
    "TaskRecord",
    "TaskRunner",
    "TaskStore",
    "WorldDesign",
    "WorldDesignAgent",
    "render_event_text",
    "web_search",
    "web_search_sync",
]
