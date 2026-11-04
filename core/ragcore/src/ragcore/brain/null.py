"""The stub backend's brain: no cards, no topics. Kept apart so the stub never loads LanceDB."""

from __future__ import annotations

from ragcore.api.schemas import Topic, TopicTree
from ragcore.brain.route import Routed, keyword_route


class NullBrain:
    building = False

    def tree(self) -> TopicTree:
        return TopicTree(status="none", topics=[])

    def topic(self, topic_id: str) -> Topic | None:
        return None

    async def start(self) -> None:
        return None

    async def recluster(self) -> None:
        return None

    async def route(self, question: str) -> Routed:
        return keyword_route(question)

    async def for_query(self, vector, **kwargs) -> None:
        return None
