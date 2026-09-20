from abc import ABC, abstractmethod

from ai_sdk.core.conversation import Conversation


class ConversationRepository(ABC):
    @abstractmethod
    def save(
        self,
        conversation: Conversation,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> Conversation:
        raise NotImplementedError
