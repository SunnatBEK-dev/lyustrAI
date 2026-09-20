import json
from pathlib import Path

from ai_sdk.core.conversation import Conversation
from ai_sdk.core.message import Message
from ai_sdk.storage.base import ConversationRepository


class JSONConversationRepository(ConversationRepository):
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def save(
        self,
        conversation: Conversation,
    ) -> None:
        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = [message.to_dict() for message in conversation.history()]

        with self.file_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )

    def load(self) -> Conversation:
        conversation = Conversation()

        if not self.file_path.exists():
            return conversation

        try:
            with self.file_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            return conversation

        if not isinstance(data, list):
            print("Invalid conversation format. Starting empty.")
            return conversation

        changed = False

        for item in data:
            if not isinstance(item, dict):
                continue

            try:
                had_identity = bool(item.get("id"))
                message = Message.from_dict(item)
                conversation.messages.append(message)

                if not had_identity:
                    changed = True

            except (KeyError, TypeError) as e:
                print(f"Invalid message skipped: {e}")

        if changed:
            self.save(conversation)

        return conversation
