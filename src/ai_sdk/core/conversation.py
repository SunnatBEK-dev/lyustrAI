from ai_sdk.core.message import Message


class Conversation:
    def __init__(self) -> None:
        self.messages: list[Message] = []

    def add_user(self, text: str) -> Message:
        message = Message.create(
            role="user",
            content=text,
        )

        self.messages.append(message)

        return message

    def add_assistant(self, text: str) -> Message:
        message = Message.create(
            role="assistant",
            content=text,
        )

        self.messages.append(message)

        return message

    def edit_message(
        self,
        message_id: str,
        new_text: str,
    ) -> bool:
        for message in self.messages:
            if message.id == message_id:
                message.content = new_text
                return True

        return False

    def delete_message(
        self,
        message_id: str,
    ) -> bool:
        for message in self.messages:
            if message.id == message_id:
                self.messages.remove(message)
                return True

        return False

    def history(self) -> list[Message]:
        return self.messages

    def recent_messages(
        self,
        limit: int = 10,
    ) -> list[Message]:
        if limit <= 0:
            return []

        return self.messages[-limit:]

    def is_empty(self) -> bool:
        return not self.messages

    def last_message(self) -> Message | None:
        if self.is_empty():
            return None

        return self.messages[-1]

    def message_count(self) -> int:
        return len(self.messages)

    def clear(self) -> None:
        self.messages.clear()
