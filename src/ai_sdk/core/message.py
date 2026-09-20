from dataclasses import dataclass
from uuid import uuid4


@dataclass
class Message:
    id: str
    role: str
    content: str

    @classmethod
    def create(cls, role: str, content: str) -> "Message":
        return cls(
            id=f"msg_{uuid4().hex[:8]}",
            role=role,
            content=content,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            id=(data.get("id") or f"msg_{uuid4().hex[:8]}"),
            role=data["role"],
            content=data["content"],
        )
