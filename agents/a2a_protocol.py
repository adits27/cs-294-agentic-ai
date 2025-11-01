"""
A2A (Agent-to-Agent) Protocol Implementation.

This module implements the A2A protocol for standardized communication
between agents in the multi-agent system.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class MessageType(str, Enum):
    """Types of A2A messages."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    ACK = "acknowledgment"


class MessageStatus(str, Enum):
    """Status of A2A message processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class A2AMessage(BaseModel):
    """
    Standard A2A message format for agent communication.

    This follows the Agent-to-Agent protocol specification for
    structured inter-agent communication.
    """
    message_id: str = Field(description="Unique message identifier")
    sender: str = Field(description="Agent identifier sending the message")
    receiver: str = Field(description="Agent identifier receiving the message")
    message_type: MessageType = Field(description="Type of message")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Message payload
    task: Optional[str] = Field(None, description="Task description or request")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data payload")

    # Response fields
    status: MessageStatus = Field(default=MessageStatus.PENDING)
    result: Optional[Dict[str, Any]] = Field(None, description="Result data from agent")
    error: Optional[str] = Field(None, description="Error message if failed")

    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class A2AProtocolHandler:
    """
    Handler for A2A protocol communication between agents.

    Manages message creation, validation, and routing between agents
    in the multi-agent system.
    """

    def __init__(self, agent_id: str):
        """
        Initialize the A2A protocol handler.

        Args:
            agent_id: Identifier for this agent
        """
        self.agent_id = agent_id
        self.message_counter = 0

    def create_request(
        self,
        receiver: str,
        task: str,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> A2AMessage:
        """
        Create an A2A request message.

        Args:
            receiver: Target agent identifier
            task: Task description
            data: Additional data payload
            metadata: Message metadata

        Returns:
            A2AMessage request object
        """
        self.message_counter += 1
        message_id = f"{self.agent_id}_{self.message_counter}_{datetime.utcnow().timestamp()}"

        return A2AMessage(
            message_id=message_id,
            sender=self.agent_id,
            receiver=receiver,
            message_type=MessageType.REQUEST,
            task=task,
            data=data or {},
            status=MessageStatus.PENDING,
            metadata=metadata or {}
        )

    def create_response(
        self,
        request_message: A2AMessage,
        result: Dict[str, Any],
        status: MessageStatus = MessageStatus.COMPLETED,
        error: Optional[str] = None
    ) -> A2AMessage:
        """
        Create an A2A response message for a request.

        Args:
            request_message: Original request message
            result: Result data
            status: Processing status
            error: Error message if failed

        Returns:
            A2AMessage response object
        """
        self.message_counter += 1
        message_id = f"{self.agent_id}_resp_{self.message_counter}_{datetime.utcnow().timestamp()}"

        return A2AMessage(
            message_id=message_id,
            sender=self.agent_id,
            receiver=request_message.sender,
            message_type=MessageType.RESPONSE,
            task=request_message.task,
            data=request_message.data,
            status=status,
            result=result,
            error=error,
            metadata={
                "request_id": request_message.message_id,
                "request_timestamp": request_message.timestamp
            }
        )

    def validate_message(self, message: A2AMessage) -> bool:
        """
        Validate that a message is properly formatted.

        Args:
            message: A2AMessage to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            if not message.message_id or not message.sender or not message.receiver:
                return False

            # Check message type
            if message.message_type not in MessageType:
                return False

            return True
        except Exception:
            return False

    def serialize_message(self, message: A2AMessage) -> Dict[str, Any]:
        """
        Serialize A2A message to dictionary.

        Args:
            message: A2AMessage to serialize

        Returns:
            Dictionary representation
        """
        return message.model_dump()

    def deserialize_message(self, data: Dict[str, Any]) -> A2AMessage:
        """
        Deserialize dictionary to A2A message.

        Args:
            data: Dictionary representation

        Returns:
            A2AMessage object
        """
        return A2AMessage(**data)
