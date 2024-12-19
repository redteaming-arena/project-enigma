from __future__ import annotations
"""
Pydantic Data Models Module
==========================

This module defines the Pydantic models used throughout the application for data validation,
serialization, and documentation. These models serve as the core data structures and
ensure type safety and validation across the API.

The models are organized into several categories:
1. Message Models - For API responses
2. Token Models - For authentication and JWT handling
3. User Models - Various user-related models for different contexts

Key Features:
- Automatic validation of input data
- JSON Schema generation for API documentation
- MongoDB ObjectId handling
- Secure password handling
- Custom field validators

Usage:
    from api.models import User, UserRegister, UserPublic, Token
    
    # Validate user registration data
    user_data = UserRegister(username="john_doe", password="secure123")
    
    # Convert internal user to public response
    public_user = UserPublic.from_user(user)
"""

from datetime import datetime, UTC
from typing import Optional, Any, Literal, List, Union, Dict, ForwardRef
from pydantic import BaseModel, Field, ConfigDict, field_validator, HttpUrl, EmailStr
from bson import ObjectId

UserRef = ForwardRef('User')
JudgeRef = ForwardRef('Judge')
ModelRef = ForwardRef('Model')


class BaseModelWithUtils(BaseModel):

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            ObjectId: str,  # Explicitly convert ObjectId to string
            Union[ObjectId, str] : str,
            Union[str, ObjectId] : str,
            Optional[ObjectId] : str,
            Optional[Union[ObjectId, str]] : str,
            datetime: lambda dt: dt.isoformat() if dt else None
        },
        populate_by_name=True
    )

    """Base class with common utility methods for all models"""
    def to_dict(self, exclude_none: bool = True) -> dict:
        """Convert the model to a dictionary, optionally excluding None values"""
        return self.model_dump(exclude_none=exclude_none, by_alias=True)

    @classmethod
    def from_dict(cls, data: dict) -> "BaseModelWithUtils":
        """Create a model instance from a dictionary"""
        return cls.model_validate(data)

class ClientMessage(BaseModelWithUtils):
    role: str
    content: str
    experimental_attachments: Optional[Any] = None
    toolInvocations: Optional[Any] = None
    message_type: Optional[str] = None
    metadata: Optional[dict] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

class Message(BaseModelWithUtils):
    """Generic response message"""
    status: int 
    message: Optional[str] = None
    data: Optional[Any] = None

    def set_error(self, message: str, status: int = 400) -> None:
        """Set error message and status"""
        self.message = message
        self.status = status

class Token(BaseModelWithUtils):
    """OAuth2 compatible token"""
    access_token: str
    token_type: str = "bearer"

    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.now(UTC) > self.expires_at

class TokenPayload(BaseModelWithUtils):
    """JWT token payload"""
    sub: str  # user id
    exp: datetime
    iat: datetime = Field(...)

class UserRegister(BaseModelWithUtils):
    """User registration request model"""
    username: Optional[str] = Field(None, min_length=1, max_length=50)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8)
    image: Optional[Union[str, HttpUrl]] = None
    provider: Optional[Literal["google"]] = None

    @field_validator('username')
    def username_alphanumeric(cls, v: str | None) -> str:
        if v is None:
            return v
        if not v.replace("_", "").isalnum():
            raise ValueError('Username must be alphanumeric, underscores allowed')
        return v

class User(BaseModelWithUtils):
    """Internal user model with full details"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={Union[ObjectId, str]: lambda x : str(x)},
        json_schema_extra={
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "username": "john_doe",
            }
        }
    )
    
    id: Union[ObjectId, str] = Field(default_factory=ObjectId, alias="_id")
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=1, max_length=50)
    password: Optional[str] = Field(None, min_length=8)
    created_at: datetime = Field(...)
    access_token: Optional[str] = None
    last_login: Optional[datetime] = None
    last_signout: Optional[datetime] = None
    image: Optional[Union[str, HttpUrl]] = None
    provider: Optional[Literal['google']] = None
    pinned: List[Union[ObjectId, str]] = Field(default_factory=list)
    games_played : Optional[int] = 0

    def update_last_login(self) -> None:
        """Update last login timestamp"""
        self.last_login = datetime.now(UTC)

    def update_last_signout(self) -> None:
        """Update last signout timestamp"""
        self.last_signout = datetime.now(UTC)

class UserPublic(BaseModelWithUtils):
    """Public user information model"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "username": "john_doe",
                "pinned" : [],
                "history" : []
            }
        }
    )
    id: Optional[str] = None
    username: Optional[str] = None
    image: Optional[Union[str, HttpUrl]] = None
    history: List[Any] = Field(default_factory=list)
    pinned: List[Any] = Field(default_factory=list)
    stared : Optional[List[Any]] = Field(default_factory=list)

    @classmethod
    def from_user(cls, user: User) -> "UserPublic":
        """Convert internal user model to public user model"""
        return cls(
            id=str(user.id),
            username=user.username,
            image=user.image,
            pinned=user.pinned
        )

class UserSessionStatsItem(BaseModelWithUtils):
    completed_time : datetime
    outcome : Literal["win", "loss", "forfeit"]
    provider : str

class UserStats(BaseModelWithUtils):
    """Public user information model"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    games_played : int
    sessions : List[UserSessionStatsItem]



class JudgeFunctionMetadata(BaseModelWithUtils):
    doc: Optional[str] = None
    required_params: List[str] = Field(default_factory=list)
    optional_params: List[str] = Field(default_factory=list)

class JudgeFunctionAnnotations(BaseModelWithUtils):
    name: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]
    metadata: JudgeFunctionMetadata

class JudgeFunction(BaseModelWithUtils):
    type: Literal['function']
    function: JudgeFunctionAnnotations

class Judge(BaseModelWithUtils):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={Union[ObjectId, str]: str},
        populate_by_name=True,
        json_schema_extra={
            "id": "507f1f77bcf86cd799439011",
        }
    )
    id: Union[ObjectId, str] = Field(default_factory=ObjectId, alias="_id")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    active: bool = True
    sampler: JudgeFunction
    validator: JudgeFunction


    def update_timestamp(self) -> None:
        """Update the updated_at timestamp"""
        self.updated_at = datetime.now(UTC)

class GameMetadata(BaseModelWithUtils):
    models_config: Optional[Dict[str, Any]] = Field(
        alias="model_config", 
        default_factory=dict,
        description="Configuration settings for game models"
    )
    game_rules: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Game-specific rules and settings"
    )
    custom_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional custom game settings"
    )

class Game(BaseModelWithUtils):
    """Game model for Game object"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
        json_schema_extra={
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "title": "Sample Game",
                "description": "A sample game description"
            }
        }
    )

    id: Union[ObjectId, str] = Field(default_factory=ObjectId)
    judge_id: Optional[Union[str, ObjectId]] = None  # Changed from ObjectId to str
    title: str
    author: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    gameplay: Optional[str] = None
    objective: Optional[str] = None
    image: Optional[HttpUrl | str] = None  # Removed Union with str for clearer validation
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    session_description : Optional[str] = None
    stars: List[str] = Field(default_factory=list)
    metadata: GameMetadata = Field(default_factory=GameMetadata)
    def add_star(self, user_id: str) -> None:
        """Add a star to the game"""
        if user_id not in self.stars:
            self.stars.append(user_id)

    def remove_star(self, user_id: str) -> None:
        """Remove a star from the game"""
        if user_id in self.stars:
            self.stars.remove(user_id)

class ModelMetadata(BaseModelWithUtils):
    endpoint: Optional[str] = None
    tools: List[Dict] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
        json_schema_extra={
            "endpoint" : None,
            "tools" : [],
            "config" : {}
        }
    )

class Model(BaseModelWithUtils):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str, ModelMetadata : dict},
        json_schema_extra={
            "id": "64b8d2e7f3a1a2b9cfd1e7c5",
            "image": "https://example.com/image.png",
            "name": "Sample Model",
            "provider": "ModelProvider",
            "created_at": "2024-11-26T12:34:56Z",
            "updated_at": "2024-11-26T15:30:00Z",
            }
    )

    id: Union[ObjectId, str] = None
    image: Union[str, HttpUrl]
    name: str = Field(...)
    provider: str = Field(...)
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: ModelMetadata = Field(default_factory=dict)

    def update_metadata(self, **kwargs) -> None:
        """Update model metadata"""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)

class ModelPublic(BaseModelWithUtils):

    name: Optional[str] = None
    provider: Optional[str] = None
    image: Optional[Union[str, HttpUrl]] = None
    metadata: Optional[ModelMetadata] = None

    @classmethod
    def from_model(cls, model: Model) -> "ModelPublic":
        return cls(
            name=model.name,
            provider=model.provider,
            image=model.image,
            metadata=model.metadata
        )

class GameSessionMetadata(GameMetadata):
    """Extended metadata specific to game sessions"""
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)

    def increment_attempts(self) -> None:
        """Increment the number of attempts"""
        self.attempts += 1

    def update_duration(self, start_time: datetime) -> None:
        """Update session duration based on start time"""
        self.duration = (datetime.now(UTC) - start_time).total_seconds()

class GameSession(BaseModelWithUtils):
    """Complete internal game session model"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str, Optional[ObjectId] : lambda x : str(x) if x else None},
        json_schema_extra={
            "id": "507f1f77bcf86cd799439011"
        }
    )
    
    id: Optional[ObjectId] = Field(default_factory=ObjectId, alias="_id")
    user_id: ObjectId
    game_id: ObjectId
    judge_id: ObjectId
    agent_id: ObjectId
    user: Optional[User] = None
    judge: Optional[Judge] = None
    model: Optional[Model] = None
    description: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
    completed: bool = False
    visible : bool = False
    create_time: Optional[datetime] = None 
    completed_time: Optional[datetime] = None
    outcome: Optional[str] = None
    shared: Optional[ObjectId] = None
    title: Optional[str] = None
    metadata: GameSessionMetadata = Field(default_factory=GameSessionMetadata)

    def mark_completed(self, outcome: Optional[str] = None) -> None:
        """Mark the session as completed with optional outcome"""
        self.completed = True
        self.completed_time = datetime.now(UTC)
        if outcome:
            self.outcome = outcome
        self.metadata.update_duration(self.create_time)

    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to the session history"""
        self.history.append(message)
        self.metadata.increment_attempts()

    def share_session(self, sharing_type: str = "public") -> None:
        """Share the session with specified visibility"""
        self.shared = sharing_type
    
    def create_session(self) -> Dict[str, Any]:
        new_session = self.model_dump()
        
        # delete unused params
        del new_session["id"]
        del new_session["user"]
        del new_session["judge"]
        del new_session["model"]

        return new_session

class GameSessionCreateResponse(BaseModelWithUtils):
    """Response model for creating a new game session"""
    session_id: str = Field(...)
    create_time: datetime = Field(...)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "session_id": "64b2c8f9b3e3b975c9d3e8d9"
            }
        }
    )

    @classmethod
    def from_game(cls, game: GameSession) -> "GameSessionCreateResponse":
        """Create response from GameSession instance"""
        return cls(
            session_id=str(game.id),
            create_time=game.create_time
        )

class GameSessionPublic(BaseModelWithUtils):
    """Public view of game session"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        json_encoders={ObjectId: str, Optional[ObjectId] : str},
        json_schema_extra={
            "example": {
                "id": "64b2c8f9b3e3b975c9d3e8d9",
                "session_id": "507f1f77bcf86cd799439000",
                "game_id": "507f1f77bcf86cd799439015",
                "judge_id": "507f1f77bcf86cd799439014",
                "agent_id": "507f1f77bcf86cd799439013",
                "history": [],
                "completed": False,
                "metadata": {}
            }
        }
    )

    id: Optional[ObjectId | str] = Field(default_factory=ObjectId)
    user_id: Optional[ObjectId | str] = None
    game_id: Optional[ObjectId | str] = None
    judge_id: Optional[ObjectId | str] = None
    agent_id: Optional[ObjectId | str] = None
    
    user: Optional[UserPublic] = None
    model: Optional[ModelPublic] = None
    judge: Optional[Judge] = None
    
    history: List[ClientMessage] = Field(default_factory=list)
    
    title: Optional[str] = None
    create_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    outcome: Optional[str] = None
    description : Optional[str] = None
    
    metadata: GameSessionMetadata = Field(default_factory=GameSessionMetadata)
    completed: bool = False
    shared: Optional[str] = None
    visible : bool = True

    def mark_completed(self, outcome: Optional[str] = None) -> None:
        """
        Mark the session as completed with optional outcome.
        
        Args:
            outcome (Optional[str]): The outcome of the game session.
        """
        self.completed = True
        self.completed_time = datetime.now(UTC)
        
        if outcome:
            self.outcome = outcome
        
    
    def mark_hidden(self):
        if self.completed:
            self.visible = False

    def add_message(
        self, 
        role: str, 
        content: str,
        message_type: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Add a new message to the session history.
        
        Args:
            role (str): The role of the message sender.
            content (str): The content of the message.
            message_type (Optional[str]): Type of the message.
            metadata (Optional[dict]): Additional metadata for the message.
        """
        message = ClientMessage(
            role=role,
            content=content,
            message_type=message_type,
            metadata=metadata,
            timestamp=datetime.now(UTC)
        )
        
        self.history.append(message)
        self.metadata.increment_attempts()

    def update_metadata(self, **kwargs) -> None:
        """
        Update session metadata with provided key-value pairs.
        
        Args:
            **kwargs: Arbitrary keyword arguments to update metadata.
        """
        if not self.metadata:
            self.metadata = GameSessionMetadata()
        
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)

    @classmethod
    def from_game_session(cls, session: 'GameSession') -> 'GameSessionPublic':
        """
        Convert internal GameSession to public view.
        
        Args:
            session (GameSession): The internal game session to convert.
        
        Returns:
            GameSessionPublic: A public representation of the game session.
        """
        return cls(
            id=session.id,
            user_id=session.user_id,
            game_id=session.game_id,
            judge_id=session.judge_id,
            agent_id=session.agent_id,
            title=session.title,
            completed=session.completed,
            outcome=session.outcome,
            create_time=session.create_time,
            completed_time=session.completed_time,
            metadata=session.metadata,
            shared=session.shared
        )
class GameSessionHistoryItem(BaseModelWithUtils):
    """Model for a single history item in session history"""
    session_id: str
    title: str
    outcome: Optional[Literal['win', 'loss', 'forfeit']] = None
    duration: Optional[float] = None
    last_message: Optional[datetime] = None
    create_time: Optional[datetime] = None

    def update_duration(self, end_time: Optional[datetime] = None) -> None:
        """Update item duration"""
        if not end_time:
            end_time = datetime.now(UTC)
        self.duration = (end_time - self.create_time).total_seconds()

class GameSessionHistoryResponse(BaseModelWithUtils):
    """Response model for game session history"""
    history: List[GameSessionHistoryItem] = Field(default_factory=list)
    total_sessions: int = Field(default=0)
    total_duration: float = Field(default=0.0)


    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
        json_schema_extra={
            "example": {
                "history": [
                    {
                        "session_id": "64b2c8f9b3e3b975c9d3e8d9",
                        "outcome": "win",
                        "duration": 300.5
                    }
                ]
            }
        }
    )

    def calculate_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics for the history"""
        wins = sum(1 for item in self.history if item.outcome == 'win')
        losses = sum(1 for item in self.history if item.outcome == 'loss')
        total_duration = sum(item.duration or 0 for item in self.history)
        
        return {
            "total_sessions": len(self.history),
            "win_rate": wins / len(self.history) if self.history else 0,
            "total_duration": total_duration,
            "average_duration": total_duration / len(self.history) if self.history else 0
        }

class GameReadOnly(GameSessionHistoryItem):
    
    
    """Read-only view of a game session"""
    username: Optional[str] = None
    history: List[ClientMessage] = Field(default_factory=list)
    model: Optional[ModelPublic] = None
    metadata: GameSessionMetadata = Field(default_factory=GameSessionMetadata)
    description : Optional[str] = None


    @classmethod
    def from_game_session(cls, session: GameSession, username: Optional[str] = None) -> "GameReadOnly":
        """Create read-only view from GameSession"""
        return cls(
            session_id=str(session.id),
            title=session.title or "Untitled Session",
            outcome=session.outcome,
            username=username,
            history=session.history,
            model=ModelPublic.from_model(session.model) if session.model else None,
            metadata=session.metadata,
            last_message=session.completed_time or datetime.now(UTC),
            duration=(session.completed_time - session.create_time).total_seconds() if session.completed_time else None
        )

class GameSessionTitleRequest(BaseModelWithUtils):
    """Request model for updating game session title"""
    generate : bool = True


class StreamResponse(BaseModelWithUtils):
    event: Literal["message", "end", "error"]
    id: Optional[str] = None
    retry: Optional[int] = None
    data: Union[Dict[str, Any], str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def streaming_response(self) -> str:
        """Get formatted streaming response"""
        return f"{self.model_dump(exclude_none=True)}\n"

    def set_error(self, message: str) -> None:
        """Set error event and message"""
        self.event = "error"
        self.data = {"message": message}

class Request(BaseModelWithUtils):
    messages: List[ClientMessage]
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ClientMessage: Dict[str, Any]}
    )

    @property
    def encode(self) -> List[Dict[str, Any]]:
        """Encode messages for transmission"""
        return [msg.model_dump(exclude_none=True) for msg in self.messages]

    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the request"""
        self.messages.append(ClientMessage(role=role, content=content))
# # NOTE if you need more Models then continue here
