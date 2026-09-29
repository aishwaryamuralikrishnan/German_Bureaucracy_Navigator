"""Typed exception hierarchy so the UI and tools can react to failures precisely."""


class NavigatorError(Exception):
    """Base class for all application errors."""

    user_message = "Something went wrong. Please try again."


class ConfigError(NavigatorError):
    user_message = "The app is not configured correctly. Check the README setup steps."


class ValidationError(NavigatorError):
    """Raised when user or tool input fails validation."""

    user_message = "That input is not valid."


class GuardrailViolation(NavigatorError):
    """Raised when a guardrail blocks a request."""

    user_message = "I can't help with that request."


class ExternalServiceError(NavigatorError):
    """Raised when a third-party API (OpenRouter, OSM, ...) fails."""

    user_message = "An external service is unavailable right now. Please try again shortly."


class KnowledgeBaseError(NavigatorError):
    user_message = "The knowledge base is not available. Run the ingestion first."
