"""
Service layer for the AutoStack CLI.
"""
from autostack_cli.services.llm_service import LLMService

# Create singleton instances
llm = LLMService()
