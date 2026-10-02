"""Agents. Importing this package registers every built-in agent."""

from lunarlander_rl.agents import random_agent as random_agent
from lunarlander_rl.agents.base import Agent, Observation, Transition
from lunarlander_rl.agents.registry import available_agents, build_agent, register

__all__ = [
    "Agent",
    "Observation",
    "Transition",
    "available_agents",
    "build_agent",
    "register",
]
