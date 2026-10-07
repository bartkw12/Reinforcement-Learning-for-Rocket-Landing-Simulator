"""Agents. Importing this package registers every built-in agent."""

from lunarlander_rl.agents import dqn as dqn
from lunarlander_rl.agents import q_learning as q_learning
from lunarlander_rl.agents import random_agent as random_agent
from lunarlander_rl.agents import reinforce as reinforce
from lunarlander_rl.agents import sb3 as sb3
from lunarlander_rl.agents.base import Agent, Observation, Policy, SelfTrainingAgent, Transition
from lunarlander_rl.agents.registry import available_agents, build_agent, register

__all__ = [
    "Agent",
    "Observation",
    "Policy",
    "SelfTrainingAgent",
    "Transition",
    "available_agents",
    "build_agent",
    "register",
]
