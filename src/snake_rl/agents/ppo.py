from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from dataclasses import dataclass, field
from typing import Any


class ActorCritic(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_dims: list[int]):
        super().__init__()
        
        # Actor network
        actor_layers = []
        curr_dim = state_size
        for h_dim in hidden_dims:
            actor_layers.append(nn.Linear(curr_dim, h_dim))
            actor_layers.append(nn.ReLU())
            curr_dim = h_dim
        actor_layers.append(nn.Linear(curr_dim, action_size))
        actor_layers.append(nn.Softmax(dim=-1))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network
        critic_layers = []
        curr_dim = state_size
        for h_dim in hidden_dims:
            critic_layers.append(nn.Linear(curr_dim, h_dim))
            critic_layers.append(nn.ReLU())
            curr_dim = h_dim
        critic_layers.append(nn.Linear(curr_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        probs = self.actor(state)
        dist = Categorical(probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


@dataclass
class PPOAgent:
    state_size: int
    action_size: int
    hidden_dims: list[int]
    learning_rate: float
    gamma: float
    eps_clip: float
    k_epochs: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        self.policy = ActorCritic(self.state_size, self.action_size, self.hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.policy_old = ActorCritic(self.state_size, self.action_size, self.hidden_dims).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.memory = []

    def select_action(self, state: tuple[int, ...]) -> int:
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state_tensor)
        
        # Store transition
        self.memory_state = state_tensor
        self.memory_action = action
        self.memory_logprob = action_logprob
        
        return int(action.item())

    def store_transition(self, reward: float, is_terminal: bool):
        self.memory.append({
            "state": self.memory_state,
            "action": self.memory_action,
            "logprob": self.memory_logprob,
            "reward": reward,
            "is_terminal": is_terminal
        })

    def update(self) -> float:
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for transition in reversed(self.memory):
            if transition["is_terminal"]:
                discounted_reward = 0
            discounted_reward = transition["reward"] + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.stack([t["state"] for t in self.memory]).detach().to(self.device)
        old_actions = torch.stack([t["action"] for t in self.memory]).detach().to(self.device)
        old_logprobs = torch.stack([t["logprob"] for t in self.memory]).detach().to(self.device)

        loss_accum = 0.0
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            loss_accum += loss.mean().item()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear memory
        self.memory = []
        
        return loss_accum / self.k_epochs

    def snapshot(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "eps_clip": self.eps_clip,
            "k_epochs": self.k_epochs,
            "hidden_dims": self.hidden_dims,
        }
