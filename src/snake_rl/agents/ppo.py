import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from dataclasses import dataclass, field
from typing import Any


class ActorCritic(nn.Module):
    def __init__(self, stateSize: int, actionSize: int, hiddenDims: list[int]):
        super().__init__()
        
        actorLayers = []
        currentDim = stateSize
        for dim in hiddenDims:
            actorLayers.append(nn.Linear(currentDim, dim))
            actorLayers.append(nn.ReLU())
            currentDim = dim
        actorLayers.append(nn.Linear(currentDim, actionSize))
        actorLayers.append(nn.Softmax(dim=-1))
        self.actor = nn.Sequential(*actorLayers)
        
        criticLayers = []
        currentDim = stateSize
        for dim in hiddenDims:
            criticLayers.append(nn.Linear(currentDim, dim))
            criticLayers.append(nn.ReLU())
            currentDim = dim
        criticLayers.append(nn.Linear(currentDim, 1))
        self.critic = nn.Sequential(*criticLayers)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        actionLogProb = dist.log_prob(action)
        return action.detach(), actionLogProb.detach()

    def evaluate(self, state, action):
        probs = self.actor(state)
        dist = Categorical(probs)
        actionLogProb = dist.log_prob(action)
        distEntropy = dist.entropy()
        stateValues = self.critic(state)
        return actionLogProb, stateValues, distEntropy


@dataclass
class PPOAgent:
    stateSize: int
    actionSize: int
    hiddenDims: list[int]
    learningRate: float
    gamma: float
    epsClip: float
    k_epochs: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        self.policy = ActorCritic(self.stateSize, self.actionSize, self.hiddenDims).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learningRate)
        self.oldPolicy = ActorCritic(self.stateSize, self.actionSize, self.hiddenDims).to(self.device)
        self.oldPolicy.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.memory = []

    def select_action(self, state: tuple[int, ...]) -> int:
        stateTensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, actionLogProb = self.oldPolicy.act(stateTensor)
        
        self.memory_state = stateTensor
        self.memory_action = action
        self.memory_logprob = actionLogProb
        
        return int(action.item())

    def storeTransition(self, reward: float, is_terminal: bool):
        self.memory.append({
            "state": self.memory_state,
            "action": self.memory_action,
            "logprob": self.memory_logprob,
            "reward": reward,
            "is_terminal": is_terminal
        })

    def update(self) -> float:
        rewards = []
        discountedReward = 0
        for transition in reversed(self.memory):
            if transition["is_terminal"]:
                discountedReward = 0
            discountedReward = transition["reward"] + (self.gamma * discountedReward)
            rewards.insert(0, discountedReward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        oldStates = torch.stack([t["state"] for t in self.memory]).detach().to(self.device)
        oldActions = torch.stack([t["action"] for t in self.memory]).detach().to(self.device)
        oldLogprobs = torch.stack([t["logprob"] for t in self.memory]).detach().to(self.device)

        accumulatedLoss = 0.0
        for _ in range(self.k_epochs):
            logprobs, stateValues, distEntropy = self.policy.evaluate(oldStates, oldActions)
            stateValues = torch.squeeze(stateValues)
            ratios = torch.exp(logprobs - oldLogprobs.detach())

            advantages = rewards - stateValues.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsClip, 1+self.epsClip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(stateValues, rewards) - 0.01*distEntropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            accumulatedLoss += loss.mean().item()
            
        self.oldPolicy.load_state_dict(self.policy.state_dict())
        self.memory = []
        
        return accumulatedLoss / self.k_epochs

    def snapshot(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learningRate,
            "gamma": self.gamma,
            "eps_clip": self.epsClip,
            "k_epochs": self.k_epochs,
            "hidden_dims": self.hiddenDims,
        }
