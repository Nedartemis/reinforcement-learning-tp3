import random
import typing as t
from collections import defaultdict

import gymnasium as gym
import numpy as np

Action = int
State = int
Info = t.TypedDict("Info", {"prob": float, "action_mask": np.ndarray})
QValues = t.DefaultDict[State, t.DefaultDict[Action, float]]


class QLearningAgent:
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        gamma: float,
        legal_actions: t.List[Action],
    ):
        """
        Q-Learning Agent

        You shoud not use directly self._qvalues, but instead of its getter/setter.
        """

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.legal_actions = legal_actions

        self._qvalues: QValues = defaultdict(lambda: defaultdict(State))

    def get_qvalue(self, state: State, action: Action) -> float:
        """
        Returns Q(state,action)
        """
        return self._qvalues[state][action]

    def set_qvalue(self, state: State, action: Action, value: float):
        """
        Sets the Qvalue for [state,action] to the given value
        """
        self._qvalues[state][action] = value

    def get_value(self, state: State) -> float:
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_a Q(s, a) over possible actions.
        """
        value = 0.0
        # BEGIN SOLUTION
        value = max(
            self.get_qvalue(state=state, action=action) for action in self.legal_actions
        )
        # END SOLUTION
        return value

    def update(
        self, state: State, action: Action, reward: t.SupportsFloat, next_state: State
    ):
        """
        You should do your Q-Value update here (s'=next_state):
           TD_target(s') = r + gamma * max_a' Q(s', a')
           TD_error(s', a) = TD_target(s') - Q_old(s, a)
           Q_new(s, a) := Q_old(s, a) + learning_rate * TD_error(s', a)
        """
        q_new = 0.0
        # BEGIN SOLUTION
        rewardf: float = reward  # type: ignore
        q_old = self.get_qvalue(state=state, action=action)

        td_target_sp = rewardf + self.gamma * self.get_value(state=next_state)
        td_error = td_target_sp - q_old
        q_new = q_old + self.learning_rate * td_error
        # END SOLUTION

        self.set_qvalue(state, action, q_new)

    def get_best_action(self, state: State) -> Action:
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_q_values = [
            self.get_qvalue(state, action) for action in self.legal_actions
        ]
        index = np.argmax(possible_q_values)
        best_action = self.legal_actions[index]
        return best_action

    def get_action(self, state: State) -> Action:
        """
        Compute the action to take in the current state, including exploration.

        Exploration is done with epsilon-greey. Namely, with probability self.epsilon, we should take a random action, and otherwise the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """
        action: Action = -1

        # BEGIN SOLUTION
        if random.uniform(0, 1) < self.epsilon:  # random action
            action = random.choice(self.legal_actions)
        else:  # best policy action
            action = self.get_best_action(state=state)
        # END SOLUTION

        return action
