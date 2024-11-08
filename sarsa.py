import random
import typing as t
from collections import defaultdict

import gymnasium as gym
import numpy as np

Action = int
State = int
Info = t.TypedDict("Info", {"prob": float, "action_mask": np.ndarray})
QValues = t.DefaultDict[int, t.DefaultDict[Action, float]]


class SarsaAgent:
    def __init__(
        self,
        learning_rate: float,
        gamma: float,
        legal_actions: t.List[Action],
    ):
        """
        SARSA  Agent

        You shoud not use directly self._qvalues, but instead of its getter/setter.
        """
        self.legal_actions = legal_actions
        self._qvalues: QValues = defaultdict(lambda: defaultdict(int))
        self.learning_rate = learning_rate
        self.gamma = gamma

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
        # END SOLUTION
        return value

    def update(
        self, state: State, action: Action, reward: t.SupportsFloat, next_state: State
    ):
        """
        You should do your Q-Value update here (s'=next_state):
           TD_target(s, a, r, s', a') = r + gamma * Q_old(s', a')
           TD_error(s, a, r, s', a') = TD_target(s, a, r, s', a') - Q_old(s, a)
           Q_new(s, a) := Q_old(s, a) + learning_rate * TD_error(s, a, R(s, a), s', a')
        """
        q_new = 0.0
        # BEGIN SOLUTION
        ap = self.get_best_action(state=next_state)
        q_old_sp_ap = self.get_qvalue(state=next_state, action=ap)
        q_old_s_a = self.get_qvalue(state=state, action=action)
        rewardf: float = reward  # type: ignore

        td_target = rewardf + self.gamma * q_old_sp_ap
        td_error = td_target - q_old_s_a
        q_new = q_old_s_a + self.learning_rate * td_error
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
        """
        action = self.legal_actions[0]

        # BEGIN SOLUTION
        action = self.get_best_action(state=state)
        # END SOLUTION

        return action
