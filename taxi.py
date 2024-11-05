"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import os
import typing as t

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from qlearning import Info, QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
from tqdm import tqdm


def custom_frame(
    frame,
    state: int,
    reward: float,
    total_reward: float,
    action: int,
    info: Info,
    savefile: t.Optional[str] = None,
) -> np.ndarray:
    frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame)

    # Define the text and position
    text = f"State: {state}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Info: {info}"
    draw.text((10, 10), text, fill="white")

    if savefile is not None:
        frame.save(savefile)

    return np.array(frame)


#################################################
# 1. Play with QLearningAgent
#################################################


def play_and_train(
    env: gym.Env, agent: QLearningAgent, id_video: int, t_max=200, last: bool = False
) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: float = 0.0
    info: Info
    current_state, info = env.reset()

    record_video: bool = last

    if record_video:
        env = gym.wrappers.RecordVideo(
            env=env,
            video_folder=os.path.abspath("videos"),
            name_prefix="test-video",
        )
        env.episode_id = id_video
        env.start_video_recorder()
        vr = env.video_recorder
        assert vr is not None

    def convert_last_frame():
        frame = vr.recorded_frames[-1]
        frame = custom_frame(
            frame, current_state, rewardf, total_reward, action, info=info
        )
        vr.recorded_frames[-1] = frame

    for index in range(t_max):
        # Get agent to pick action given state s
        action = agent.get_action(current_state)
        next_state, reward, done, _, info = env.step(action)

        # BEGIN SOLUTION
        rewardf: float = reward  # type: ignore
        total_reward += rewardf

        if rewardf > 0:
            custom_frame(
                env.render(),
                state=current_state,
                action=action,
                reward=rewardf,
                total_reward=total_reward,
                info=info,
                savefile=f"picture/id-{id_video}-index-{index}.png",
            )

        if record_video and vr is not None:
            convert_last_frame()

        # Train agent for state s
        agent.update(
            state=current_state, action=action, reward=reward, next_state=next_state
        )
        current_state = next_state

        if done:
            break
        # END SOLUTION

    if isinstance(env, gym.wrappers.RecordVideo):
        convert_last_frame()
        env.close_video_recorder()

    return total_reward


def test_loop(env: gym.Env, agent, curve_label: str):

    rewards = []
    last = 1000
    for i in tqdm(range(1, last + 1)):
        total_reword = play_and_train(
            env, agent, t_max=200, id_video=i, last=(i > last - 2)
        )
        rewards.append(total_reword)
        if i % 100 == 0:
            print("mean reward", np.mean(rewards[-100:]))

    df = pd.DataFrame(data={"rewards": rewards})
    df["rewards_mean"] = df.groupby(df.index // 100)["rewards"].transform("mean")

    df.plot(kind="line")
    plt.savefig("curves_" + curve_label + ".png")

    # assert np.mean(rewards[-100:]) > 0.0


def test_qlearning(env: gym.Env, n_actions: int):

    agent = QLearningAgent(
        learning_rate=0.5,
        epsilon=0.25,
        gamma=0.99,
        legal_actions=list(range(n_actions)),
    )
    test_loop(env, agent, curve_label="qlearning")


# TODO: créer des vidéos de l'agent en action

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################


def test_qlearning_scheduling(env: gym.Env, n_actions: int):
    agent = QLearningAgentEpsScheduling(
        learning_rate=0.5,
        epsilon=0.25,
        gamma=0.99,
        legal_actions=list(range(n_actions)),
    )

    test_loop(env, agent, curve_label="qlearning_scheduling")


# TODO: créer des vidéos de l'agent en action


####################
# 3. Play with SARSA
####################


def test_sarsa(env: gym.Env, n_actions: int):
    agent = SarsaAgent(
        learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions))
    )

    test_loop(env, agent, curve_label="sarsa")


def test():
    env = gym.make("Taxi-v3", render_mode="rgb_array")

    n_actions = env.action_space.n  # type: ignore
    test_qlearning_scheduling(env, n_actions)
    env.close()


if __name__ == "__main__":
    test()
