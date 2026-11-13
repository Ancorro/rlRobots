"""SO-101 MuJoCo tasks driven by the rule decider (CPU, headless, no rendering)."""
import gymnasium as gym
import numpy as np
import pytest

import so101  # noqa: F401  registers SO101Reach-v0
from so101.decide_env import SO101DecideReach
from so101.decide_pick import SO101DecidePick
from so101.deciders import Random, Rule, options
from so101.skill_env import SO101Sort
from so101.tabletop import SO101Tabletop


def test_reach_env_steps():
    env = gym.make("SO101Reach-v0")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(10):
        obs, reward, *_ = env.step(env.action_space.sample())
    assert np.isfinite(obs).all() and np.isfinite(reward)


@pytest.mark.parametrize("make", [SO101DecideReach, SO101DecidePick, SO101Sort, SO101Tabletop])
def test_rule_decider_solves_task(make):
    env, rule = make(), Rule()
    env.reset(seed=1000)
    done = False
    while not done:  # the same loop as decide.py
        answers = rule.decide(env)
        action = [opts.index(answers[q]["choice"]) for q, opts in options(env).items()]
        _, _, term, trunc, info = env.step(action)
        done = term or trunc
    assert info["is_success"]


def test_random_decider_answers_every_question():
    env = SO101Sort()
    env.reset(seed=0)
    answers = Random(seed=0).decide(env)
    assert set(answers) == set(env.questions)
    for q, a in answers.items():
        assert a["choice"] in options(env)[q]
        assert abs(sum(a["probabilities"].values()) - 1) < 1e-9
