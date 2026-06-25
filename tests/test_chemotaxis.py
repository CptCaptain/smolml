"""Tasks 1-3 acceptance tests for the chemotaxis control rung (self-contained)."""

import numpy as np
import torch

from smolml.envs.chemotaxis import (
    N_ACTIONS,
    ChemoConfig,
    ChemoEnv,
    RandomPolicy,
    RunAndTumble,
    action_slice,
    action_token,
    chemo_env_spec,
    conc_slice,
    drift_rates,
    ringdist,
    vocab_size,
)
from smolml.envs.spec import make_distillation_batch

# --- Task 1: Environment seam, config, vocab helpers ---------------------------


def test_vocab_layout_and_disjoint_slices():
    cfg = ChemoConfig(levels=8)
    assert vocab_size(cfg) == 8 + N_ACTIONS
    cs, as_ = conc_slice(cfg), action_slice(cfg)
    conc_ids = set(range(cs.start, cs.stop))
    act_ids = set(range(as_.start, as_.stop))
    assert conc_ids.isdisjoint(act_ids)
    assert conc_ids | act_ids == set(range(vocab_size(cfg)))
    assert [action_token(cfg, i) for i in range(N_ACTIONS)] == [8, 9, 10]


def test_drift_pools_disjoint():
    train, eval_ = set(drift_rates("train")), set(drift_rates("eval"))
    assert train and eval_ and train.isdisjoint(eval_)


def test_ringdist_wraps():
    assert ringdist(0, 15, 16) == 1
    assert ringdist(2, 6, 16) == 4


# --- Task 2: ChemoEnv dynamics + reference policies ----------------------------


def _roll(policy, env, horizon):
    c = env.reset()
    total = 0.0
    if hasattr(policy, "reset"):
        policy.reset()
    for _ in range(horizon):
        a = policy.act(c)
        c, r = env.step(a)
        total += r
    return total / horizon


def test_env_deterministic_given_seed_and_actions():
    cfg = ChemoConfig(horizon=32)
    e1, e2 = ChemoEnv(cfg, split="eval", seed=5), ChemoEnv(cfg, split="eval", seed=5)
    c1, c2 = [e1.reset()], [e2.reset()]
    for a in [0, 2, 1, 2, 0, 0, 2, 1]:
        r1 = e1.step(a)
        r2 = e2.step(a)
        assert r1 == r2
        c1.append(r1[0])
        c2.append(r2[0])
    assert c1 == c2


class _Oracle:
    def __init__(self, env):
        self.env = env

    def act(self, _conc):
        return self.env.oracle_action()


def test_metric_bounds_oracle_gt_tumble_gt_random():
    cfg = ChemoConfig(horizon=64)
    oracle, tumble, rand = [], [], []
    for s in range(40):
        e = ChemoEnv(cfg, split="eval", seed=s)
        oracle.append(_roll(_Oracle(e), e, cfg.horizon))
        e = ChemoEnv(cfg, split="eval", seed=s)
        tumble.append(_roll(RunAndTumble(epsilon=0.0, seed=s), e, cfg.horizon))
        e = ChemoEnv(cfg, split="eval", seed=s)
        rand.append(_roll(RandomPolicy(seed=s), e, cfg.horizon))
    mo, mt, mr = np.mean(oracle), np.mean(tumble), np.mean(rand)
    assert mo > mt > mr
    assert mo > 0.8  # oracle climbs and tracks the peak (reward in [0,1])


def test_source_shows_within_episode_improvement():
    cfg = ChemoConfig(horizon=64)
    first, second = [], []
    half = cfg.horizon // 2
    for s in range(40):
        e = ChemoEnv(cfg, split="train", seed=s)
        pol = RunAndTumble(epsilon=0.1, seed=s)
        c = e.reset()
        rs = []
        for _ in range(cfg.horizon):
            c, r = e.step(pol.act(c))
            rs.append(r)
        first.append(np.mean(rs[:half]))
        second.append(np.mean(rs[half:]))
    assert np.mean(second) > np.mean(first)  # the source is a learner to distill


# --- Task 3: Trajectory + make_distillation_batch (tape format) ----------------


def test_distillation_tape_format_and_shift():
    cfg = ChemoConfig(width=16, levels=8, horizon=8)
    x, y = make_distillation_batch(
        chemo_env_spec(cfg), batch_size=4, seed=0, device=torch.device("cpu")
    )
    assert x.shape == (4, 2 * cfg.horizon) == y.shape
    assert torch.equal(x[:, 1:], y[:, :-1])  # y is the next-token shift of x
    cs, as_ = conc_slice(cfg), action_slice(cfg)
    # even tape positions are concentrations, odd are actions
    full = torch.cat([x, y[:, -1:]], dim=1)  # reconstruct the (2H+1)-length tape
    for b in range(4):
        for t in range(full.shape[1]):
            tok = int(full[b, t])
            if t % 2 == 0:
                assert cs.start <= tok < cs.stop
            else:
                assert as_.start <= tok < as_.stop
