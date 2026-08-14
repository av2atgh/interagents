"""
Two coupled agents (plan.md sections 3-4), built on the queueing substrate of
Oliveira & Vazquez, arXiv:0710.4916.

The paper's model draws the interacting-task priority i.i.d. from a fixed pdf.
Here the agents CHOOSE it. That single change turns a stochastic process into a
coupled inference problem, and everything the theory needs falls out of it.

--------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------
Agent j has a = L_j - 1 solitary tasks and one interacting task I.

  solitary priority   x_O ~ highest of a uniforms  (pdf a*x^(a-1))
  interacting priority x_I  <- CHOSEN by the agent's policy
  agent selects I  iff  x_I > x_O     =>  reach probability p = x_I^a

Payoffs per step:
  both reach          -> R_I     (the interaction; requires simultaneity)
  reach alone         -> x_O - c (unreciprocated: the step is wasted, minus c)
  do not reach        -> x_O     (solitary pleasure)

--------------------------------------------------------------------------
WHY THIS IS THE RIGHT MODEL FOR P5
--------------------------------------------------------------------------
The myopically optimal priority to assign the interaction is obtained from
    p*R_I + (1-p)(x_O - c) > x_O
which gives the theory's central policy equation:

        x_I* = R_I - c * (1 - p) / p                              (*)

i.e. THE PRIORITY YOU ASSIGN TO AN INTERACTION IS ITS VALUE, DISCOUNTED BY THE
RISK THAT THE OTHER WILL NOT BE THERE. The agent cannot compute (*) without a
model of the other -> P5 clause (c). It cannot execute I alone -> P5 clause (b),
"cannot be controlled, at least not fully". And p = 0 is ABSORBING: if I believe
you will not reach, I do not reach, you observe that I did not, and we both stay
solitary forever.

So this is a stag hunt. It has two attractors:
  - the coupled equilibrium   (payoff-dominant)
  - the solitary equilibrium  (risk-dominant, absorbing)
Solitary pleasure is therefore NOT a counterexample to P5 (plan.md 6.2). It is
a phase of the coupled dynamics.

--------------------------------------------------------------------------
LEVELS (plan.md 6.1: bounding the recursion)
--------------------------------------------------------------------------
  level 0: no other-model. x_I = R_I. Ignores the partner entirely.
  level 1: other-model. Tracks p_hat = partner's reach rate, applies (*).
  level 2: SELF-MODEL. Additionally tracks s_hat = the partner's estimate of ME,
           and rolls the coupled dynamics forward to see that reaching now
           raises s_hat, which raises the partner's x_I, which raises their
           reach probability. A level-2 agent will pay a short-run cost to pull
           the partner out of the solitary basin. This is the test of plan.md
           3.4: does representing yourself-as-modeled-by-another do work?
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class AgentConfig:
    L: int = 3              # queue length; a = L-1 competing solitary tasks
    level: int = 1          # theory-of-mind depth: 0, 1 or 2
    tau_mem: float = 50.0   # memory time constant of the other-model, in steps
    R_I: float = 0.99       # value of a successful interaction
    c: float = 0.35         # cost of an unreciprocated reach
    horizon: int = 40       # level-2 lookahead depth
    gamma: float = 0.97     # level-2 discount
    commit_steps: int = 1
    """How many CONSECUTIVE steps of unilateral reaching the level-2 agent is
    willing to evaluate, and to latch onto if it decides to.

    With commit_steps=1 the agent only ever asks 'is reaching worth it for one
    step?'. Reaching once at a collapsed partner almost never pays, because the
    partner's estimate barely moves before the agent reverts. Patience is
    therefore a separate axis from lookahead depth (horizon) and discount
    (gamma): an agent can be arbitrarily far-sighted and still never consider
    persisting. Sweeping this separates 'the rollout is too shallow' from 'there
    is a genuine point of no return'.

    Ignored when commit_grid is set.
    """
    commit_grid: tuple = None
    """If set, the commitment length becomes ENDOGENOUS: the agent evaluates the
    rollout at each candidate length in this tuple and latches onto the argmax.
    0 must be included -- it is the myopic branch. The chosen length is then a
    property of the agent's beliefs rather than a parameter we impose, which is
    what makes 'would an optimal agent persist?' a well-posed question.

    horizon must exceed max(commit_grid) by enough to see the payoff of the
    longest commitment, or long commitments are penalised by truncation rather
    than judged on merit.
    """
    p_init: float = 0.85    # initial belief; start inside the coupled basin
    epsilon: float = 0.0
    """Epistemic reach rate: probability per step of reaching REGARDLESS of
    belief, purely to find out whether the other is there.

    This is the epistemic term of plan.md 3.3, and it is not decoration. Without
    it p = 0 is strictly absorbing: I stop reaching, so I never observe you
    reaching, so I never revise. With it the solitary phase becomes escapable
    (both agents must explore on the same step, rate ~ epsilon^2), so the system
    can switch between phases instead of falling once. Intermittency -- and
    hence any hope of recovering the paper's power law in the LEARNED system --
    exists only for epsilon > 0.
    """
    hold_silence: bool = False
    """How the other-model treats a step with no news.

    False: the belief decays toward zero -- silence is read as rejection.
    True : the belief is HELD across the gap -- absence of evidence is not
           treated as evidence of absence. This requires the other-model to be
           maintained with no input at all, which is exactly P1. Spontaneous
           activity thereby acquires a function: keeping the other alive
           between encounters.
    """


class Agent:
    def __init__(self, cfg: AgentConfig, rng):
        self.cfg = cfg
        self.rng = rng
        self.a = cfg.L - 1
        self.lam = np.exp(-1.0 / cfg.tau_mem)      # EMA retention per step
        self.p_hat = cfg.p_init                    # other-model: P(partner reaches)
        self.s_hat = cfg.p_init                    # self-model: partner's estimate of me
        self.x_I = cfg.R_I
        # The level-2 rollout depends only on (p_hat, s_hat); deliberation is
        # expensive, so memoise the decision on a coarse belief grid.
        self._cache = {}
        self._commit_left = 0
        self.last_k = 0

    # -- policy ------------------------------------------------------------
    def _myopic_x_I(self, p):
        """Equation (*). The value of the interaction, risk-discounted."""
        p = max(p, 1e-9)
        return float(np.clip(self.cfg.R_I - self.cfg.c * (1.0 - p) / p, 0.0, 1.0))

    def _rollout(self, k):
        """Level-2: forward-simulate the coupled loop under my own model of the
        partner's model of me. Requires representing s_hat -> a self-model.

        `k` is the number of leading steps in which I participate unilaterally
        regardless of belief; k=0 is the purely myopic branch.
        """
        cfg = self.cfg
        p, s = self.p_hat, self.s_hat
        total, disc = 0.0, 1.0
        for t in range(cfg.horizon):
            # my priority this step: hold the commitment for its first k steps
            x = 1.0 if t < k else self._myopic_x_I(p)
            my_p = x**self.a
            # expected payoff this step (E[x_O] = a/(a+1) for the solitary draw)
            ex_O = self.a / (self.a + 1.0)
            total += disc * (my_p * (p * cfg.R_I + (1 - p) * (ex_O - cfg.c))
                             + (1 - my_p) * ex_O)
            # partner's model of me updates toward my reach rate...
            s = self.lam * s + (1 - self.lam) * my_p
            # ...and the partner best-responds to it, changing their reach rate
            p = self._myopic_x_I(s) ** self.a
            disc *= cfg.gamma
        return total

    def choose(self):
        if self.cfg.level == 0:
            self.x_I = float(np.clip(self.cfg.R_I, 0.0, 1.0))
        elif self.cfg.level == 1:
            self.x_I = self._myopic_x_I(self.p_hat)
        elif self._commit_left > 0:
            self._commit_left -= 1          # latched: honour the commitment
            self.x_I = 1.0
        else:
            key = (int(self.p_hat * 200), int(self.s_hat * 200))
            k = self._cache.get(key)
            if k is None:
                if self.cfg.commit_grid is None:
                    # fixed commitment length: commit or don't
                    k = (self.cfg.commit_steps
                         if self._rollout(self.cfg.commit_steps) > self._rollout(0)
                         else 0)
                else:
                    # ENDOGENOUS: the agent chooses how long to persist
                    ks = self.cfg.commit_grid
                    vals = [self._rollout(kk) for kk in ks]
                    k = ks[int(np.argmax(vals))]
                self._cache[key] = k
            self.last_k = k
            if k > 0:
                self._commit_left = k - 1
                self.x_I = 1.0
            else:
                self.x_I = self._myopic_x_I(self.p_hat)
        if self.cfg.epsilon > 0.0 and self.rng.random() < self.cfg.epsilon:
            self.x_I = 1.0                      # epistemic reach
        return self.x_I

    # -- belief updates ----------------------------------------------------
    def observe(self, partner_reached, observed, my_reach_prob):
        """Update the other-model (and, at level 2, the self-model).

        `observed` is False when the agent cannot see what the partner did this
        step. Unobserved steps still DECAY the belief -- an other-model is not
        maintained for free across a silent gap. This is where interaction
        timescale enters the architecture.
        """
        if observed:
            self.p_hat = self.lam * self.p_hat + (1 - self.lam) * float(partner_reached)
        elif not self.cfg.hold_silence:
            self.p_hat = self.lam * self.p_hat            # silence read as rejection
        self.p_hat = float(np.clip(self.p_hat, 1e-9, 1.0))
        # s_hat is maintained at EVERY level, not just level 2. Only level 2
        # *uses* it, but keeping it always means (a) err_self is meaningful for
        # any pairing, and (b) an agent upgraded to level 2 mid-run does not
        # start from a stale self-model. See run(upgrade_at=...).
        if observed or not self.cfg.hold_silence:
            self.s_hat = self.lam * self.s_hat + (1 - self.lam) * my_reach_prob


def run(cfg_A: AgentConfig, cfg_B: AgentConfig, n_steps=200_000,
        observability=1.0, avail=1.0, seed=0, burn_frac=0.2,
        upgrade_at=None, upgrade_levels=(None, None)):
    """Simulate the coupled pair.

    observability: probability of seeing the partner's choice on a step where no
      interaction occurred. 1.0 = you always see whether they reached for you.
      0.0 = you only learn about them when an interaction actually happens, so
      the interevent time controls your information rate.

    avail: probability that the interaction OPPORTUNITY exists at all on a given
      step. This sets the interaction timescale (mean interevent time >= 1/avail)
      independently of the agents' willingness, which is what makes the
      timescale question separable from the coordination question. When the
      opportunity is absent the agents neither interact nor observe each other:
      the encounter simply does not happen.
    """
    rng = np.random.default_rng(seed)
    A, B = Agent(cfg_A, rng), Agent(cfg_B, rng)
    events, payoff_A, payoff_B = [], 0.0, 0.0
    err_other, err_self, n_err = 0.0, 0.0, 0
    last_event, burn = -1, int(n_steps * burn_frac)
    p_traj = np.empty(n_steps, dtype=np.float32)

    for t in range(n_steps):
        if upgrade_at is not None and t == upgrade_at:
            # Raise theory-of-mind level mid-run. cfg_A and cfg_B MUST be
            # distinct objects or this mutates both agents at once.
            if upgrade_levels[0] is not None:
                A.cfg.level = upgrade_levels[0]
            if upgrade_levels[1] is not None:
                B.cfg.level = upgrade_levels[1]
        xA, xB = A.choose(), B.choose()
        pA, pB = xA**A.a, xB**B.a
        xOA = rng.random() ** (1.0 / A.a)
        xOB = rng.random() ** (1.0 / B.a)
        rA, rB = xA > xOA, xB > xOB
        opportunity = avail >= 1.0 or rng.random() < avail
        both = rA and rB and opportunity

        if not opportunity:
            # no encounter: solitary payoff, and no news about the other
            payoff_A += xOA
            payoff_B += xOB
            A.observe(False, False, pA)
            B.observe(False, False, pB)
            p_traj[t] = 0.5 * (pA + pB)
            if t >= burn:
                err_other += abs(A.p_hat - pB) + abs(B.p_hat - pA)
                err_self += abs(A.s_hat - B.p_hat) + abs(B.s_hat - A.p_hat)
                n_err += 2
            continue

        if both:
            payoff_A += cfg_A.R_I
            payoff_B += cfg_B.R_I
            if last_event >= 0 and t >= burn:
                events.append(t - last_event)
            last_event = t
        else:
            payoff_A += (xOA - cfg_A.c) if rA else xOA
            payoff_B += (xOB - cfg_B.c) if rB else xOB

        seen = both or (rng.random() < observability)
        A.observe(rB, seen, pA)
        B.observe(rA, seen, pB)

        p_traj[t] = 0.5 * (pA + pB)
        if t >= burn:
            err_other += abs(A.p_hat - pB) + abs(B.p_hat - pA)
            err_self += abs(A.s_hat - B.p_hat) + abs(B.s_hat - A.p_hat)
            n_err += 2

    tau = np.array(events, dtype=float)
    return dict(
        tau=tau,
        n_events=tau.size,
        rate=tau.size / max(1, n_steps - burn),
        payoff_A=payoff_A / n_steps,
        payoff_B=payoff_B / n_steps,
        err_other=err_other / max(1, n_err),
        err_self=err_self / max(1, n_err),
        p_final=float(p_traj[-5000:].mean()),
        p_traj=p_traj,
    )


def alpha_of(tau, q=0.90, tau_min_floor=3.0):
    """Hill MLE of the interevent-time tail index."""
    if tau.size < 500:
        return np.nan
    tm = max(tau_min_floor, float(np.quantile(tau, q)))
    x = tau[tau >= tm]
    if x.size < 100:
        return np.nan
    return 1.0 + x.size / np.sum(np.log(x / tm))


def regime(res, thresh=0.02):
    return "solitary" if res["p_final"] < thresh else "coupled"
