/* The problem-independent part of the explorer (rl/dp/explore.html): seeded random draws, sampling the
   next state from the MDP's sparse successor lists, and the period-by-period replay of a policy with
   action overrides. rl/dp/explore.py inlines this file into the page; tests/unit/test_dp_explore.py also
   runs it under Node and checks the replay against the Python model. */
(function (root) {
  "use strict";

  // Deterministic 32-bit generator (mulberry32): the same seed gives the same draws everywhere.
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // One uniform draw per (seed, period), independent of everything that happened before: common random
  // numbers, so overriding the action in period t changes only what follows from it, not the later draws.
  function drawFor(seed, t) {
    let h = (Math.imul(seed >>> 0, 0x9e3779b1) ^ Math.imul((t + 1) >>> 0, 0x85ebca77)) >>> 0;
    h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
    h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    return mulberry32(h)();
  }

  // Inverse-CDF sampling over [[state, probability], ...] (the probabilities sum to one).
  function sampleNext(successors, u) {
    let acc = 0;
    for (let i = 0; i < successors.length; i++) {
      acc += successors[i][1];
      if (u < acc) return successors[i][0];
    }
    return successors[successors.length - 1][0];
  }

  function discountedSum(costs, gamma) {
    let total = 0;
    let discount = 1;
    for (const c of costs) {
      total += discount * c;
      discount *= gamma;
    }
    return total;
  }

  // Replay of the first periods of a policy. `mdp` = {gamma, cost[s][a], succ[s][a] = [[s', p], ...]};
  // `policy(t, s)` gives the action index; `horizon` ends the replay after that many periods (finite-horizon
  // policies), `maxSteps` caps the replay length.
  class Walk {
    constructor(mdp, options) {
      this.mdp = mdp;
      this.start = 0;
      this.seed = 0;
      this.maxSteps = 10;
      this.horizon = Infinity;
      this.policy = function () { return 0; };
      this.rows = [];
      this.configure(options || {});
    }

    configure(options) {
      if (options.start !== undefined) this.start = options.start;
      if (options.seed !== undefined) this.seed = options.seed;
      if (options.maxSteps !== undefined) this.maxSteps = options.maxSteps;
      if (options.horizon !== undefined) this.horizon = options.horizon;
      if (options.policy !== undefined) this.policy = options.policy;
      this.rows = [];
      return this;
    }

    get t() { return this.rows.length; }
    get state() { return this.rows.length ? this.rows[this.rows.length - 1].next : this.start; }
    get limit() { return Math.min(this.maxSteps, this.horizon); }
    get finished() { return this.t >= this.limit; }
    get total() { return this.rows.length ? this.rows[this.rows.length - 1].cumulative : 0; }

    // Advance one period; `override` (an action index) replaces the policy's choice.
    step(override) {
      if (this.finished) return null;
      const t = this.t;
      const s = this.state;
      const suggested = this.policy(t, s);
      const a = override === undefined || override === null ? suggested : override;
      const u = drawFor(this.seed, t);
      const next = sampleNext(this.mdp.succ[s][a], u);
      const discount = Math.pow(this.mdp.gamma, t);
      const cost = this.mdp.cost[s][a];
      const row = {
        t: t, s: s, a: a, suggested: suggested, overridden: a !== suggested, cost: cost, u: u, next: next,
        discount: discount, discounted: discount * cost, cumulative: this.total + discount * cost,
      };
      this.rows.push(row);
      return row;
    }

    back() { return this.rows.pop() || null; }
    reset() { this.rows = []; return this; }

    // Re-run recorded choices (null = follow the policy) under the current policy and seed.
    replay(choices) {
      this.rows = [];
      for (const c of choices) this.step(c);
      return this;
    }
  }

  root.ExploreEngine = { mulberry32: mulberry32, drawFor: drawFor, sampleNext: sampleNext, discountedSum: discountedSum, Walk: Walk };
})(typeof module !== "undefined" && module.exports ? module.exports : window);
