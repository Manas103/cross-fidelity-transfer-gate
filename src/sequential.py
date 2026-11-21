"""A sequential readout of the Tier A vs Tier B route-completion gap:
Barnard's exact test on the two marginal "fully completed" counts,
recomputed at every checkpoint as rollouts accumulate.

This deliberately follows TRI's own published evaluation methodology
(Barnard's exact test on binary outcomes; see README Sources) rather than a
paired test such as McNemar's, even though the rollouts here are in fact
paired by initial condition for variance reduction elsewhere in this repo.
Barnard's test does not use that pairing, so it has less power than a test
that did; this is disclosed, not hidden, in the README Limitations."""
from scipy.stats import barnard_exact


def sequential_readout(reached_goal_a, reached_goal_b, checkpoint_every=50, alpha=0.05):
    """reached_goal_a/b are boolean arrays: did the episode actually reach
    the end of the route (Episode.reached_goal), not a threshold on the
    continuous completion fraction (see README, "What broke": a 0.999
    threshold on that fraction undercounted true completions because the
    goal check itself tolerates up to one route-sampling step, DS, short of
    the nominal length)."""
    n_total = len(reached_goal_a)
    success_a = list(reached_goal_a)
    success_b = list(reached_goal_b)
    checkpoints = []
    first_significant_n = None
    stayed_significant = False
    for n in range(checkpoint_every, n_total + 1, checkpoint_every):
        sa, fa = int(sum(success_a[:n])), n - int(sum(success_a[:n]))
        sb, fb = int(sum(success_b[:n])), n - int(sum(success_b[:n]))
        table = [[sa, fa], [sb, fb]]
        result = barnard_exact(table, alternative="two-sided")
        p = float(result.pvalue)
        significant = p < alpha
        checkpoints.append(dict(n=n, success_a=sa, success_b=sb, p_value=p, significant=significant))
        if significant and first_significant_n is None:
            first_significant_n = n
    if first_significant_n is not None:
        idx = checkpoints.index(next(c for c in checkpoints if c["n"] == first_significant_n))
        stayed_significant = all(c["significant"] for c in checkpoints[idx:])
    return dict(checkpoints=checkpoints, first_significant_n=first_significant_n,
                stayed_significant_after_first=stayed_significant)
