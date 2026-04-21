import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import chi2
from scipy.special import logsumexp


# ============================================================
# 1) DATA GENERATION
# ============================================================

def simulate_centered_data(n, d=2, seed=None):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(n, d))
    Y = Y - Y.mean(axis=0, keepdims=True)   # enforce ybar = 0
    return Y


# ============================================================
# 2) SPECIAL-CASE LOG LIKELIHOOD
# ============================================================

def compute_logC(Y):
    n, d = Y.shape
    sumsq = np.sum(Y**2)
    return -0.5 * n * d * np.log(2 * np.pi) - 0.5 * sumsq


def loglik_theta(theta, logC, n):
    return logC - 0.5 * n * np.dot(theta, theta)


# ============================================================
# 3) EXACT CONSTRAINED SAMPLER
# ============================================================

def sample_uniform_sphere(d, rng):
    z = rng.normal(size=d)
    norm_z = np.linalg.norm(z)
    while norm_z == 0:
        z = rng.normal(size=d)
        norm_z = np.linalg.norm(z)
    return z / norm_z


def sample_prior_constrained(logL_min, logC, n, d, rng):
    """
    Sample from prior N(0, I_d) conditional on log L(theta) > logL_min.
    In this special case: ||theta||^2 < r^2 with
        r^2 = (2/n)(logC - logL_min)
    """
    r2 = (2.0 / n) * (logC - logL_min)
    if r2 <= 0:
        raise ValueError("Empty constrained region.")

    cdf_upper = chi2.cdf(r2, df=d)
    u = rng.uniform(0.0, cdf_upper)
    s = chi2.ppf(u, df=d)

    radius = np.sqrt(s)
    direction = sample_uniform_sphere(d, rng)
    theta = radius * direction
    return theta


# ============================================================
# 4) NESTED SAMPLING WITH HISTORY
# ============================================================

def nested_sampling_2d_history(Y, N_live=30, max_iter=50, seed=None,
                               use_random_shrinkage=True):
    rng = np.random.default_rng(seed)
    n, d = Y.shape
    assert d == 2, "This visualization is for d=2 only."

    logC = compute_logC(Y)

    # Initialize live points from prior
    live_thetas = rng.normal(size=(N_live, d))
    live_logLs = np.array([loglik_theta(theta, logC, n) for theta in live_thetas])

    X_prev = 1.0
    log_terms = []
    history = []

    # Save initial state
    history.append({
        "iter": 0,
        "live_thetas": live_thetas.copy(),
        "live_logLs": live_logLs.copy(),
        "dead_theta": None,
        "dead_logL": None,
        "new_theta": None,
        "X_prev": 1.0,
        "X_new": 1.0,
        "w_t": None,
        "radius": None
    })

    for t in range(1, max_iter + 1):
        # 1. Find worst live point = lowest likelihood = farthest from origin
        worst_idx = np.argmin(live_logLs)
        dead_theta = live_thetas[worst_idx].copy()
        dead_logL = live_logLs[worst_idx]

        # 2. Shrink prior mass
        if use_random_shrinkage:
            T_t = rng.beta(N_live, 1)
        else:
            T_t = np.exp(-1.0 / N_live)

        X_new = X_prev * T_t
        w_t = X_prev - X_new

        # 3. Add contribution
        log_terms.append(np.log(w_t) + dead_logL)

        # 4. New constrained sample
        new_theta = sample_prior_constrained(dead_logL, logC, n, d, rng)
        new_logL = loglik_theta(new_theta, logC, n)

        # Replace worst point
        live_thetas[worst_idx] = new_theta
        live_logLs[worst_idx] = new_logL

        # Threshold circle radius
        r2 = (2.0 / n) * (logC - dead_logL)
        radius = np.sqrt(max(r2, 0.0))

        # Save state
        history.append({
            "iter": t,
            "live_thetas": live_thetas.copy(),
            "live_logLs": live_logLs.copy(),
            "dead_theta": dead_theta,
            "dead_logL": dead_logL,
            "new_theta": new_theta,
            "X_prev": X_prev,
            "X_new": X_new,
            "w_t": w_t,
            "radius": radius
        })

        X_prev = X_new

    # Final evidence estimate
    log_live_meanL = logsumexp(live_logLs) - np.log(N_live)
    logZ_live = np.log(X_prev) + log_live_meanL
    logZ_hat_no_live = logsumexp(log_terms)
    logZ_hat = logsumexp([logZ_hat_no_live, logZ_live])

    return {
        "history": history,
        "logZ_hat": logZ_hat,
        "final_X": X_prev,
        "final_live_logLs": live_logLs.copy()
    }


# ============================================================
# 5) STATIC PLOT OF ONE ITERATION
# ============================================================

def plot_iteration(history, k, xlim=(-3, 3), ylim=(-3, 3)):
    state = history[k]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Background: likelihood contours
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    Xg, Yg = np.meshgrid(xx, yy)
    R2 = Xg**2 + Yg**2
    Z = np.exp(-0.5 * R2)  # only for visual effect
    ax.contourf(Xg, Yg, Z, levels=20, alpha=0.35, cmap="viridis")

    # Live points
    live = state["live_thetas"]
    ax.scatter(live[:, 0], live[:, 1], c="black", s=35, label="Live points")

    # Dead and new points if not initial frame
    if state["dead_theta"] is not None:
        ax.scatter(state["dead_theta"][0], state["dead_theta"][1],
                   c="red", s=70, label="Removed point")
    if state["new_theta"] is not None:
        ax.scatter(state["new_theta"][0], state["new_theta"][1],
                   c="limegreen", s=70, label="New point")

    # Threshold circle
    if state["radius"] is not None:
        circle = plt.Circle((0, 0), state["radius"],
                            fill=False, linestyle="--", linewidth=2,
                            color="blue", label="Likelihood threshold")
        ax.add_patch(circle)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")

    if k == 0:
        title = "Iteration 0: initial live points from the prior"
    else:
        title = (
            f"Iteration {state['iter']} | "
            f"X_prev={state['X_prev']:.4f}, X_new={state['X_new']:.4f}"
        )
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.show()


# ============================================================
# 6) ANIMATION
# ============================================================

def animate_history(history, xlim=(-3, 3), ylim=(-3, 3), interval=800):
    fig, ax = plt.subplots(figsize=(7, 7))

    # Background contours
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    Xg, Yg = np.meshgrid(xx, yy)
    R2 = Xg**2 + Yg**2
    Z = np.exp(-0.5 * R2)
    ax.contourf(Xg, Yg, Z, levels=20, alpha=0.35, cmap="viridis")

    live_scatter = ax.scatter([], [], c="black", s=35, label="Live points")
    dead_scatter = ax.scatter([], [], c="red", s=70, label="Removed point")
    new_scatter = ax.scatter([], [], c="limegreen", s=70, label="New point")
    circle = plt.Circle((0, 0), 0.0, fill=False, linestyle="--",
                        linewidth=2, color="blue", label="Threshold")
    ax.add_patch(circle)

    text_box = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.legend(loc="upper right")

    def update(frame):
        state = history[frame]
        live = state["live_thetas"]
        live_scatter.set_offsets(live)

        if state["dead_theta"] is None:
            dead_scatter.set_offsets(np.empty((0, 2)))
            new_scatter.set_offsets(np.empty((0, 2)))
            circle.set_radius(0.0)
            ax.set_title("Iteration 0: initialization")
            text_box.set_text("All live points are sampled from the prior.")
        else:
            dead_scatter.set_offsets(state["dead_theta"].reshape(1, 2))
            new_scatter.set_offsets(state["new_theta"].reshape(1, 2))
            circle.set_radius(state["radius"])
            ax.set_title(f"Iteration {state['iter']}")
            text_box.set_text(
                f"Removed point = lowest likelihood\n"
                f"X_prev = {state['X_prev']:.5f}\n"
                f"X_new  = {state['X_new']:.5f}\n"
                f"w_t    = {state['w_t']:.5f}\n"
                f"radius = {state['radius']:.5f}"
            )

        return live_scatter, dead_scatter, new_scatter, circle, text_box

    anim = FuncAnimation(fig, update, frames=len(history),
                         interval=interval, blit=False, repeat=True)
    plt.show()
    return anim


# ============================================================
# 7) DRIVER
# ============================================================

if __name__ == "__main__":
    n = 20
    d = 2
    Y = simulate_centered_data(n, d=d, seed=42)

    out = nested_sampling_2d_history(
        Y,
        N_live=100,
        max_iter=40,
        seed=1,
        use_random_shrinkage=True
    )

    history = out["history"]

    # Show one chosen iteration
    plot_iteration(history, k=1)
    plot_iteration(history, k=10)
    plot_iteration(history, k=25)

    # Animate all iterations
    animate_history(history, xlim=(-3, 3), ylim=(-3, 3), interval=900)