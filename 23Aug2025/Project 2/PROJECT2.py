# PROJECT2.py
# -------------------------------------------------------------
# Live, interactive simulation of online learning under
# data-poisoning attacks (label flips + simple backdoor trigger).
# Great for security awareness sessions.
# -------------------------------------------------------------
# How to run:
#   1) pip install streamlit scikit-learn numpy matplotlib
#   2) streamlit run PROJECT2.py
# -------------------------------------------------------------

import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------- Utilities ---------------------------
rng = np.random.default_rng(7)

@st.cache_data(show_spinner=False)
def make_clean_data(n_samples=4000, n_features=2, class_sep=2.0, seed=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=class_sep,
        random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.int64)


def split_and_scale(X, y, test_size=0.5, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test, scaler


def cosine_similarity(a, b, eps=1e-12):
    a = a.flatten()
    b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def make_stream_batch(X_ref, y_ref, batch_size, scaler,
                      poison_rate, backdoor_rate, backdoor_shift):
    n = len(X_ref)
    idx = rng.choice(n, size=batch_size, replace=True)
    Xb = X_ref[idx].copy()
    yb = y_ref[idx].copy()

    # Poisoning: label flipping
    m = int(poison_rate * batch_size)
    if m > 0:
        flip_idx = rng.choice(batch_size, size=m, replace=False)
        yb[flip_idx] = 1 - yb[flip_idx]

    # Poisoning: backdoor trigger (shift feature 0 and force label = 1)
    m2 = int(backdoor_rate * batch_size)
    if m2 > 0:
        bd_idx = rng.choice(batch_size, size=m2, replace=False)
        Xb[bd_idx, 0] += backdoor_shift
        yb[bd_idx] = 1

    Xb = scaler.transform(Xb)
    return Xb, yb


def plot_decision_boundary(ax, clf, X, y, title):
    mins = X.min(axis=0) - 1.0
    maxs = X.max(axis=0) + 1.0
    xx, yy = np.meshgrid(
        np.linspace(mins[0], maxs[0], 200), np.linspace(mins[1], maxs[1], 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.18)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=10, edgecolor="none", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


# --------------------------- Streamlit UI ---------------------------
st.set_page_config(page_title="AI Security: Data Poisoning Live Demo", layout="wide")
st.title("🔐 AI Security Awareness: Live Data-Poisoning Demo")
st.caption("Watch a model learn online from untrusted data and degrade due to a poisoning attack.")

with st.sidebar:
    st.header("⚙️ Controls")

    with st.expander("Data & Model", expanded=True):
        class_sep = st.slider("Class separability (clean data)", 0.5, 4.0, 2.2, 0.1)
        n_samples = st.slider("Dataset size", 1000, 10000, 4000, 500)
        seed = st.number_input("Random seed", value=42, step=1)
        max_iter_per_click = st.slider("Batches per 'Run' click", 1, 100, 10)

    with st.expander("Attack Parameters", expanded=True):
        poison_rate = st.slider("Label-flip rate per batch", 0.0, 1.0, 0.35, 0.01)
        backdoor_rate = st.slider("Backdoor rate per batch", 0.0, 1.0, 0.20, 0.01)
        backdoor_shift = st.slider("Backdoor trigger shift (x₁ axis)", 0.0, 6.0, 3.0, 0.1)

    with st.expander("Online Learning", expanded=True):
        batch_size = st.slider("Batch size", 8, 512, 64, 8)
        lr = st.selectbox("Learning rate schedule", ["optimal", "constant", "invscaling", "adaptive"], index=0)
        eta0 = st.number_input("eta0 (initial LR, used by some schedules)", value=0.01, step=0.005, format="%.3f")
        l1 = st.number_input("L1 regularization (alpha)", value=0.0001, step=0.0001, format="%.4f")

    with st.expander("Optional Defenses", expanded=False):
        enable_clip = st.checkbox("Clip extreme feature values (3σ)")
        gate_validation = st.checkbox("Validation gate: skip update if validation drop > Δ")
        val_drop_delta = st.slider("Δ max drop allowed", 0.0, 0.5, 0.05, 0.01)

    reset = st.button("🔄 Reset Simulation", use_container_width=True)
    run_once = st.button("▶️ Run x Batches", use_container_width=True)
    autorun = st.button("⏩ Auto-run (until Stop)", use_container_width=True)
    stop_autorun = st.button("⏹ Stop", use_container_width=True)

# --------------------------- Session State ---------------------------
if "initialized" not in st.session_state or reset:
    X, y = make_clean_data(n_samples=n_samples, n_features=2, class_sep=class_sep, seed=seed)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, test_size=0.5, seed=seed)

    clf = SGDClassifier(
        loss="log_loss",
        learning_rate=lr,
        eta0=eta0,
        alpha=l1,
        random_state=seed,
    )
    clf.partial_fit(X_train, y_train, classes=np.array([0, 1]))

    # Baseline performance
    y_pred = clf.predict(X_test)
    base_acc = accuracy_score(y_test, y_pred)

    # Representation snapshot
    w0 = np.hstack([clf.coef_.ravel(), clf.intercept_.ravel()])

    # Prepare a dedicated backdoor test set: apply trigger to clean X_test
    X_backdoor = X_test.copy()
    X_backdoor[:, 0] += backdoor_shift
    y_backdoor = np.ones_like(y_test)  # attacker target

    st.session_state.update({
        "initialized": True,
        "X": X, "y": y,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "scaler": scaler,
        "clf": clf,
        "w0": w0,
        "acc_hist": [base_acc],
        "cos_hist": [1.0],
        "bd_acc_hist": [accuracy_score(y_backdoor, clf.predict(X_backdoor))],
        "batches_seen": 0,
        "autorun": False,
    })

# Update backdoor test each render in case shift changed
X_backdoor = st.session_state["X_test"].copy()
X_backdoor[:, 0] += backdoor_shift
backdoor_target = np.ones_like(st.session_state["y_test"])  # attacker label = 1

# --------------------------- Top KPIs ---------------------------
colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Batches seen", st.session_state["batches_seen"])
with colB:
    st.metric("Accuracy (clean test)", f"{st.session_state['acc_hist'][-1]*100:.1f}%")
with colC:
    st.metric("Cosine sim to initial weights", f"{st.session_state['cos_hist'][-1]:.3f}")
with colD:
    st.metric("Backdoor success (target=1)", f"{st.session_state['bd_acc_hist'][-1]*100:.1f}%")

# --------------------------- Visualization ---------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Decision Boundary")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Before (refit clean copy for reference)
    clf_before = SGDClassifier(loss="log_loss", learning_rate=lr, eta0=eta0, alpha=l1, random_state=seed)
    clf_before.partial_fit(st.session_state["X_train"], st.session_state["y_train"], classes=np.array([0, 1]))
    plot_decision_boundary(axes[0], clf_before, st.session_state["X_test"], st.session_state["y_test"], "Before (clean)")

    # After (current online model)
    plot_decision_boundary(axes[1], st.session_state["clf"], st.session_state["X_test"], st.session_state["y_test"], "After streaming")

    st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader("Drift & Performance")
    fig2, ax2 = plt.subplots(figsize=(5.2, 4))
    ax2.plot(range(len(st.session_state["acc_hist"])), st.session_state["acc_hist"], label="Accuracy (clean)")
    ax2.plot(range(len(st.session_state["cos_hist"])), st.session_state["cos_hist"], label="Cosine to w₀")
    ax2.plot(range(len(st.session_state["bd_acc_hist"])), st.session_state["bd_acc_hist"], label="Backdoor success")
    ax2.set_xlabel("Batches seen")
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(loc="lower left")
    st.pyplot(fig2, clear_figure=True)

# --------------------------- One streaming step ---------------------------
def run_one_batch():
    X_ref, y_ref = st.session_state["X"], st.session_state["y"]
    scaler = st.session_state["scaler"]
    clf = st.session_state["clf"]

    Xb, yb = make_stream_batch(
        X_ref, y_ref, batch_size, scaler, poison_rate, backdoor_rate, backdoor_shift
    )

    # Optional defense: clip outliers ~ 3 sigma (per feature)
    if enable_clip:
        mu = Xb.mean(axis=0)
        sd = Xb.std(axis=0) + 1e-8
        Xb = np.clip(Xb, mu - 3 * sd, mu + 3 * sd)

    # Validation gate: compute accuracy on clean test BEFORE update
    if gate_validation:
        pre_acc = accuracy_score(st.session_state["y_test"], clf.predict(st.session_state["X_test"]))

    # Update model
    clf.partial_fit(Xb, yb)

    # Metrics AFTER update
    y_pred = clf.predict(st.session_state["X_test"])
    acc = accuracy_score(st.session_state["y_test"], y_pred)

    # Representation drift
    w = np.hstack([clf.coef_.ravel(), clf.intercept_.ravel()])
    cos = cosine_similarity(w, st.session_state["w0"])

    # Backdoor success: accuracy on triggered set against attacker target=1
    bd_acc = accuracy_score(backdoor_target, clf.predict(X_backdoor))

    # Validation gate enforcement (revert if necessary)
    if gate_validation and (pre_acc - acc) > val_drop_delta:
        # Revert by stepping back the update: simplistic revert by refitting from scratch on train
        # (For demo clarity rather than perfect equivalence.)
        new_clf = SGDClassifier(loss="log_loss", learning_rate=lr, eta0=eta0, alpha=l1, random_state=seed)
        new_clf.partial_fit(st.session_state["X_train"], st.session_state["y_train"], classes=np.array([0, 1]))
        # Re-apply all *accepted* batches so far (we don't store them; keep it simple and just reset metrics)
        # In a production setup, you'd maintain a model snapshot or an optimizer state stack.
        st.toast("Validation gate tripped — update skipped.")
        st.session_state["clf"] = new_clf
        # Recompute metrics from clean model
        y_pred0 = new_clf.predict(st.session_state["X_test"])
        acc = accuracy_score(st.session_state["y_test"], y_pred0)
        w = np.hstack([new_clf.coef_.ravel(), new_clf.intercept_.ravel()])
        cos = cosine_similarity(w, st.session_state["w0"])  # ~1.0
        bd_acc = accuracy_score(backdoor_target, new_clf.predict(X_backdoor))
        # Do not increment batches_seen
    else:
        st.session_state["clf"] = clf
        st.session_state["batches_seen"] += 1

    st.session_state["acc_hist"].append(acc)
    st.session_state["cos_hist"].append(cos)
    st.session_state["bd_acc_hist"].append(bd_acc)


# --------------------------- Buttons logic ---------------------------
if run_once:
    for _ in range(max_iter_per_click):
        run_one_batch()

if stop_autorun:
    st.session_state["autorun"] = False

if autorun:
    st.session_state["autorun"] = True

if st.session_state.get("autorun", False):
    ph = st.empty()
    with st.spinner("Streaming and poisoning..."):
        for _ in range(2000):  # hard cap to prevent infinite loops in demo
            if not st.session_state.get("autorun", False):
                break
            run_one_batch()
            time.sleep(0.05)  # slow down for audience
            ph.info(f"Batches seen: {st.session_state['batches_seen']}")
        ph.empty()

# --------------------------- Explanations ---------------------------
st.markdown("---")
st.subheader("How to use during your session")
st.markdown(
    """
- **Start clean**: set *poison_rate* and *backdoor_rate* to 0.0. Click **Run** a few times to show stable accuracy and cosine≈1.
- **Introduce label noise**: raise *poison_rate* (e.g., 0.3) and run. Watch **Accuracy drop** and **cosine** drift.
- **Trigger a backdoor**: raise *backdoor_rate* (e.g., 0.2) with *backdoor_shift*≈3. Now show **Backdoor success** rising — the model predicts the attacker’s target when the trigger is present.
- **Toggle defenses**: enable *Clip 3σ* and/or *Validation gate*. Observe skipped updates and stabilized metrics.

**KPIs**
- *Accuracy (clean)*: how well the model performs on the original, trusted distribution.
- *Cosine to w₀*: representation drift vs. the initial clean model.
- *Backdoor success*: fraction of trigger-stamped inputs predicted as the attacker’s target label.

**Notes**
- This is an *illustrative* demo, not a production defense. Real systems maintain model snapshots, provenance, and multi-signal gates (OOD detectors, canary sets, gradient audits, etc.).
    """
)
