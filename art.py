import streamlit as st
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
# set_page_config MUST be the very first Streamlit call.
# The try/except prevents crash if Colab re-runs this cell.
try:
    st.set_page_config(
        page_title="ART Security Tool",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
except st.errors.StreamlitAPIException:
    pass

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# CSS — LIGHT THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #f5f7ff;
    color: #1a1f36;
}
.stApp {
    background: linear-gradient(160deg, #eef2ff 0%, #f5f7ff 50%, #fdf4ff 100%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 100% !important;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 55%, #9333ea 100%);
    border-radius: 18px;
    padding: 2.4rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 48px rgba(79,70,229,0.28);
}
.hero::before {
    content:''; position:absolute;
    top:-70px; right:-50px;
    width:240px; height:240px;
    background:rgba(255,255,255,0.07); border-radius:50%;
}
.hero::after {
    content:''; position:absolute;
    bottom:-50px; left:35%;
    width:160px; height:160px;
    background:rgba(255,255,255,0.04); border-radius:50%;
}
.hero h1 { font-size:2.2rem; font-weight:800; color:#fff; margin:0 0 0.4rem; letter-spacing:-0.5px; }
.hero p  { font-size:0.97rem; color:rgba(255,255,255,0.72); margin:0; font-weight:500; }
.hero-pill {
    display:inline-block;
    background:rgba(255,255,255,0.16);
    color:#fff; font-size:0.68rem; font-weight:700;
    letter-spacing:2.5px; text-transform:uppercase;
    padding:0.25rem 0.85rem; border-radius:20px;
    margin-bottom:0.9rem; border:1px solid rgba(255,255,255,0.28);
}

/* ── STEP LABEL ── */
.step-lbl {
    font-size:0.68rem; font-weight:700;
    letter-spacing:3px; text-transform:uppercase;
    color:#6d28d9; margin-bottom:0.5rem;
    display:flex; align-items:center; gap:0.5rem;
}
.step-lbl::before {
    content:''; width:18px; height:2px;
    background:#6d28d9; border-radius:2px; display:inline-block;
}

/* ── DIVIDER ── */
.divider { border:none; border-top:1px solid #e2e8f0; margin:1.6rem 0; }

/* ── RADIO ── */
div[data-testid="stRadio"] > div { gap:0.7rem !important; flex-direction:row !important; }
div[data-testid="stRadio"] label {
    background:#fff !important; border:2px solid #e2e8f0 !important;
    border-radius:12px !important; padding:0.85rem 1.5rem !important;
    font-weight:600 !important; font-size:0.93rem !important;
    color:#1a1f36 !important; cursor:pointer !important;
    transition:all 0.18s !important; flex:1 !important;
}
div[data-testid="stRadio"] label:hover {
    border-color:#6d28d9 !important; background:#faf5ff !important;
}

/* ── SELECTBOX ── */
div[data-testid="stSelectbox"] > div > div {
    background:#fff !important; border:2px solid #e2e8f0 !important;
    border-radius:10px !important; font-weight:500 !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background:linear-gradient(135deg,#4338ca,#6d28d9) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-size:0.9rem !important; font-weight:700 !important;
    padding:0.62rem 2rem !important;
    box-shadow:0 4px 14px rgba(79,70,229,0.35) !important;
    transition:all 0.18s !important; width:100% !important;
}
.stButton > button:hover {
    transform:translateY(-1px) !important;
    box-shadow:0 6px 20px rgba(79,70,229,0.45) !important;
}

/* ── FILE UPLOADER ── */
div[data-testid="stFileUploader"] {
    background:#faf5ff !important; border:2px dashed #c4b5fd !important;
    border-radius:14px !important; padding:0.8rem !important;
}

/* ── NUMBER INPUT ── */
div[data-testid="stNumberInput"] input {
    background:#fff !important; border:2px solid #e2e8f0 !important;
    border-radius:10px !important; color:#1a1f36 !important; font-weight:600 !important;
}

/* ── TEXT / NUMBER INPUT (custom row inputs) ── */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background:#fff !important; border:2px solid #e2e8f0 !important;
    border-radius:8px !important; color:#1a1f36 !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border-color:#6d28d9 !important;
    box-shadow:0 0 0 3px rgba(109,40,217,0.1) !important;
}

/* ── METRICS ROW ── */
.m-row { display:flex; gap:0.9rem; margin:1.3rem 0; flex-wrap:wrap; }
.m-card {
    flex:1; min-width:120px;
    background:#fff; border-radius:14px;
    padding:1.1rem 1.3rem; border:1px solid #e2e8f0;
    box-shadow:0 3px 14px rgba(0,0,0,0.05);
    text-align:center; position:relative; overflow:hidden;
}
.m-card::before {
    content:''; position:absolute;
    top:0; left:0; right:0; height:3px; border-radius:14px 14px 0 0;
}
.mc-v::before { background:linear-gradient(90deg,#4338ca,#6d28d9); }
.mc-g::before { background:linear-gradient(90deg,#10b981,#059669); }
.mc-r::before { background:linear-gradient(90deg,#ef4444,#dc2626); }
.mc-a::before { background:linear-gradient(90deg,#f59e0b,#d97706); }
.m-val { font-family:'JetBrains Mono',monospace; font-size:1.75rem; font-weight:600; line-height:1; margin-bottom:0.2rem; }
.mc-v .m-val { color:#4338ca; }
.mc-g .m-val { color:#10b981; }
.mc-r .m-val { color:#ef4444; }
.mc-a .m-val { color:#f59e0b; }
.m-lbl { font-size:0.68rem; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:#94a3b8; }

/* ── PREDICTION RESULT BOX ── */
.pred-box {
    background:#fff; border-radius:14px;
    padding:1.4rem 1.8rem; border:1px solid #e2e8f0;
    box-shadow:0 4px 16px rgba(0,0,0,0.05); margin-top:1rem;
}
.pred-row {
    display:flex; align-items:center; justify-content:space-between;
    padding:0.65rem 0; border-bottom:1px solid #f1f5f9;
}
.pred-row:last-child { border-bottom:none; }
.pred-key { font-size:0.74rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#94a3b8; }
.pred-val { font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:600; }
.pv-v { color:#4338ca; }
.pv-r { color:#ef4444; }

/* ── CHANGE HIGHLIGHT BOX ── */
.change-box {
    border-radius:14px; padding:1.4rem 1.8rem; margin-top:1rem;
    border-width:2px; border-style:solid;
}
.changed-box {
    background:linear-gradient(135deg,#fef2f2,#fff5f5);
    border-color:#fca5a5;
}
.same-box {
    background:linear-gradient(135deg,#f0fdf4,#f7fef9);
    border-color:#86efac;
}
.change-icon { font-size:2.4rem; margin-bottom:0.5rem; }
.change-title { font-size:1.15rem; font-weight:800; margin-bottom:0.3rem; }
.changed-box .change-title { color:#dc2626; }
.same-box    .change-title { color:#16a34a; }
.change-desc { font-size:0.88rem; color:#64748b; line-height:1.6; }
.before-after-row {
    display:flex; gap:1rem; margin-top:1rem; flex-wrap:wrap;
}
.ba-card {
    flex:1; min-width:120px;
    border-radius:10px; padding:0.9rem 1.2rem; text-align:center;
}
.ba-before { background:#eff6ff; border:1.5px solid #bfdbfe; }
.ba-after-chg { background:#fef2f2; border:1.5px solid #fecaca; }
.ba-after-same { background:#f0fdf4; border:1.5px solid #bbf7d0; }
.ba-lbl { font-size:0.65rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#94a3b8; margin-bottom:0.3rem; }
.ba-val { font-family:'JetBrains Mono',monospace; font-size:2rem; font-weight:700; }
.ba-before .ba-val { color:#2563eb; }
.ba-after-chg  .ba-val { color:#dc2626; }
.ba-after-same .ba-val { color:#16a34a; }
.arrow-sep { font-size:1.5rem; color:#94a3b8; display:flex; align-items:center; padding-top:1rem; }

/* ── BADGES ── */
.badge {
    display:inline-flex; align-items:center; gap:0.35rem;
    padding:0.28rem 0.85rem; border-radius:20px;
    font-size:0.75rem; font-weight:700; letter-spacing:0.5px;
}
.b-danger  { background:#fef2f2; color:#dc2626; border:1px solid #fecaca; }
.b-success { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; }
.b-warn    { background:#fffbeb; color:#d97706; border:1px solid #fde68a; }
.b-info    { background:#eff6ff; color:#2563eb; border:1px solid #bfdbfe; }

/* ── OBSERVATION ── */
.obs-box {
    background:linear-gradient(135deg,#fffbeb,#fef9ee);
    border:1px solid #fde68a; border-radius:12px;
    padding:1.1rem 1.4rem; margin-top:1.1rem;
}
.obs-box p { color:#92400e; font-size:0.88rem; margin:0.5rem 0 0; line-height:1.6; }

/* ── BLUE PLACEHOLDER ── */
.blue-ph {
    background:linear-gradient(135deg,#eff6ff,#f5f3ff);
    border:2px dashed #c4b5fd; border-radius:16px;
    padding:3rem 2rem; text-align:center; margin-top:1rem;
}
.blue-ph h3 { color:#4338ca; font-size:1.35rem; margin-bottom:0.55rem; }
.blue-ph p  { color:#6b7280; font-size:0.93rem; line-height:1.7; }

/* ── INPUT GRID for custom prediction ── */
.input-grid-label {
    font-size:0.72rem; font-weight:700; color:#64748b;
    letter-spacing:1.5px; text-transform:uppercase;
    margin-bottom:0.25rem;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
<div class="hero-pill">🛡️ Adversarial Robustness Toolbox</div>
<h1>ML Attack Simulation Platform</h1>
<p>Simulate real-world adversarial attacks on machine learning models — Red Team &amp; Blue Team modes</p>
</div>
""", unsafe_allow_html=True)

import uuid
# =========================================================
# STEP 1 — TEAM
# =========================================================
# =========================================================
# STEP 1 — TEAM SELECTION
# =========================================================
# =========================================================
# STEP 1 — TEAM SELECTION
# =========================================================
st.markdown('<div class="step-lbl">Step 01 — Select Team Mode</div>', unsafe_allow_html=True)

team_container = st.empty()

with team_container:
    team = st.radio(
        "Select Team Mode", 
        ["🔴  Red Teaming  —  Attack & Exploit", 
         "🔵  Blue Teaming  —  Defend & Detect"],
        label_visibility="collapsed",
        horizontal=True,
        key="team_selection_radio"
    )

is_red = team.startswith("🔴")
st.markdown('<hr class="divider">', unsafe_allow_html=True)
# =========================================================
# STEP 2 — ATTACK TYPE
# =========================================================
attack_type = None

if "attack_type" not in st.session_state:
    st.session_state.attack_type = "— Choose an attack —"

attack_type = None  # safe initialization

if is_red:
    st.markdown('<div class="step-lbl">Step 02 — Select Attack Type</div>', unsafe_allow_html=True)

    attack_type = st.selectbox(
        "atk",
        ["— Choose an attack —",
         "⚡ Evasion Attack  (Fast Gradient Sign Method)",
         "☣️ Poisoning Attack  (Label Flipping)"],
        label_visibility="collapsed",
        key="attack_type_selectbox"
    )

    # ✅ NO st.stop()
    if attack_type.startswith("—"):
        st.markdown(
            '<div style="background:#fff;border-radius:14px;padding:2rem;text-align:center;color:#94a3b8;border:1px solid #e2e8f0;">👆 Choose an attack type above to get started</div>',
            unsafe_allow_html=True
        )
    else:
        # 👉 continue your Step 3, Step 4 logic INSIDE this else
        pass

    st.markdown('<hr class="divider">', unsafe_allow_html=True)


# =========================================================
# ⚡ EVASION ATTACK
# =========================================================
    if "Evasion" in attack_type:

        st.markdown('<div class="step-lbl">Step 03 — Upload Dataset</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("CSV", type=["csv"], key="ev_csv", label_visibility="collapsed")

        if not uploaded:
            st.markdown('<div style="background:#fff;border-radius:14px;padding:1.5rem;text-align:center;color:#94a3b8;border:1px solid #e2e8f0;">📂 Upload a CSV file to continue</div>', unsafe_allow_html=True)
            st.stop()

        data = pd.read_csv(uploaded)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{data.shape[0]:,}")
        col2.metric("Columns", data.shape[1])
        col3.metric("File", uploaded.name)
        with st.expander("👁️ Preview Dataset"):
            st.dataframe(data.head(10), use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="step-lbl">Step 04 — Configure & Run</div>', unsafe_allow_html=True)

        ca, cb = st.columns([2, 1])
        with ca:
            target_col = st.selectbox(
                "🎯 Target Column",
                data.columns.tolist(),
                index=len(data.columns)-1,
                key="ev_target_col"
            )    
        with cb:
            eps = st.slider("⚡ Attack Strength (ε)", 0.01, 1.0, 0.3, 0.01)

        sc = "#10b981" if eps < 0.3 else ("#f59e0b" if eps < 0.6 else "#ef4444")
        sl = "Low" if eps < 0.3 else ("Medium" if eps < 0.6 else "High")
        st.markdown(f'<div style="margin-bottom:0.7rem;"><span class="badge" style="background:{sc}18;color:{sc};border:1px solid {sc}44;">{sl} Strength — ε = {eps}</span></div>', unsafe_allow_html=True)

        if st.button("⚡  Run Evasion Attack", key="run_ev"):
            with st.spinner("Training model & generating adversarial examples…"):
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.metrics import accuracy_score
                    from art.estimators.classification import SklearnClassifier
                    from art.attacks.evasion import FastGradientMethod

                    feat = [c for c in data.columns if c != target_col]
                    X = data[feat].values.astype(np.float32)
                    y = data[target_col].values

                    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42)
                    X_ts, X_vl, y_ts, y_vl   = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

                    sc_obj = StandardScaler()
                    X_tr = sc_obj.fit_transform(X_tr)
                    X_ts = sc_obj.transform(X_ts)
                    X_vl = sc_obj.transform(X_vl)

                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_tr, y_tr)

                    clean_ts = accuracy_score(y_ts, model.predict(X_ts))
                    clean_vl = accuracy_score(y_vl, model.predict(X_vl))
                    test_pred = model.predict(X_ts)

                    clf = SklearnClassifier(model=model)
                    X_adv = FastGradientMethod(estimator=clf, eps=float(eps)).generate(x=X_ts)
                    adv_pred = model.predict(X_adv)
                    adv_acc  = accuracy_score(y_ts, adv_pred)
                    drop     = clean_ts - adv_acc

                    # Metrics
                    st.markdown("### 📊 Attack Results")
                    st.markdown(f"""
                    <div class="m-row">
                    <div class="m-card mc-v"><div class="m-val">{clean_ts:.4f}</div><div class="m-lbl">Clean Test Acc</div></div>
                    <div class="m-card mc-g"><div class="m-val">{clean_vl:.4f}</div><div class="m-lbl">Clean Val Acc</div></div>
                    <div class="m-card mc-r"><div class="m-val">{adv_acc:.4f}</div><div class="m-lbl">Adversarial Acc</div></div>
                    <div class="m-card mc-a"><div class="m-val">↓{drop:.4f}</div><div class="m-lbl">Accuracy Drop</div></div>
                    </div>""", unsafe_allow_html=True)

                    # Chart + Comparison table side by side
                    ch1, ch2 = st.columns([1, 1])
                    with ch1:
                        fig, ax = plt.subplots(figsize=(4.5, 3.2), facecolor="white")
                        ax.set_facecolor("#fafbff")
                        bars = ax.bar(["Clean Test","Clean Val","Adversarial"],
                                    [clean_ts, clean_vl, adv_acc],
                                    color=["#4338ca","#10b981","#ef4444"],
                                    width=0.42, edgecolor="white", linewidth=1.5)
                        ax.set_ylim(0, 1.12)
                        ax.set_ylabel("Accuracy", fontsize=9, color="#64748b")
                        ax.set_title("Before vs After Evasion Attack", fontsize=10, fontweight="bold", color="#1a1f36", pad=10)
                        ax.tick_params(colors="#64748b", labelsize=9)
                        for sp in ax.spines.values(): sp.set_color("#e2e8f0")
                        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                        ax.yaxis.grid(True, color="#f1f5f9", linewidth=1); ax.set_axisbelow(True)
                        for bar, val in zip(bars, [clean_ts, clean_vl, adv_acc]):
                            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.025,
                                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1a1f36")
                        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

                    with ch2:
                        st.markdown("**🔍 First Sample: Original vs Perturbed**")
                        orig_inv = sc_obj.inverse_transform([X_ts[0]])[0]
                        adv_inv  = sc_obj.inverse_transform([X_adv[0]])[0]
                        cmp_df = pd.DataFrame({
                            "Feature":        feat,
                            "Original":       [round(float(v), 4) for v in orig_inv],
                            "Perturbed":      [round(float(v), 4) for v in adv_inv],
                            "Δ":              [round(float(adv_inv[i]-orig_inv[i]), 4) for i in range(len(orig_inv))],
                        })
                        st.dataframe(cmp_df, use_container_width=True, height=250)

                    # Prediction result with clear visual
                    st.markdown("### 🎯 Prediction Result")
                    bp = int(test_pred[0]); ap = int(adv_pred[0]); chg = bp != ap

                    if chg:
                        box_cls, icon, title, desc, after_cls = (
                            "changed-box", "🚨",
                            "Prediction Changed After Attack!",
                            "The adversarial perturbation successfully fooled the model into predicting a different label.",
                            "ba-after-chg"
                        )
                    else:
                        box_cls, icon, title, desc, after_cls = (
                            "same-box", "✅",
                            "Prediction Unchanged — Model Robust",
                            "The model resisted the perturbation. Try increasing ε to force a prediction change.",
                            "ba-after-same"
                        )

                    st.markdown(f"""
                    <div class="change-box {box_cls}">
                        <div class="change-icon">{icon}</div>
                        <div class="change-title">{title}</div>
                        <div class="change-desc">{desc}</div>
                        <div class="before-after-row">
                            <div class="ba-card ba-before">
                                <div class="ba-lbl">Original Input</div>
                                <div class="ba-val">{bp}</div>
                            </div>
                            <div class="arrow-sep">→</div>
                            <div class="ba-card {after_cls}">
                                <div class="ba-lbl">Perturbed Input</div>
                                <div class="ba-val">{ap}</div>
                            </div>
                        </div>
                    </div>
                    <div class="obs-box">
                        <span class="badge b-warn">⚠ Observation</span>
                        <p>FGSM perturbs features along the model's gradient direction. Even small ε values
                        cause accuracy drops. Higher ε = stronger perturbation = lower adversarial accuracy.
                        Use adversarial training to improve robustness.</p>
                    </div>
                    """, unsafe_allow_html=True)

                except ImportError as e:
                    st.error(f"Missing library: {e}\n\n👉 pip install adversarial-robustness-toolbox")
                except Exception as e:
                    st.error(f"❌ Error: {e}")


    # =========================================================
    # ☣️ POISONING ATTACK
    # =========================================================
    elif "Poisoning" in attack_type:

        st.markdown('<div class="step-lbl">Step 03 — Upload Dataset</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("CSV", type=["csv"], key="po_csv", label_visibility="collapsed")

        if not uploaded:
            st.markdown('<div style="background:#fff;border-radius:14px;padding:1.5rem;text-align:center;color:#94a3b8;border:1px solid #e2e8f0;">📂 Upload a CSV file to continue</div>', unsafe_allow_html=True)
            st.stop()

        data = pd.read_csv(uploaded)
        total_rows = len(data)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", f"{total_rows:,}")
        col2.metric("Columns", data.shape[1])
        col3.metric("File", uploaded.name)
        with st.expander("👁️ Preview Dataset"):
            st.dataframe(data.head(10), use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="step-lbl">Step 04 — Configure Poison Injection</div>', unsafe_allow_html=True)

        ca, cb = st.columns([2, 1])
        with ca:
            target_col = st.selectbox(
            "🎯 Target Column",
            data.columns.tolist(),
            index=len(data.columns)-1,
            key="po_target_col"
            )   
        with cb:
            poison_rows = st.number_input(
                f"☣️ Rows to Poison (1 – {total_rows})",
                min_value=1, max_value=total_rows, value=min(10, total_rows), step=1
            )

        pct = (int(poison_rows) / total_rows) * 100
        bc  = "#10b981" if pct < 20 else ("#f59e0b" if pct < 50 else "#ef4444")
        sl  = "Low" if pct < 20 else ("Medium" if pct < 50 else "High")
        st.markdown(f"""
        <div style="margin-bottom:0.9rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem;">
                <span style="font-size:0.68rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Poison Severity</span>
                <span class="badge" style="background:{bc}18;color:{bc};border:1px solid {bc}44;">{sl} — {pct:.1f}%</span>
            </div>
            <div style="background:#f1f5f9;border-radius:20px;height:7px;overflow:hidden;">
                <div style="height:100%;width:{min(pct,100):.1f}%;background:{bc};border-radius:20px;"></div>
            </div>
            <div style="font-size:0.82rem;color:#64748b;font-weight:600;margin-top:0.35rem;">
                Injecting <strong>{int(poison_rows)}</strong> label-flipped rows into <strong>{total_rows}</strong> total rows
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("☣️  Run Poisoning Attack", key="run_po"):
            with st.spinner("Training models and injecting poison…"):
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.metrics import accuracy_score

                    feat = [c for c in data.columns if c != target_col]
                    X = data[feat]; y = data[target_col]

                    # Clean model
                    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=42)
                    m_clean = RandomForestClassifier(n_estimators=100, random_state=42)
                    m_clean.fit(X_tr, y_tr)
                    clean_acc = accuracy_score(y_ts, m_clean.predict(X_ts))

                    # Inject poison
                    p_data = data.nlargest(int(poison_rows), feat).copy()
                    p_data[target_col] = 1 - p_data[target_col]
                    pois_ds = pd.concat([data, p_data], ignore_index=True)

                    # Poisoned model
                    Xp = pois_ds[feat]; yp = pois_ds[target_col]
                    Xp_tr, Xp_ts, yp_tr, yp_ts = train_test_split(Xp, yp, test_size=0.3, random_state=42)
                    m_pois = RandomForestClassifier(n_estimators=100, random_state=42)
                    m_pois.fit(Xp_tr, yp_tr)
                    pois_acc = accuracy_score(yp_ts, m_pois.predict(Xp_ts))
                    drop = clean_acc - pois_acc

                    # Store models in session for custom prediction later
                    st.session_state["po_m_clean"]  = m_clean
                    st.session_state["po_m_pois"]   = m_pois
                    st.session_state["po_feat"]      = feat
                    st.session_state["po_pois_ds"]   = pois_ds
                    st.session_state["po_clean_acc"] = clean_acc
                    st.session_state["po_pois_acc"]  = pois_acc
                    st.session_state["po_pct"]       = pct
                    st.session_state["po_n_rows"]    = int(poison_rows)
                    st.session_state["po_data"]      = data

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.stop()

        # ── Show results if models are trained ─────────────────
        if "po_m_clean" in st.session_state:
            m_clean   = st.session_state["po_m_clean"]
            m_pois    = st.session_state["po_m_pois"]
            feat      = st.session_state["po_feat"]
            pois_ds   = st.session_state["po_pois_ds"]
            clean_acc = st.session_state["po_clean_acc"]
            pois_acc  = st.session_state["po_pois_acc"]
            pct       = st.session_state["po_pct"]
            n_rows    = st.session_state["po_n_rows"]
            orig_data = st.session_state["po_data"]
            drop      = clean_acc - pois_acc

            # ── Metrics
            st.markdown("### 📊 Attack Results")
            st.markdown(f"""
            <div class="m-row">
            <div class="m-card mc-v"><div class="m-val">{clean_acc:.4f}</div><div class="m-lbl">Clean Model Acc</div></div>
            <div class="m-card mc-r"><div class="m-val">{pois_acc:.4f}</div><div class="m-lbl">Poisoned Model Acc</div></div>
            <div class="m-card mc-a"><div class="m-val">↓{drop:.4f}</div><div class="m-lbl">Accuracy Drop</div></div>
            <div class="m-card mc-r"><div class="m-val">{pct:.1f}%</div><div class="m-lbl">Poison Rate</div></div>
            </div>""", unsafe_allow_html=True)

            # ── Chart + Datasets
            ch1, ch2 = st.columns([1, 1])
            with ch1:
                fig, ax = plt.subplots(figsize=(4.5, 3.2), facecolor="white")
                ax.set_facecolor("#fafbff")
                bars = ax.bar(["Clean Model","Poisoned Model"], [clean_acc, pois_acc],
                            color=["#4338ca","#ef4444"], width=0.38, edgecolor="white", linewidth=1.5)
                ax.set_ylim(0, 1.12)
                ax.set_ylabel("Accuracy", fontsize=9, color="#64748b")
                ax.set_title("Clean vs Poisoned Model", fontsize=10, fontweight="bold", color="#1a1f36", pad=10)
                ax.tick_params(colors="#64748b", labelsize=9)
                for sp in ax.spines.values(): sp.set_color("#e2e8f0")
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                ax.yaxis.grid(True, color="#f1f5f9", linewidth=1); ax.set_axisbelow(True)
                for bar, val in zip(bars, [clean_acc, pois_acc]):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.025,
                            f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1a1f36")
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            with ch2:
                st.markdown("**🗂️ Datasets**")
                tab1, tab2 = st.tabs([f"✅ Original ({len(orig_data):,} rows)", f"☣️ Poisoned ({len(pois_ds):,} rows)"])
                with tab1:
                    st.dataframe(orig_data, use_container_width=True, height=220)
                with tab2:
                    st.dataframe(pois_ds.tail(n_rows+5), use_container_width=True, height=220)
                    st.caption(f"Showing last {n_rows+5} rows — bottom {n_rows} are poison (flipped labels)")

            # ── Mean Sample Prediction
            st.markdown("### 🔬 Mean Sample Prediction")
            samp = pd.DataFrame([{col: orig_data[col].mean() for col in feat}])
            bp   = int(m_clean.predict(samp)[0])
            ap   = int(m_pois.predict(samp)[0])
            chg  = bp != ap

            if chg:
                box_cls, icon, title, desc, after_cls = (
                    "changed-box","🚨","Prediction Changed Due to Poisoning!",
                    "The poisoned model now predicts a different label for the same input — data integrity compromised.",
                    "ba-after-chg"
                )
            else:
                box_cls, icon, title, desc, after_cls = (
                    "same-box","✅","Prediction Unchanged for This Sample",
                    "The mean input was not affected. Try a custom input below or increase poison rows.",
                    "ba-after-same"
                )

            st.markdown(f"""
            <div class="change-box {box_cls}">
                <div class="change-icon">{icon}</div>
                <div class="change-title">{title}</div>
                <div class="change-desc">{desc}</div>
                <div class="before-after-row">
                    <div class="ba-card ba-before">
                        <div class="ba-lbl">Before Poisoning</div>
                        <div class="ba-val">{bp}</div>
                    </div>
                    <div class="arrow-sep">→</div>
                    <div class="ba-card {after_cls}">
                        <div class="ba-lbl">After Poisoning</div>
                        <div class="ba-val">{ap}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            # ── CUSTOM INPUT PREDICTION ────────────────────────────
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("### ✏️ Test Your Own Input")
            st.markdown('<p style="color:#64748b;font-size:0.9rem;margin-bottom:1rem;">Enter custom values for each feature below and see how poisoning changed the prediction.</p>', unsafe_allow_html=True)

            # Build input grid — 4 columns wide
            user_vals = {}
            cols_per_row = 4
            feat_chunks = [feat[i:i+cols_per_row] for i in range(0, len(feat), cols_per_row)]

            for chunk in feat_chunks:
                grid_cols = st.columns(cols_per_row)
                for i, f in enumerate(chunk):
                    with grid_cols[i]:
                        default_val = float(round(orig_data[f].mean(), 4))
                        user_vals[f] = st.number_input(
                            f,
                            value=default_val,
                            format="%.4f",
                            key=f"usr_{f}",
                        )

            if st.button("🔍  Predict with Custom Input", key="custom_pred"):
                custom_df = pd.DataFrame([user_vals])

                c_pred = int(m_clean.predict(custom_df)[0])
                p_pred = int(m_pois.predict(custom_df)[0])
                chg2   = c_pred != p_pred

                if chg2:
                    b2, i2, t2, d2, ac2 = (
                        "changed-box", "🚨",
                        "⚠️ Prediction Changed After Poisoning!",
                        "Your custom input gets a DIFFERENT prediction from the poisoned model — the attack succeeded.",
                        "ba-after-chg"
                    )
                else:
                    b2, i2, t2, d2, ac2 = (
                        "same-box", "✅",
                        "Prediction Stable for This Input",
                        "Both models agree on this input. Try different values or increase poison rows.",
                        "ba-after-same"
                    )

                st.markdown(f"""
                <div class="change-box {b2}" style="margin-top:1rem;">
                    <div class="change-icon">{i2}</div>
                    <div class="change-title">{t2}</div>
                    <div class="change-desc">{d2}</div>
                    <div class="before-after-row">
                        <div class="ba-card ba-before">
                            <div class="ba-lbl">Clean Model Says</div>
                            <div class="ba-val">{c_pred}</div>
                        </div>
                        <div class="arrow-sep">→</div>
                        <div class="ba-card {ac2}">
                            <div class="ba-lbl">Poisoned Model Says</div>
                            <div class="ba-val">{p_pred}</div>
                        </div>
                    </div>
                </div>
                <div class="obs-box">
                    <span class="badge b-warn">⚠ Observation</span>
                    <p>Label-flipping corrupts the training data by injecting high-impact rows with inverted labels.
                    This shifts the model's decision boundary, causing different predictions for the same inputs.
                    Defend using data sanitization, provenance tracking, and outlier detection before training.</p>
                </div>
                """, unsafe_allow_html=True)
