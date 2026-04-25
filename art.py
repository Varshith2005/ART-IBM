import streamlit as st
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

/* ── BLUE TEAM SPECIFIC STYLES ── */
.blue-header {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 55%, #60a5fa 100%);
    border-radius: 18px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.blue-header h2 {
    color: #fff;
    margin: 0 0 0.3rem;
    font-weight: 700;
}
.blue-header p {
    color: rgba(255,255,255,0.8);
    margin: 0;
}
.defense-card {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border-left: 4px solid #2563eb;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.defense-badge {
    background: #2563eb20;
    color: #1e40af;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 0.5rem;
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


# =========================================================
# CORRECTED CUSTOM FGSM ATTACK
# =========================================================
def custom_fgsm_attack(model, X, y, eps, scaler=None):
    """Corrected FGSM attack implementation for logistic regression"""
    try:
        coef = model.coef_.flatten()
        intercept = model.intercept_[0]
        
        X_adv = X.copy()
        
        for i in range(X.shape[0]):
            logits = np.dot(X[i], coef) + intercept
            prob = 1 / (1 + np.exp(-logits))
            
            if y[i] == 1:
                grad = -(1 - prob) * coef
            else:
                grad = prob * coef
            
            perturbation = eps * np.sign(grad)
            X_adv[i] = X[i] + perturbation
        
        return X_adv
    except Exception as e:
        st.error(f"FGSM attack failed: {e}")
        return X

# =========================================================
# STEP 1 — TEAM SELECTION
# =========================================================
st.markdown('<div class="step-lbl">Step 01 — Select Team Mode</div>', unsafe_allow_html=True)

if "team_mode" not in st.session_state:
    st.session_state.team_mode = "🔴  Red Teaming  —  Attack & Exploit"

team = st.radio(
    "Select Team Mode", 
    ["🔴  Red Teaming  —  Attack & Exploit", 
     "🔵  Blue Teaming  —  Defend & Detect"],
    label_visibility="collapsed",
    horizontal=True,
    index=0 if st.session_state.team_mode.startswith("🔴") else 1,
    key="team_selection_main"
)

st.session_state.team_mode = team
is_red = team.startswith("🔴")
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# =========================================================
# STEP 2 — ATTACK TYPE (Only for Red Team)
# =========================================================
attack_type = None

if is_red:
    st.markdown('<div class="step-lbl">Step 02 — Select Attack Type</div>', unsafe_allow_html=True)
    
    if "attack_type_selected" not in st.session_state:
        st.session_state.attack_type_selected = "— Choose an attack —"
    
    attack_type = st.selectbox(
        "Attack Type",
        ["— Choose an attack —",
         "⚡ Evasion Attack  (Fast Gradient Sign Method)",
         "☣️ Poisoning Attack  (Label Flipping)"],
        label_visibility="collapsed",
        key="attack_type_main"
    )
    
    st.session_state.attack_type_selected = attack_type
    
    if attack_type.startswith("—"):
        st.markdown(
            '<div style="background:#fff;border-radius:14px;padding:2rem;text-align:center;color:#94a3b8;border:1px solid #e2e8f0;">👆 Choose an attack type above to get started</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

# =========================================================
# EVASION ATTACK
# =========================================================
if is_red and attack_type and "Evasion" in attack_type:
    st.markdown('<div class="step-lbl">Step 03 — Upload Dataset</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload CSV file", 
        type=["csv"], 
        key="evasion_file_uploader",
        label_visibility="collapsed"
    )
    
    if uploaded_file is None:
        st.markdown(
            '<div style="background:#fff;border-radius:14px;padding:2rem;text-align:center;color:#94a3b8;border:1px solid #e2e8f0;">📂 Please upload a CSV file to continue</div>',
            unsafe_allow_html=True
        )
        st.stop()
    
    data = pd.read_csv(uploaded_file)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{data.shape[0]:,}")
    col2.metric("Total Columns", data.shape[1])
    col3.metric("File Name", uploaded_file.name)
    
    with st.expander("📊 Preview Dataset"):
        st.dataframe(data.head(10), use_container_width=True)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="step-lbl">Step 04 — Configure Attack</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        target_column = st.selectbox(
            "🎯 Target Column",
            data.columns.tolist(),
            index=len(data.columns)-1,
            key="evasion_target"
        )
    
    with col2:
        epsilon = st.slider(
            "⚡ Attack Strength (ε)", 
            0.0, 1.0, 0.3, 0.01,
            key="evasion_epsilon"
        )
    
    if epsilon == 0:
        strength_color = "#94a3b8"
        strength_label = "No Attack"
    elif epsilon < 0.1:
        strength_color = "#10b981"
        strength_label = "Very Low"
    elif epsilon < 0.2:
        strength_color = "#10b981"
        strength_label = "Low"
    elif epsilon < 0.4:
        strength_color = "#f59e0b"
        strength_label = "Medium"
    else:
        strength_color = "#ef4444"
        strength_label = "High"
    
    st.markdown(
        f'<div style="margin-bottom:1rem;"><span class="badge" style="background:{strength_color}18;color:{strength_color};">{strength_label} Strength — ε = {epsilon:.2f}</span></div>',
        unsafe_allow_html=True
    )
    
    if st.button("⚡ Run Evasion Attack", key="run_evasion_button"):
        with st.spinner("Training model and generating adversarial examples..."):
            try:
                feature_cols = [c for c in data.columns if c != target_column]
                X = data[feature_cols].values.astype(np.float32)
                y = data[target_column].values
                
                X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
                val_acc = accuracy_score(y_val, model.predict(X_val_scaled))
                test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
                
                X_test_adv = X_test_scaled.copy()
                coef = model.coef_.flatten()
                intercept = model.intercept_[0]
                
                for i in range(X_test_scaled.shape[0]):
                    logits = np.dot(X_test_scaled[i], coef) + intercept
                    prob = 1 / (1 + np.exp(-logits))
                    
                    if y_test[i] == 1:
                        grad = (1 - prob) * coef
                    else:
                        grad = prob * coef
                    
                    perturbation = epsilon * np.sign(grad)
                    X_test_adv[i] = X_test_scaled[i] + perturbation
                
                adv_pred = model.predict(X_test_adv)
                adv_acc = accuracy_score(y_test, adv_pred)
                accuracy_drop = test_acc - adv_acc
                
                st.markdown("### 📊 Attack Results")
                
                st.markdown(f"""
                <div class="m-row">
                    <div class="m-card mc-v">
                        <div class="m-val">{train_acc:.4f}</div>
                        <div class="m-lbl">Clean Train Acc</div>
                    </div>
                    <div class="m-card mc-g">
                        <div class="m-val">{val_acc:.4f}</div>
                        <div class="m-lbl">Clean Val Acc</div>
                    </div>
                    <div class="m-card mc-v">
                        <div class="m-val">{test_acc:.4f}</div>
                        <div class="m-lbl">Clean Test Acc</div>
                    </div>
                    <div class="m-card {"mc-r" if adv_acc < test_acc else "mc-a"}">
                        <div class="m-val">{adv_acc:.4f}</div>
                        <div class="m-lbl">Adversarial Acc</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(
                    ["Clean Train", "Clean Val", "Clean Test", "Adversarial"],
                    [train_acc, val_acc, test_acc, adv_acc],
                    color=["#10b981", "#3b82f6", "#6366f1", "#ef4444"]
                )
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("Accuracy", fontsize=12)
                ax.set_title(f"Impact of FGSM Attack (ε = {epsilon:.2f})", fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3, axis='y')
                
                for bar, val in zip(bars, [train_acc, val_acc, test_acc, adv_acc]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
                
                st.pyplot(fig)
                plt.close()
                
                if test_acc > 0:
                    attack_success = (accuracy_drop / test_acc) * 100
                    st.success(f"🎯 **Attack Impact:** Accuracy dropped by {accuracy_drop:.1%} ({attack_success:.1f}% relative decrease)")
                
                st.markdown("### 🎯 Sample Predictions (First 5 Test Samples)")
                
                num_samples = min(5, len(X_test_scaled))
                clean_predictions = []
                
                for i in range(num_samples):
                    sample_2d = X_test_scaled[i].reshape(1, -1)
                    clean_predictions.append(int(model.predict(sample_2d)[0]))
                
                sample_df = pd.DataFrame({
                    "Sample ID": list(range(1, num_samples + 1)),
                    "True Label": [int(y_test[i]) for i in range(num_samples)],
                    "Clean Prediction": clean_predictions,
                    "Adversarial Prediction": [int(adv_pred[i]) for i in range(num_samples)],
                    "Changed?": ["✅" if clean_predictions[i] != int(adv_pred[i]) else "❌" for i in range(num_samples)]
                })
                st.dataframe(sample_df, use_container_width=True)
                
                if num_samples > 0:
                    st.markdown("### 🔍 Detailed Sample Analysis (First Test Sample)")
                    
                    sample_idx = 0
                    first_sample_2d = X_test_scaled[sample_idx].reshape(1, -1)
                    orig_pred = int(model.predict(first_sample_2d)[0])
                    adv_pred_sample = int(adv_pred[sample_idx])
                    
                    if orig_pred != adv_pred_sample:
                        st.markdown(f"""
                        <div class="change-box changed-box">
                            <div class="change-icon">🚨</div>
                            <div class="change-title">Prediction Changed After Attack!</div>
                            <div class="change-desc">The adversarial perturbation successfully fooled the model.</div>
                            <div class="before-after-row">
                                <div class="ba-card ba-before">
                                    <div class="ba-lbl">Original Prediction</div>
                                    <div class="ba-val">{orig_pred}</div>
                                </div>
                                <div class="arrow-sep">→</div>
                                <div class="ba-card ba-after-chg">
                                    <div class="ba-lbl">After Attack</div>
                                    <div class="ba-val">{adv_pred_sample}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="change-box same-box">
                            <div class="change-icon">⚠️</div>
                            <div class="change-title">Prediction Maintained</div>
                            <div class="change-desc">This sample resisted the attack. Try increasing ε for stronger effect.</div>
                            <div class="before-after-row">
                                <div class="ba-card ba-before">
                                    <div class="ba-lbl">Prediction</div>
                                    <div class="ba-val">{orig_pred}</div>
                                </div>
                                <div class="arrow-sep">→</div>
                                <div class="ba-card ba-after-same">
                                    <div class="ba-lbl">After Attack</div>
                                    <div class="ba-val">{adv_pred_sample}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if len(feature_cols) > 0:
                        st.markdown("### 📊 Feature Perturbation (First Sample - First 10 Features)")
                        num_features = min(10, len(feature_cols))
                        pert_data = {
                            "Feature": feature_cols[:num_features],
                            "Original Value": [f"{X_test_scaled[sample_idx, i]:.4f}" for i in range(num_features)],
                            "Adversarial Value": [f"{X_test_adv[sample_idx, i]:.4f}" for i in range(num_features)],
                            "Perturbation": [f"{X_test_adv[sample_idx, i] - X_test_scaled[sample_idx, i]:+.4f}" for i in range(num_features)]
                        }
                        pert_df = pd.DataFrame(pert_data)
                        st.dataframe(pert_df, use_container_width=True)
                
                severity = "high" if accuracy_drop > 0.2 else "moderate" if accuracy_drop > 0.1 else "low"
                st.markdown(f"""
                <div class="obs-box">
                    <span class="badge b-warn">⚠️ Security Observation</span>
                    <p><strong>FGSM Attack Analysis:</strong> With attack strength ε = {epsilon:.2f}, the model's accuracy dropped 
                    from <strong>{test_acc:.1%}</strong> to <strong>{adv_acc:.1%}</strong> — a decrease of <strong>{accuracy_drop:.1%}</strong>. 
                    This demonstrates <strong>{severity}</strong> vulnerability to gradient-based attacks.</p>
                    <p><strong>Recommendation:</strong> 
                    • Implement adversarial training with FGSM examples<br>
                    • Use defensive distillation or gradient masking<br>
                    • Consider ensemble methods for improved robustness<br>
                    • Apply input preprocessing and feature squeezing</p>
                </div>
                """, unsafe_allow_html=True)
                
                total_changed = sum([1 for i in range(num_samples) if clean_predictions[i] != int(adv_pred[i])])
                if num_samples > 0:
                    st.info(f"📊 **Quick Summary:** Out of {num_samples} shown samples, {total_changed} ({total_changed/num_samples:.0%}) had their predictions flipped by the attack.")
                
            except Exception as e:
                st.error(f"Error during attack simulation: {str(e)}")
                st.exception(e)

# =========================================================
# POISONING ATTACK
# =========================================================
elif is_red and attack_type and "Poisoning" in attack_type:

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
                X = data[feat]
                y = data[target_col]

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
                Xp = pois_ds[feat]
                yp = pois_ds[target_col]
                Xp_tr, Xp_ts, yp_tr, yp_ts = train_test_split(Xp, yp, test_size=0.3, random_state=42)
                m_pois = RandomForestClassifier(n_estimators=100, random_state=42)
                m_pois.fit(Xp_tr, yp_tr)
                pois_acc = accuracy_score(yp_ts, m_pois.predict(Xp_ts))
                drop = clean_acc - pois_acc

                # Store models in session
                st.session_state["po_m_clean"]  = m_clean
                st.session_state["po_m_pois"]   = m_pois
                st.session_state["po_feat"]      = feat
                st.session_state["po_pois_ds"]   = pois_ds
                st.session_state["po_clean_acc"] = clean_acc
                st.session_state["po_pois_acc"]  = pois_acc
                st.session_state["po_pct"]       = pct
                st.session_state["po_n_rows"]    = int(poison_rows)
                st.session_state["po_data"]      = data

                # Display results
                st.markdown("### 📊 Attack Results")
                st.markdown(f"""
                <div class="m-row">
                    <div class="m-card mc-v">
                        <div class="m-val">{clean_acc:.4f}</div>
                        <div class="m-lbl">Clean Model Acc</div>
                    </div>
                    <div class="m-card mc-r">
                        <div class="m-val">{pois_acc:.4f}</div>
                        <div class="m-lbl">Poisoned Model Acc</div>
                    </div>
                    <div class="m-card mc-a">
                        <div class="m-val">↓{drop:.4f}</div>
                        <div class="m-lbl">Accuracy Drop</div>
                    </div>
                    <div class="m-card mc-r">
                        <div class="m-val">{pct:.1f}%</div>
                        <div class="m-lbl">Poison Rate</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(["Clean Model", "Poisoned Model"], [clean_acc, pois_acc],
                            color=["#4338ca", "#ef4444"])
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("Accuracy")
                ax.set_title("Clean vs Poisoned Model Performance")
                for bar, val in zip(bars, [clean_acc, pois_acc]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f"{val:.4f}", ha="center", fontweight="bold")
                st.pyplot(fig)
                plt.close()

                st.markdown(f"""
                <div class="obs-box">
                    <span class="badge b-warn">☣️ Poisoning Attack Analysis</span>
                    <p><strong>Impact:</strong> Injecting {int(poison_rows)} poisoned rows ({pct:.1f}% of original data) 
                    reduced model accuracy from <strong>{clean_acc:.1%}</strong> to <strong>{pois_acc:.1%}</strong> 
                    — a drop of <strong>{drop:.1%}</strong>.</p>
                    <p><strong>Defense Recommendations:</strong><br>
                    • Implement data provenance and validation pipelines<br>
                    • Use outlier detection to identify poisoned samples<br>
                    • Employ robust aggregation methods (e.g., trimmed mean)<br>
                    • Regular data auditing and anomaly detection</p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)

    # Show results if models are trained
    if "po_m_clean" in st.session_state:
        m_clean   = st.session_state["po_m_clean"]
        m_pois    = st.session_state["po_m_pois"]
        feat      = st.session_state["po_feat"]
        clean_acc = st.session_state["po_clean_acc"]
        pois_acc  = st.session_state["po_pois_acc"]
        pct       = st.session_state["po_pct"]
        orig_data = st.session_state["po_data"]
        drop      = clean_acc - pois_acc

        st.markdown("### 🔬 Test Custom Input")
        st.markdown('<p style="color:#64748b;font-size:0.9rem;margin-bottom:1rem;">Enter custom values to see how poisoning changed predictions.</p>', unsafe_allow_html=True)

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
                        key=f"usr_po_{f}",
                    )

        if st.button("🔍 Predict with Custom Input", key="custom_pred_po"):
            custom_df = pd.DataFrame([user_vals])
            c_pred = int(m_clean.predict(custom_df)[0])
            p_pred = int(m_pois.predict(custom_df)[0])
            chg2 = c_pred != p_pred

            if chg2:
                box_cls, icon, title, after_cls = "changed-box", "🚨", "⚠️ Prediction Changed After Poisoning!", "ba-after-chg"
            else:
                box_cls, icon, title, after_cls = "same-box", "✅", "Prediction Stable for This Input", "ba-after-same"

            st.markdown(f"""
            <div class="change-box {box_cls}" style="margin-top:1rem;">
                <div class="change-icon">{icon}</div>
                <div class="change-title">{title}</div>
                <div class="before-after-row">
                    <div class="ba-card ba-before">
                        <div class="ba-lbl">Clean Model</div>
                        <div class="ba-val">{c_pred}</div>
                    </div>
                    <div class="arrow-sep">→</div>
                    <div class="ba-card {after_cls}">
                        <div class="ba-lbl">Poisoned Model</div>
                        <div class="ba-val">{p_pred}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# BLUE TEAM MODULE
# =========================================================
if not is_red:
    st.markdown("""
    <div class="blue-header">
        <h2>🔵 Blue Team — Defend & Detect</h2>
        <p>Implement adversarial defenses, monitor model behavior, and harden ML systems</p>
    </div>
    """, unsafe_allow_html=True)
    
    bt_tab1, bt_tab2, bt_tab3 = st.tabs([
        "🛡️ Adversarial Training",
        "🔍 Input Defenses",
        "📊 Model Hardening"
    ])
    
    # Tab 1: Adversarial Training
    with bt_tab1:
        st.markdown("### Adversarial Training Defense")
        st.markdown('<div class="step-lbl">Augment training data with adversarial examples</div>', unsafe_allow_html=True)
        
        adv_train_file = st.file_uploader("📂 Upload Dataset", type=["csv"], key="blue_adv_train")
        
        if adv_train_file is not None:
            df_adv = pd.read_csv(adv_train_file)
            col1, col2 = st.columns(2)
            col1.metric("Total Samples", f"{len(df_adv):,}")
            col2.metric("Features", df_adv.shape[1] - 1)
            
            target_adv = st.selectbox("🎯 Target Column", df_adv.columns.tolist(), key="blue_adv_target")
            adv_eps = st.slider("Attack Strength (ε)", 0.0, 0.5, 0.1, 0.01, key="blue_adv_eps")
            poison_ratio = st.slider("Adversarial Augmentation Ratio", 0.0, 1.0, 0.3, 0.05, key="blue_adv_ratio")
            
            if st.button("🚀 Train Robust Model", key="blue_adv_train_btn"):
                with st.spinner("Training adversarially robust model..."):
                    try:
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.model_selection import train_test_split
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.metrics import accuracy_score
                        
                        X_cols = [c for c in df_adv.columns if c != target_adv]
                        X = df_adv[X_cols].values
                        y = df_adv[target_adv].values
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Baseline model
                        baseline = LogisticRegression(max_iter=1000, random_state=42)
                        baseline.fit(X_train_scaled, y_train)
                        baseline_acc = accuracy_score(y_test, baseline.predict(X_test_scaled))
                        
                        # Generate adversarial examples
                        X_train_adv = X_train_scaled.copy()
                        coef = baseline.coef_.flatten()
                        intercept = baseline.intercept_[0]
                        
                        num_adv = int(len(X_train_scaled) * poison_ratio)
                        adv_indices = np.random.choice(len(X_train_scaled), num_adv, replace=False)
                        
                        for idx in adv_indices:
                            logits = np.dot(X_train_scaled[idx], coef) + intercept
                            prob = 1 / (1 + np.exp(-logits))
                            if y_train[idx] == 1:
                                grad = -(1 - prob) * coef
                            else:
                                grad = prob * coef
                            X_train_adv[idx] = X_train_scaled[idx] + adv_eps * np.sign(grad)
                        
                        # Train robust model
                        X_combined = np.vstack([X_train_scaled, X_train_adv])
                        y_combined = np.hstack([y_train, y_train])
                        robust_model = LogisticRegression(max_iter=1000, random_state=42)
                        robust_model.fit(X_combined, y_combined)
                        
                        # Generate adversarial test set
                        X_test_adv = X_test_scaled.copy()
                        for i in range(len(X_test_scaled)):
                            logits = np.dot(X_test_scaled[i], coef) + intercept
                            prob = 1 / (1 + np.exp(-logits))
                            if y_test[i] == 1:
                                grad = -(1 - prob) * coef
                            else:
                                grad = prob * coef
                            X_test_adv[i] = X_test_scaled[i] + adv_eps * np.sign(grad)
                        
                        baseline_adv_acc = accuracy_score(y_test, baseline.predict(X_test_adv))
                        robust_adv_acc = accuracy_score(y_test, robust_model.predict(X_test_adv))
                        robust_clean_acc = accuracy_score(y_test, robust_model.predict(X_test_scaled))
                        
                        st.markdown("### 📊 Results")
                        st.markdown(f"""
                        <div class="m-row">
                            <div class="m-card mc-v"><div class="m-val">{baseline_acc:.4f}</div><div class="m-lbl">Baseline (Clean)</div></div>
                            <div class="m-card mc-r"><div class="m-val">{baseline_adv_acc:.4f}</div><div class="m-lbl">Baseline (Adv)</div></div>
                            <div class="m-card mc-g"><div class="m-val">{robust_clean_acc:.4f}</div><div class="m-lbl">Robust (Clean)</div></div>
                            <div class="m-card mc-v"><div class="m-val">{robust_adv_acc:.4f}</div><div class="m-lbl">Robust (Adv)</div></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        fig, ax = plt.subplots(figsize=(8, 5))
                        categories = ['Baseline', 'Adversarially Trained']
                        clean_scores = [baseline_acc, robust_clean_acc]
                        adv_scores = [baseline_adv_acc, robust_adv_acc]
                        x = np.arange(len(categories))
                        width = 0.35
                        ax.bar(x - width/2, clean_scores, width, label='Clean Test', color='#3b82f6')
                        ax.bar(x + width/2, adv_scores, width, label='Adversarial Test', color='#ef4444')
                        ax.set_ylabel('Accuracy')
                        ax.set_title(f'Adversarial Training Defense (ε={adv_eps:.2f})')
                        ax.set_xticks(x)
                        ax.set_xticklabels(categories)
                        ax.legend()
                        ax.set_ylim(0, 1.1)
                        st.pyplot(fig)
                        plt.close()
                        
                        improvement = robust_adv_acc - baseline_adv_acc
                        st.success(f"🎯 **Defense Success:** Robustness improved by **{improvement:.1%}**")
                        
                        st.markdown(f"""
                        <div class="obs-box">
                            <span class="badge b-info">🛡️ Blue Team Observation</span>
                            <p>Adversarial training improved robustness from <strong>{baseline_adv_acc:.1%}</strong> to <strong>{robust_adv_acc:.1%}</strong> 
                            against ε={adv_eps:.2f} attacks — an improvement of <strong>{improvement:.1%}</strong>.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("📂 Upload a dataset to begin adversarial training")
    
    # Tab 2: Input Defenses
    with bt_tab2:
        st.markdown("### Input Preprocessing Defenses")
        st.markdown('<div class="step-lbl">Apply preprocessing to neutralize adversarial perturbations</div>', unsafe_allow_html=True)
        
        defense_type = st.selectbox("🛡️ Select Defense", ["Feature Squeezing", "Gaussian Smoothing"], key="blue_defense_type")
        defense_file = st.file_uploader("📂 Upload Dataset", type=["csv"], key="blue_defense_file")
        
        if defense_file is not None:
            df_def = pd.read_csv(defense_file)
            target_def = st.selectbox("🎯 Target Column", df_def.columns.tolist(), key="blue_def_target")
            test_eps = st.slider("Test Attack Strength", 0.0, 1.0, 0.2, 0.01, key="blue_test_eps")
            
            if st.button("🛡️ Evaluate Defense", key="blue_def_eval"):
                with st.spinner("Evaluating defense..."):
                    try:
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.model_selection import train_test_split
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.metrics import accuracy_score
                        
                        X_cols = [c for c in df_def.columns if c != target_def]
                        X = df_def[X_cols].values
                        y = df_def[target_def].values
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X_train_scaled, y_train)
                        clean_acc = accuracy_score(y_test, model.predict(X_test_scaled))
                        
                        X_adv = X_test_scaled + test_eps * np.random.normal(0, 1, X_test_scaled.shape)
                        adv_before = accuracy_score(y_test, model.predict(X_adv))
                        
                        if "Feature Squeezing" in defense_type:
                            X_adv_defended = np.round(X_adv, decimals=2)
                        else:
                            X_adv_defended = X_adv + np.random.normal(0, 0.05, X_adv.shape)
                        
                        adv_after = accuracy_score(y_test, model.predict(X_adv_defended))
                        improvement = adv_after - adv_before
                        
                        st.markdown(f"""
                        <div class="m-row">
                            <div class="m-card mc-v"><div class="m-val">{clean_acc:.4f}</div><div class="m-lbl">Clean Accuracy</div></div>
                            <div class="m-card mc-r"><div class="m-val">{adv_before:.4f}</div><div class="m-lbl">Before Defense</div></div>
                            <div class="m-card mc-g"><div class="m-val">{adv_after:.4f}</div><div class="m-lbl">After Defense</div></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success(f"🛡️ Defense improved accuracy by **{improvement:.1%}** against ε={test_eps:.2f} attacks")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("📂 Upload a dataset to evaluate defenses")
    
    # Tab 3: Model Hardening
    with bt_tab3:
        st.markdown("### Ensemble Hardening")
        st.markdown('<div class="step-lbl">Combine multiple models for robust defense</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="defense-card">
            <div class="defense-badge">🏆 RECOMMENDED APPROACH</div>
            <strong>Ensemble methods</strong> provide inherent robustness by requiring attackers to fool multiple models simultaneously.
        </div>
        """, unsafe_allow_html=True)
        
        ensemble_models = st.multiselect("Select Models for Ensemble", 
                                         ["Logistic Regression", "Random Forest", "Gradient Boosting"],
                                         default=["Logistic Regression"], key="blue_ensemble")
        
        ensemble_file = st.file_uploader("📂 Upload Dataset", type=["csv"], key="blue_ensemble_file")
        
        if ensemble_file is not None and len(ensemble_models) > 0:
            df_ens = pd.read_csv(ensemble_file)
            target_ens = st.selectbox("🎯 Target Column", df_ens.columns.tolist(), key="blue_ens_target")
            
            if st.button("🔒 Build Ensemble", key="blue_ens_btn"):
                with st.spinner("Building ensemble model..."):
                    try:
                        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.model_selection import train_test_split
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.metrics import accuracy_score
                        
                        X_cols = [c for c in df_ens.columns if c != target_ens]
                        X = df_ens[X_cols].values
                        y = df_ens[target_ens].values
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        models = []
                        if "Logistic Regression" in ensemble_models:
                            lr = LogisticRegression(max_iter=1000, random_state=42)
                            lr.fit(X_train_scaled, y_train)
                            models.append(lr)
                        if "Random Forest" in ensemble_models:
                            rf = RandomForestClassifier(n_estimators=100, random_state=42)
                            rf.fit(X_train_scaled, y_train)
                            models.append(rf)
                        if "Gradient Boosting" in ensemble_models:
                            gb = GradientBoostingClassifier(random_state=42)
                            gb.fit(X_train_scaled, y_train)
                            models.append(gb)
                        
                        # Ensemble prediction by voting
                        individual_preds = [model.predict(X_test_scaled) for model in models]
                        ensemble_preds = np.round(np.mean(individual_preds, axis=0)).astype(int)
                        ensemble_acc = accuracy_score(y_test, ensemble_preds)
                        
                        individual_accs = [accuracy_score(y_test, pred) for pred in individual_preds]
                        
                        st.markdown("### 📊 Ensemble Results")
                        acc_cols = st.columns(len(models) + 1)
                        for i, (name, acc) in enumerate(zip(ensemble_models, individual_accs)):
                            with acc_cols[i]:
                                st.metric(name, f"{acc:.3f}")
                        with acc_cols[-1]:
                            st.metric("Ensemble", f"{ensemble_acc:.3f}", 
                                     delta=f"+{ensemble_acc - np.mean(individual_accs):.3f}")
                        
                        st.success(f"🎯 Ensemble achieved {ensemble_acc:.1%} accuracy, outperforming individual models!")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("📂 Select models and upload a dataset to build an ensemble")
