import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import joblib
import re
from groq import Groq

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="Skill Recommendation System",
    page_icon="🎯",
    layout="wide"
)

# =============================================================================


# =============================================================================
# CUSTOM LAYER — harus didefinisikan sebelum load_model
# =============================================================================
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        attention_weights = tf.nn.softmax(inputs, axis=1)
        context_vector    = attention_weights * inputs
        context_vector    = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def get_config(self):
        return super().get_config()

# =============================================================================
# NORMALISASI JOB TITLE
# =============================================================================
def normalize_job(title: str) -> str:
    """
    Memetakan variasi job title ke kategori standar.
    Contoh: 'Senior Data Scientist Intern' → 'data scientist'
    """
    t = title.lower().strip()

    if re.search(r'data scien',                          t): return 'data scientist'
    if re.search(r'data analy',                          t): return 'data analyst'
    if re.search(r'data engineer',                       t): return 'data engineer'
    if re.search(r'ai engineer|machine learning engineer', t): return 'ai engineer'
    if re.search(r'devops|dev ops|site reliability|\bsre\b', t): return 'devops engineer'
    if re.search(r'full.?stack|fullstack',               t): return 'fullstack developer'
    if re.search(r'backend|back.end',                    t): return 'backend developer'
    if re.search(r'frontend|front.end',                  t): return 'frontend developer'
    if re.search(r'mobile|android|\bios\b',              t): return 'mobile developer'
    if re.search(r'software dev|software eng',           t): return 'software developer'
    if re.search(r'\bqa\b|quality assur|tester',         t): return 'qa engineer'
    if re.search(r'cloud|infrastructure',                t): return 'cloud engineer'
    if re.search(r'security|cybersec',                   t): return 'security engineer'
    if re.search(r'product manager|\bpm\b',              t): return 'product manager'
    if re.search(r'project manager',                     t): return 'project manager'
    if re.search(r'data warehouse|\betl\b|bi developer|business intel', t): return 'bi developer'
    if re.search(r'it support|helpdesk|help desk',       t): return 'it support'
    if re.search(r'network|\bnoc\b',                     t): return 'network engineer'
    if re.search(r'programmer|developer',                t): return 'software developer'
    if re.search(r'lecturer|professor|instructor',       t): return 'lecturer'

    cleaned = re.sub(r'[^a-zA-Z\s]', '', t)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# =============================================================================
# LOAD MODEL & ARTIFACTS
# =============================================================================
@st.cache_resource
def load_all_assets():
    try:
        model = tf.keras.models.load_model(
            'skill_recommendation_model_v3.keras',
            compile=False 
        )
        mlb = joblib.load('mlb_v3.pkl')
        return model, mlb
    except Exception as e:
        st.error(f'Gagal memuat model: {e}')
        return None, None

model, mlb = load_all_assets()

# =============================================================================
# FUNGSI PREDIKSI
# =============================================================================
def predict_skills(job_title: str, top_n: int = 8, threshold: float = 0.2):
    """
    Prediksi skill dari job title.

    Alur:
    1. Normalisasi job title (kunci utama perbaikan)
    2. Masukkan ke model
    3. Ambil top-N skill di atas threshold

    Return: list of (skill_name, confidence_score)
    """
    # Step 1: normalisasi — ini yang membuat prediksi jadi berbeda per job
    job_normalized = normalize_job(job_title)

    # Step 2: prediksi
    input_tensor = tf.constant([job_normalized])
    pred         = model.predict(input_tensor, verbose=0)[0]  # shape: (n_skills,)

    # Step 3: ambil top-N di atas threshold
    top_indices = np.argsort(pred)[::-1][:top_n]
    results = [
        (mlb.classes_[i], float(pred[i]))
        for i in top_indices
        if pred[i] >= threshold
    ]

    if not results:
        top5 = np.argsort(pred)[::-1][:5]
        results = [(mlb.classes_[i], float(pred[i])) for i in top5]

    return results, job_normalized

# =============================================================================
# FUNGSI GROQ — roadmap karier
# =============================================================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

@st.cache_data(show_spinner=False)
def get_groq_roadmap(user_goal: str, skills_string: str) -> str:
    prompt = f"""
Saya ingin menjadi {user_goal}.

Berdasarkan analisis data lowongan kerja, skill yang paling banyak diminta untuk posisi ini adalah:
{skills_string}

Berikan dalam format Markdown yang rapi:
1. **Roadmap Belajar Step-by-Step** (urut dari skill paling dasar)
2. **Skill Tambahan** yang sering dibutuhkan tapi belum ada di list
3. **Rekomendasi Sertifikasi** yang relevan dan diakui industri
4. **Estimasi Waktu** belajar per skill
5. **Saran Karier** (jalur dari posisi entry hingga senior)

Gunakan Bahasa Indonesia yang profesional.
"""
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Gagal mendapatkan roadmap: {str(e)}"

# =============================================================================
# TAMPILAN UI
# =============================================================================
st.title("Skill Recommendation System")
st.caption("Masukkan nama pekerjaan impian kamu, sistem akan merekomendasikan skill yang dibutuhkan.")

# --- Input ---
user_input = st.text_area(
    "Masukkan nama pekerjaan impian kamu:",
    placeholder="Contoh: Data Scientist, AI Engineer, DevOps Engineer...",
    height=120
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    run = st.button("🔍 Prediksi Skill & Rekomendasi", use_container_width=True)


if run:
    if not user_input.strip():
        st.warning("⚠️ Input tidak boleh kosong.")
    elif model is None or mlb is None:
        st.error("❌ Model gagal dimuat. Periksa file model di folder yang sama dengan app.py.")
    else:
        results, job_normalized = predict_skills(user_input, top_n=8, threshold=0.2)
        skills_string = ", ".join([s for s, _ in results])

        # Info normalisasi
        if job_normalized != user_input.lower().strip():
            st.info(
                f"Input kamu **\"{user_input}\"** dikenali sebagai kategori: "
                f"**\"{job_normalized}\"**"
            )

        col1, col2 = st.columns([1, 1.4])

        with col1:
            st.subheader("Rekomendasi Skill")
            st.write("Skill yang direkomendasikan:")
            st.markdown("---")

            for skill, score in results:
                label_col, bar_col = st.columns([1.2, 2])
                with label_col:
                    st.markdown(f"**{skill}**")
                    st.caption(f"skor: {score:.2f}")
                with bar_col:
                    st.progress(min(score, 1.0))
                st.markdown("")

        with col2:
            st.subheader("Roadmap & Saran Karier (AI)")
            with st.spinner("Sedang menyusun roadmap belajar..."):
                roadmap = get_groq_roadmap(user_input, skills_string)
            st.markdown(roadmap)

# =============================================================================
# FOOTER DEBUG MODE
# =============================================================================
with st.expander("Debug: Test prediksi langsung (tanpa Groq)", expanded=False):
    st.caption("Gunakan bagian ini untuk memastikan model berjalan dengan benar.")
    debug_input = st.text_input("Ketik nama job untuk debug:", placeholder="data scientist")
    if st.button("Test Prediksi"):
        if debug_input.strip() and model is not None:
            res, norm = predict_skills(debug_input, top_n=10, threshold=0.0)
            st.write(f"**Normalized:** `{norm}`")
            st.dataframe(
                {"Skill": [r[0] for r in res], "Score": [round(r[1], 4) for r in res]},
                use_container_width=True
            )
        else:
            st.warning("Masukkan nama job terlebih dahulu.")
