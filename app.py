import streamlit as st
from utils import pdf_bytes_to_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_missing_skills(resume_text: str, jd_text: str, skills: list[str]) -> list[str]:
    r = resume_text.lower()
    j = jd_text.lower()
    needed = [s for s in skills if s in j]
    missing = [s for s in needed if s not in r]
    return missing

with open("skills_seed.txt", "r", encoding="utf-8") as f:
    SKILLS = [line.strip() for line in f if line.strip()]


st.set_page_config(page_title="AI Resume & Interview Helper", page_icon="🧠")
st.title("🧠 AI Resume Analyzer & Interview Prep")

# lazy-load model once
@st.cache_resource
def get_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

tab1, tab2 = st.tabs(["Resume Analyzer", "Interview Prep"])
with tab1:
   with tab1:
   
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    jd_text = st.text_area("Paste Job Description")

    if st.button("Compute Match"):
        if not resume_file or not jd_text.strip():
            st.warning("Please upload a resume PDF and paste a JD.")
        else:
            # extract text from resume
            resume_text = pdf_bytes_to_text(resume_file.read())

            # load model & compute similarity
            model = get_model()
            emb = model.encode([resume_text, jd_text])
            score = float(cosine_similarity([emb[0]], [emb[1]])[0, 0])

            # show match score
            st.metric("Semantic Match (0-1)", f"{score:.2f}")

            # 👉 check for missing skills
            missing = find_missing_skills(resume_text, jd_text, SKILLS)
            st.subheader("Suggested skills to add")
            st.write(", ".join(missing) if missing else "Great coverage!")



   import json

with tab2:
    st.subheader("Pick a topic to practice")
    topics = ["python","ml"]
    sel = st.selectbox("Topic", topics)

    # load bank
    bank = json.load(open("qa_bank.json","r",encoding="utf-8"))
    cand = [q for q in bank if q["topic"] == sel][0]
    st.write("**Question:**", cand["question"])

    user_answer = st.text_area("Your answer (short paragraph)")
    if st.button("Get Feedback"):
        if not user_answer.strip():
            st.warning("Type your answer first.")
        else:
            model = get_model()
            # compare user answer to reference
            emb = model.encode([user_answer, cand["answer"]])
            sim = float(cosine_similarity([emb[0]], [emb[1]])[0,0])
            st.metric("Similarity to ideal answer", f"{sim:.2f}")
            # simple rubric
            if sim > 0.70:
                st.success("Great! You covered most key points.")
            elif sim > 0.45:
                st.info("Decent. Add more detail or examples.")
            else:
                st.warning("Too far off. Revisit core definitions and give 1–2 examples.")
            # optionally reveal ideal
            with st.expander("Show an ideal answer"):
                st.write(cand["answer"])
