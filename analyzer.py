from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skills import SKILLS

def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill in SKILLS:
        if skill in text:
            found_skills.append(skill)

    return found_skills


def calculate_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(vectors)[0][1]

    return round(similarity * 100, 2)


def get_missing_skills(resume_skills, job_desc):
    job_skills = extract_skills(job_desc)
    missing = list(set(job_skills) - set(resume_skills))
    return missing
