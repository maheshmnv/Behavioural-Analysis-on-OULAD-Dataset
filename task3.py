"""
TASK 3 — Course Recommendation Engine (OULAD) - UPGRADED VERSION
================================================================
Fixes applied: Strict Temporal Splitting (No Time-Machine leakage),
Vectorized Matrix Operations (O(1) lookups), Contextual Content Filtering,
and resolution of the Evaluation Paradox.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# --------------------------------------------------------------------------
# 1. DATA LOADING & TEMPORAL SPLIT (The most important fix)
# --------------------------------------------------------------------------
def load_and_split_data(data_dir=".", test_semester="2014J"):
    """
    Loads data and explicitly splits it into Past (Train) and Future (Test).
    This prevents the model from peeking at future outcomes.
    """
    files = {
        "studentInfo": "studentInfo.csv",
        "studentVle": "full_student_vle.csv",
        "vle": "vle.csv",
    }
    tables = {}
    for name, fname in files.items():
        try:
            tables[name] = pd.read_csv(f"{data_dir}/{fname}")
            print(f"Loaded {name}")
        except FileNotFoundError:
            exit(f"FATAL: {fname} is required but not found in {data_dir}.")

    # --- TEMPORAL SPLIT ---
    # We build the recommendation logic ONLY using history.
    history_mask = tables["studentInfo"]["code_presentation"] != test_semester
    
    train_tables = {
        "studentInfo": tables["studentInfo"][history_mask].copy(),
        "studentVle": tables["studentVle"][tables["studentVle"]["code_presentation"] != test_semester].copy(),
        "vle": tables["vle"].copy()
    }
    
    # The test set is ONLY students taking courses in the test_semester
    test_info = tables["studentInfo"][~history_mask].copy()
    
    return train_tables, test_info

# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (Strictly on Training Data)
# --------------------------------------------------------------------------
def build_course_features(train_tables):
    """Build course popularity strictly using historical data."""
    si = train_tables["studentInfo"].copy()
    
    score_map = {"Distinction": 90, "Pass": 70, "Fail": 50, "Withdrawn": 30}
    si["score_num"] = si["final_result"].map(score_map).fillna(0)
    si["passed"] = si["final_result"].isin(["Pass", "Distinction"]).astype(int)

    # Note: We group by code_module only, treating all past presentations as overall historical popularity
    course = si.groupby("code_module").agg(
        enrolment=("id_student", "count"),
        completion_rate=("passed", "mean"),
        avg_score=("score_num", "mean"),
    ).reset_index()

    domain_map = {"AAA":0, "BBB":1, "CCC":2, "DDD":3, "EEE":4, "FFF":5, "GGG":6}
    course["domain_enc"] = course["code_module"].map(domain_map).fillna(-1)
    
    # Normalize features for Content-Based cosine similarity
    for col in ["enrolment", "completion_rate", "avg_score"]:
        max_val = course[col].max()
        course[f"{col}_norm"] = course[col] / (max_val if max_val > 0 else 1)

    return course

def build_vle_matrix(train_tables):
    """
    Vectorized VLE matrix. Rows = Students, Cols = Activity Types.
    Uses historical clicks only.
    """
    svle = train_tables["studentVle"].merge(train_tables["vle"][["id_site","activity_type"]], on="id_site")
    svle["sum_click"] = pd.to_numeric(svle["sum_click"], errors="coerce").fillna(0)
    
    pivot = svle.pivot_table(
        index="id_student", columns="activity_type",
        values="sum_click", aggfunc="sum", fill_value=0
    )

    # Fast L2 normalization using numpy
    matrix = pivot.values
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix_norm = matrix / norms
    
    return pd.DataFrame(matrix_norm, index=pivot.index, columns=pivot.columns)

# --------------------------------------------------------------------------
# 3. RECOMMENDATION ENGINE
# --------------------------------------------------------------------------
class RecommendationEngine:
    def __init__(self, train_tables, course_features, vle_matrix):
        self.course_features = course_features.set_index("code_module")
        self.vle_matrix = vle_matrix
        
        # Pre-compute historical passes for fast O(1) lookups
        si = train_tables["studentInfo"]
        passed = si[si["final_result"].isin(["Pass", "Distinction"])]
        self.student_history = passed.groupby("id_student")["code_module"].apply(set).to_dict()

    def get_cold_start(self, top_k=3, preferred_domain=None):
        """Fallback for new students using a weighted popularity score."""
        df = self.course_features.copy()
        df["popularity"] = (0.5 * df["completion_rate_norm"] + 
                            0.3 * df["avg_score_norm"] + 
                            0.2 * np.log1p(df["enrolment"]))
        
        if preferred_domain:
            mask = df.index.str.upper().str.startswith(preferred_domain.upper())
            df.loc[mask, "popularity"] *= 2.0
            
        ranked = df.sort_values("popularity", ascending=False).head(top_k)
        return [(idx, row["popularity"]) for idx, row in ranked.iterrows()]

    def get_content_based(self, student_id, top_k=3):
        """Recommends courses similar to ones the student has previously PASSED."""
        past_courses = self.student_history.get(student_id, set())
        if not past_courses:
            return []

        # Build student profile vector from passed courses
        feat_cols = ["domain_enc", "enrolment_norm", "completion_rate_norm", "avg_score_norm"]
        past_feats = self.course_features.loc[list(past_courses), feat_cols].mean().values
        
        # Compare to unseen courses
        unseen = self.course_features[~self.course_features.index.isin(past_courses)]
        if unseen.empty:
            return []
            
        unseen_feats = unseen[feat_cols].values
        
        # Vectorized Cosine Similarity
        norms = np.linalg.norm(unseen_feats, axis=1) * np.linalg.norm(past_feats)
        norms[norms == 0] = 1
        sims = np.dot(unseen_feats, past_feats) / norms
        
        # Map back to course IDs
        results = sorted(zip(unseen.index, sims), key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_collaborative(self, student_id, top_k=3, n_peers=20):
        """Finds peers with similar VLE habits using fast matrix multiplication."""
        if student_id not in self.vle_matrix.index:
            return []
            
        target_vec = self.vle_matrix.loc[student_id].values
        all_vecs = self.vle_matrix.values
        student_ids = self.vle_matrix.index
        
        # O(1) Vectorized Cosine Similarity for the entire dataset
        sims = np.dot(all_vecs, target_vec) 
        
        # Get top peers (ignoring self)
        peer_indices = np.argsort(sims)[-(n_peers+1):-1][::-1]
        
        past_courses = self.student_history.get(student_id, set())
        course_scores = defaultdict(float)
        
        for idx in peer_indices:
            peer_id = student_ids[idx]
            peer_sim = sims[idx]
            if peer_sim < 0.1: continue
                
            peer_passes = self.student_history.get(peer_id, set())
            for course in peer_passes:
                if course not in past_courses:
                    course_scores[course] += peer_sim
                    
        return sorted(course_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def recommend(self, student_id, preferred_domain=None, top_k=3):
        """Unified Engine logic with smart padding."""
        final_recs = []
        seen = set()
        past_courses = self.student_history.get(student_id, set())

        # 1. Try Collaborative
        for course, score in self.get_collaborative(student_id, top_k):
            final_recs.append((course, score))
            seen.add(course)

        # 2. Pad with Content-Based if needed
        if len(final_recs) < top_k:
            for course, score in self.get_content_based(student_id, top_k):
                if course not in seen and course not in past_courses:
                    final_recs.append((course, score))
                    seen.add(course)
                    if len(final_recs) == top_k: break

        # 3. Pad with Cold Start if STILL needed
        if len(final_recs) < top_k:
            for course, score in self.get_cold_start(top_k, preferred_domain):
                if course not in seen and course not in past_courses:
                    final_recs.append((course, score))
                    seen.add(course)
                    if len(final_recs) == top_k: break

        return final_recs

# --------------------------------------------------------------------------
# 4. SCIENTIFIC EVALUATION (Strict Hold-Out)
# --------------------------------------------------------------------------
def evaluate_engine(engine, test_info, top_k=3):
    """
    Evaluates precision by trying to predict the courses students registered 
    for in the '2014J' semester, using ONLY their history prior to 2014J.
    """
    print("\n--- Running Scientific Evaluation (Test Set: 2014J) ---")
    
    # We only test on students who have a history in our engine
    test_students = test_info["id_student"].unique()
    eval_pool = [sid for sid in test_students if sid in engine.student_history]
    
    if not eval_pool:
        print("No returning students found in the test semester to evaluate.")
        return

    hits = 0
    total = 0

    for sid in eval_pool:
        # What did they actually take in the future (2014J)?
        actual_future_courses = set(test_info[test_info["id_student"] == sid]["code_module"])
        
        # What does our engine (trained on the past) recommend?
        recs = engine.recommend(sid, top_k=top_k)
        rec_ids = [course for course, score in recs]
        
        # Did we successfully predict at least one of their choices?
        if any(course in rec_ids for course in actual_future_courses):
            hits += 1
        total += 1

    precision = hits / total if total else 0
    print(f"Evaluated {total} returning students.")
    print(f"Precision@{top_k}: {precision:.4f} (Strict, Leakage-Free)")

# --------------------------------------------------------------------------
# 5. EXECUTION BOOTSTRAP
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("Initializing Data Pipeline...")
    train_tables, test_info = load_and_split_data(data_dir=".", test_semester="2014J")
    
    print("Engineering Historical Features...")
    course_features = build_course_features(train_tables)
    vle_matrix = build_vle_matrix(train_tables)
    
    print("Booting Recommendation Engine...")
    engine = RecommendationEngine(train_tables, course_features, vle_matrix)
    
    # Run Evaluation
    evaluate_engine(engine, test_info)
    
    # Interactive Example
    print("\n--- Example: Cold Start Request (Domain: Science/Math - 'FFF') ---")
    for idx, (course, score) in enumerate(engine.recommend(student_id=None, preferred_domain="FFF"), 1):
        print(f" {idx}. Module {course} (Score: {score:.3f})")
    sample_sid = 491606
    if sample_sid in train_tables["studentInfo"]["id_student"].values:
        print(f"Student {sample_sid} (has VLE data and past courses)")
        recs = engine.recommend(sample_sid, top_k=3)
        print("Final top-3 hybrid recommendations:")
        for i, (cid, score) in enumerate(recs, 1):
            print(f"  {i}. {cid} (score: {score:.4f})")
        cf = engine.get_collaborative(sample_sid, 3)
        cb = engine.get_content_based(sample_sid, 3)
        print("  Collaborative suggestions:", [c for c,_ in cf])
        print("  Content-Based suggestions: ", [c for c,_ in cb])
    else:
        print(f"Student {sample_sid} not found in training data – skipping this scenario.")