

import pandas as pd
import numpy as np
from scipy import stats
from typing import Sequence, Optional
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE WEIGHTS (sum to 1.0)
# Derived from OULAD literature + empirical correlation analysis.
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_WEIGHTS = {
       # Most predictive per Hlosta et al. 2017
    'frequency':       0.30,   # Normalised click rate
    'recency':         0.20,   # Recent vs. historical engagement ratio
    'trend_slope':     0.10,   # OLS slope on weekly click series
    'submission_rate': 0.20,   # Assessment submission (not score)
    'diversity':       0.10,   # Shannon entropy across VLE activity types
    'timeliness':      0.10,   # On-time submission fraction
}

assert abs(sum(FEATURE_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1."

FEATURE_NAMES = list(FEATURE_WEIGHTS.keys())


class EngagementScorer:
    """
    Computes weekly engagement scores for all students.

    Parameters
    ----------
    vle : pd.DataFrame
        Must contain: [id_student, week, activity_type, sum_click]
    assessments : pd.DataFrame
        Must contain: [id_student, due_week, submission_week, days_late]
    weights : dict, optional
        Override default feature weights. Must sum to 1.
    """

    def __init__(
        self,
        vle: pd.DataFrame,
        assessments: pd.DataFrame,
        weights: Optional[dict] = None,
    ):
        self.vle         = vle.copy()
        self.assessments = assessments.copy()
        self.weights     = weights or FEATURE_WEIGHTS

        # Pre-index for speed
        self._vle_idx   = {sid: grp for sid, grp in self.vle.groupby('id_student')}
        self._asmnt_idx = {sid: grp for sid, grp in self.assessments.groupby('id_student')}

    # ── INDIVIDUAL FEATURE FUNCTIONS ─────────────────────────────────────────

    def _recency(self, sv: pd.DataFrame, up_to_week: int) -> float:
        """
        Ratio of last-2-week clicks to cumulative weekly average.
        Detects 'silent disengagement' — active students going quiet.

        Range: [0, 3] (clipped to prevent outlier dominance).
        """
        recent   = sv[sv['week'] >= up_to_week - 1]['sum_click'].sum()
        total    = sv['sum_click'].sum()
        elapsed  = max(up_to_week, 1)
        cum_avg  = total / elapsed
        return min(3.0, recent / max(cum_avg, 1e-9))

    def _frequency(self, sv: pd.DataFrame, up_to_week: int) -> float:
        """
        Total clicks divided by weeks elapsed.
        Normalises for course stage (week 5 vs week 20 context differs).

        Unit: clicks / week.
        """
        return sv['sum_click'].sum() / max(up_to_week, 1)

    def _consistency(self, sv: pd.DataFrame, up_to_week: int) -> float:
        """
        Fraction of elapsed weeks with ≥1 click.
        Most predictive single feature in OULAD (Hlosta et al. 2017, AUC 0.73).

        Range: [0, 1].
        """
        active_weeks = sv[sv['sum_click'] > 0]['week'].nunique()
        return active_weeks / max(up_to_week, 1)

    def _diversity(self, sv: pd.DataFrame) -> float:
        """
        Shannon entropy (nats) over VLE activity types.
        Multi-modal engagement → deeper course integration.

        Range: [0, ln(n_activity_types)].
        """
        if sv.empty:
            return 0.0
        counts = sv.groupby('activity_type')['sum_click'].sum()
        probs  = counts / counts.sum()
        return float(stats.entropy(probs))

    def _trend_slope(self, sv: pd.DataFrame, up_to_week: int) -> float:
        """
        OLS slope of weekly click series.
        Positive → rising engagement; negative → declining.
        """
        if up_to_week < 3:
            return 0.0
            
        series = (sv.groupby('week')['sum_click']
                    .sum()
                    .reindex(range(0, up_to_week + 1), fill_value=0))
        
        # SAFETY FIX: Force the pandas series into a raw float array
        # This prevents SciPy from crashing on 'object' data types
        y_values = np.array(series.values, dtype=float)
        x_values = np.arange(len(y_values), dtype=float)
        
        # SAFETY FIX 2: If the student has exactly the same clicks every week 
        # (like 0 clicks every week), the line is flat. The slope is 0.
        if np.std(y_values) == 0:
            return 0.0
            
        slope, *_ = stats.linregress(x_values, y_values)
        return float(slope)

    def _submission_rate(self, sa: pd.DataFrame, sa_due: pd.DataFrame) -> float:
        """
        Fraction of due assessments that were submitted.
        Leading indicator of intent-to-complete (Kuzilek et al. 2017).

        Range: [0, 1].
        """
        return len(sa) / max(len(sa_due), 1)

    def _timeliness(self, sa: pd.DataFrame) -> float:
        """
        Fraction of submitted assessments delivered on time (days_late == 0).
        Neutral prior of 0.5 when no submissions yet.

        Range: [0, 1].
        """
        if sa.empty:
            return 0.5
        return float((sa['days_late'] == 0).mean())

    # ── SINGLE (student, week) FEATURE VECTOR ────────────────────────────────

    def compute_features(self, student_id: int, up_to_week: int) -> dict:
        """
        Return the 7-feature vector for a student, using only data ≤ up_to_week.
        No future data is ever accessed (strict causal constraint).
        """
        sv = self._vle_idx.get(student_id, pd.DataFrame(columns=self.vle.columns))
        sa_all = self._asmnt_idx.get(student_id, pd.DataFrame(columns=self.assessments.columns))

        sv     = sv[sv['week'] <= up_to_week]
        sa_due = sa_all[sa_all['due_week'] <= up_to_week]
        sa_sub = sa_all[sa_all['due_week'] <= up_to_week]

        return {
            'recency':         self._recency(sv, up_to_week),
            'frequency':       self._frequency(sv, up_to_week),
            'consistency':     self._consistency(sv, up_to_week),
            'diversity':       self._diversity(sv),
            'trend_slope':     self._trend_slope(sv, up_to_week),
            'submission_rate': self._submission_rate(sa_sub, sa_due),
            'timeliness':      self._timeliness(sa_sub),
        }

    # ── FULL SCORING RUN ─────────────────────────────────────────────────────

    def run_all_weeks(
        self,
        student_ids: Sequence[int],
        weeks: Sequence[int],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Compute engagement scores for all (student, week) pairs.

        Normalisation strategy: within-week population percentile rank.
        This ensures the score is always relative to peers at the same point in
        the semester — removing course-difficulty and seasonal confounds.

        Returns
        -------
        pd.DataFrame with columns:
            id_student, week, <7 raw features>, engagement_score
        """
        records = []
        for w in weeks:
            if verbose and w % 5 == 0:
                print(f"  Scoring week {w} / {max(weeks)} ...")
            for sid in student_ids:
                feat = self.compute_features(sid, w)
                feat['id_student'] = sid
                feat['week'] = w
                records.append(feat)

        df = pd.DataFrame(records)

        # ── Percentile normalisation per week ────────────────────────────────
        for w in weeks:
            mask = df['week'] == w
            week_slice = df.loc[mask, FEATURE_NAMES].copy()

            # Shift negatives to ≥0 before ranking
            for col in FEATURE_NAMES:
                col_min = week_slice[col].min()
                if col_min < 0:
                    week_slice[col] -= col_min

            df.loc[mask, FEATURE_NAMES] = week_slice.rank(pct=True)

        # ── Weighted sum → [0, 100] ───────────────────────────────────────────
        df['engagement_score'] = (
            sum(df[f] * w for f, w in self.weights.items()) * 100
        ).round(1)

        return df[['id_student', 'week'] + FEATURE_NAMES + ['engagement_score']]


# ──────────────────────────────────────────────────────────────────────────────
# ARCHETYPE CLASSIFIER  (post-hoc labels for reporting, not used in scoring)
# ──────────────────────────────────────────────────────────────────────────────

def classify_archetype(trajectory: Sequence[float]) -> str:
    """
    Heuristically label a student's archetype based on their score trajectory.
    Used for staff-facing dashboards, not for score computation.

    Parameters
    ----------
    trajectory : list of floats
        Weekly engagement scores in chronological order.

    Returns
    -------
    str: 'steady_engager' | 'early_dropout' | 'late_recoverer' | 'at_risk'
    """
    scores = np.array(trajectory)
    if len(scores) < 4:
        return 'at_risk'

    early  = scores[:len(scores)//3].mean()
    mid    = scores[len(scores)//3 : 2*len(scores)//3].mean()
    late   = scores[2*len(scores)//3:].mean()
    slope  = np.polyfit(np.arange(len(scores)), scores, 1)[0]
    spread = scores.std()

    if late < 35 and early > 40:
        return 'early_dropout'
    if late > 55 and early < 45 and slope > 0:
        return 'late_recoverer'
    if spread < 8 and late > 50:
        return 'steady_engager'
    return 'at_risk'


# ──────────────────────────────────────────────────────────────────────────────
# CLI DEMO
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import numpy as np
    print("Loading data...")
    vle    = pd.read_csv('C:\\Users\\mahes\\Downloads\\studor assignment\\archive (2)\\full_student_vle.csv')
    assess = pd.read_csv('C:\\Users\\mahes\\Downloads\\studor assignment\\archive (2)\\studentAssessment.csv')
    info   = pd.read_csv('C:\\Users\\mahes\\Downloads\\studor assignment\\archive (2)\\studentInfo.csv')

    # --- BARE MINIMUM REQUIRED FOR VLE ---
    # The code needs 'activity_type', which lives in vle.csv
    vle_dict = pd.read_csv('C:\\Users\\mahes\\Downloads\\studor assignment\\archive (2)\\vle.csv')
    vle = pd.merge(vle, vle_dict[['id_site', 'activity_type']], on='id_site', how='left')
    vle['week'] = np.where(vle['date'] < 0, 0, (vle['date'] // 7) + 1)

    # --- BARE MINIMUM REQUIRED FOR ASSESSMENTS ---
    # The code needs 'due_week' and 'days_late'. We have to get the deadline 'date' from assessments.csv
    assess_dict = pd.read_csv('C:\\Users\\mahes\\Downloads\\studor assignment\\archive (2)\\assessments.csv')
    assess = pd.merge(assess, assess_dict[['id_assessment', 'date']], on='id_assessment', how='left')
    
    # Some exams have no deadline in the dataset, fill them with submission date so math doesn't break
    assess['date'] = assess['date'].fillna(assess['date_submitted'])

    # Create the exact columns the class expects
    assess['due_week'] = np.where(assess['date'] < 0, 0, (assess['date'] // 7) + 1)
    assess['submission_week'] = np.where(assess['date_submitted'] < 0, 0, (assess['date_submitted'] // 7) + 1)
    assess['days_late'] = assess['date_submitted'] - assess['date']
    # Filter to just one specific class and semester
    info = info[(info['code_module'] == 'BBB') & (info['code_presentation'] == '2013J')]
    valid_students = info['id_student'].unique()

    # Then filter your VLE and Assess data using valid_students
    vle = vle[vle['id_student'].isin(valid_students)]
    assess = assess[assess['id_student'].isin(valid_students)]
    # Drop the noise we identified!
    noise_activities = ['homepage', 'subpage', 'sharedsubpage', 'glossary']
    vle = vle[~vle['activity_type'].isin(noise_activities)]
    # --- RUN ENGINE ---
    scorer = EngagementScorer(vle, assess)

    sample_ids = info['id_student'].tolist()
    weeks      = list(range(1, 26))

    print(f"Scoring {len(sample_ids)} students across {len(weeks)} weeks...")
    results = scorer.run_all_weeks(sample_ids, weeks, verbose=True)

    print("\nSample output:")
    print(results.head(10).to_string(index=False))

    print("\nWeek-20 summary:")
    w20 = results[results['week'] == 20].merge(
        info[['id_student','final_result']], on='id_student')
    print(w20.groupby('final_result')['engagement_score'].agg(['mean','std','count']).round(2))

    results.to_csv('engagement_scores_output.csv', index=False)
    print("\nSaved to engagement_scores_output.csv")
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # EXTRACTING & DRAWING TRAJECTORIES DIRECTLY IN PYTHON
    # ---------------------------------------------------------
    print("\n--- Extracting and Drawing Trajectories ---")
    import matplotlib.pyplot as plt

    # 1. CREATE THE 'TRAJECTORIES' DATA: Sort, group, and classify
    results_sorted = results.sort_values(by=['id_student', 'week'])
    trajectories = results_sorted.groupby('id_student')['engagement_score'].apply(list).reset_index(name='score_array')
    trajectories['archetype_label'] = trajectories['score_array'].apply(classify_archetype)

    # 2. SET UP THE PLOT
    plt.figure(figsize=(10, 6))
    target_archetypes = ['steady_engager', 'late_recoverer', 'early_dropout']
    
    # Assign colors to make it look professional
    colors = {
        'steady_engager': 'green', 
        'late_recoverer': 'orange', 
        'early_dropout': 'red'
    }

    # 3. LOOP THROUGH AND DRAW THE LINES
    for arch in target_archetypes:
        # Now it knows what 'trajectories' is!
        matching_students = trajectories[trajectories['archetype_label'] == arch]
        
        if not matching_students.empty:
            # Grab our representative student
            sample = matching_students.iloc[0]
            
            # Plot their line! (X-axis = Weeks 1-25, Y-axis = Their scores)
            plt.plot(
                range(1, 26), 
                sample['score_array'], 
                label=arch.replace('_', ' ').title(), 
                color=colors[arch], 
                linewidth=2.5,
                marker='o'
            )

    # 4. FORMAT AND SAVE THE CHART
    plt.title('PathAI Engagement Trajectory by Archetype', fontsize=14, fontweight='bold')
    plt.xlabel('Week of Semester', fontsize=12)
    plt.ylabel('Engagement Score (0-100)', fontsize=12)
    plt.ylim(0, 100)
    plt.xlim(1, 25)
    plt.xticks(range(1, 26)) 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Save it directly to your folder!
    plt.savefig('archetype_trajectories.png')
    print("Saved plot directly to your folder as 'archetype_trajectories.png'!")