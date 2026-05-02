  
**Author:** Goda Mahesh  

This repository contains solutions for three data-driven tasks based on the OULAD dataset. The focus is on building practical systems for student engagement analysis, early risk prediction, and course recommendation.

---
IMPORTANT NOTICE: THE DATA FILE DOESNOT CONTAIN STUDENTVLE FILE AS ITS SIZE IS VERY HIGH THAN REQUIRED I RECOMMEND YOU TO DOWNLOAD THIS REPOSITORY AND ADD STUDENTVLE FILE TO DATA AND RUN THE CODE 
## 📌 Overview

The project is divided into three main tasks:

1. **Behavioral Scoring Framework**
2. **Predictive Disengagement Model**
3. **Course Recommendation Engine**

Each task is designed to solve a real-world university problem using data and simple, scalable methods.

---

## 📊 Task 1: Behavioral Scoring Framework

### 🎯 Goal
Create a dynamic engagement score (0–100) that reflects a student’s weekly behavior instead of final grades.

### ⚙️ Approach
- Extract behavioral features from VLE clickstream data:
  - Recency
  - Frequency
  - Trend slope
  - Submission rate
  - Diversity
  - Timeliness
- Normalize features using **percentile ranking (week-wise)**
- Compute weighted score

### 🧠 Key Idea
Score updates **week-by-week**, allowing early detection of disengagement.

### 📈 Output
- Engagement trajectory per student
- Archetypes:
  - Steady Engager
  - Early Dropout
  - Late Recoverer

---

## ⚠️ Task 2: Predictive Disengagement Model

### 🎯 Goal
Predict whether a student will **withdraw or fail before Week 6**.

### ⚙️ Model
- **XGBoost Classifier**
- Handles:
  - Class imbalance (built-in weighting)
  - Categorical features
- Uses only **Week ≤ 6 data (no leakage)**

### 📊 Features Used
- Click activity (percentile)
- Activity diversity
- Recency ratio
- Weighted assessment score
- Course assignment indicator
- Socio-economic factor (IMD)

### 🔥 Important Design Choices
- **Weighted scoring for assessments**
- Handles courses **without early quizzes**
- Threshold lowered to **0.35** to improve recall

### 📈 Performance
- Precision = 0.60  
- Recall = 0.72  
- F1-score = 0.65  
- ROC-AUC = 0.76  

### 🎯 Why Recall?
Missing an at-risk student is more harmful than a false alert → recall is prioritised.

### 📉 Calibration
Model probabilities are validated using calibration curves and match real outcomes closely.

### 🛎️ Staff Alerts
Each alert includes:
- Student ID
- Course ID
- Risk %
- Risk level
- Key reasons (low clicks, low score, etc.)
- Suggested action

---

## 🎓 Task 3: Course Recommendation Engine

### 🎯 Goal
Recommend **top 3 courses** for the next semester.

### ⚙️ Approaches

#### 1. Content-Based Filtering
- Uses:
  - Course enrolment
  - Completion rate
  - Average score
  - Domain encoding

#### 2. Collaborative Filtering
- Uses student similarity based on:
  - VLE activity patterns
- Finds similar students and recommends their successful courses

### 🔄 Final Recommendation Logic
1. Collaborative filtering  
2. Content-based fallback  
3. Cold-start fallback  

---

## 🧊 Cold Start Handling
For new students:
- Recommend courses using **weighted popularity**
  - Completion rate
  - Average score
  - Enrolment

---

## 📏 Evaluation
- **Strict temporal split (no leakage)**
- Train: past semesters  
- Test: 2014J semester  

### 📊 Metric
- **Precision@3 = 0.729**

This shows the system correctly predicts future course choices in many cases.

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost

---

## 🚀 Key Highlights
- Dynamic engagement scoring system  
- Early warning system for student risk  
- Real-world recommendation engine  
- Leakage-free evaluation  
- Scalable and production-friendly design  

---

## 📂 How to Run
1. Place OULAD dataset files in the project directory  
2. Run Python scripts/notebooks for each task  
3. Outputs:
   - Engagement scores
   - Risk predictions
   - Course recommendations  

---

## 📌 Final Note
This project focuses on **practical implementation** rather than complex theory. The aim is to build systems that can actually be used by universities for decision-making.

---
