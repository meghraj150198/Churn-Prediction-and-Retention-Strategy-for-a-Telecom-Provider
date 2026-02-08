# 🤖 Machine Learning Model Recommendations
## Telecom Churn Prediction Dataset Analysis & Model Selection Guide

**Date:** February 8, 2026  
**Dataset:** 25,000 records | 84 features | Binary Classification  
**Objective:** Predict churn and enable retention interventions

---

## 📊 Dataset Characteristics Summary

### Favorable Characteristics for Modeling
✅ **Large Dataset** - 25,000 records (sufficient for complex models)  
✅ **High Quality** - 100% data completeness, validated & corrected  
✅ **Rich Features** - 84 features across 10 business-aligned categories  
✅ **Feature Engineering** - 48 derived features with business context  
✅ **No Missing Values** - Ready for immediate modeling  
✅ **Imbalanced Classification** - Real-world churn typically ~25-30% (manageable)  

### Dataset Structure for Modeling
- **Target Variable:** `churn` (Binary: 0=Retained, 1=Churned)
- **Feature Types:** 
  - Numeric (20): Age, Tenure, Charges, Usage scores, etc.
  - Categorical (7): Gender, Region, Plan Type, Connection Type, etc.
  - Engineered (48): Risk scores, composite indices, flags, trends
- **Sample Size:** 25,000 ≈ 20,000 training + 5,000 validation/test

---

## 🏆 Recommended Models (Ranked by Use Case)

### ⭐ PRIMARY RECOMMENDATION: XGBoost (Gradient Boosting)

**Why XGBoost for Your Use Case:**

✅ **Performance Leaders:** Best predictive accuracy on telecom churn  
✅ **Feature Importance:** Excellent for understanding which factors drive churn  
✅ **Handles Mixed Data:** Works with both numeric and categorical features  
✅ **Business Insight:** Easy to extract rules for retention targeting  
✅ **Robustness:** Less prone to overfitting than basic models  
✅ **Real-World Proven:** Standard in telecom churn prediction competitions  

**Model Advantages:**
- Typically achieves **82-88% accuracy** on churn datasets
- **AUC-ROC: 0.85-0.92** depending on feature quality
- Fast prediction for real-time scoring
- Handles imbalanced classes well with proper parameters
- Feature interactions automatically captured

**Implementation:**
```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y[y==0])/len(y[y==1]),  # Handle imbalance
    random_state=42
)

model.fit(X_train, y_train, 
          eval_set=[(X_test, y_test)],
          early_stopping_rounds=20,
          verbose=True)
```

**Business Application:**
- Generate churn propensity scores (0-100) for each customer
- Identify top features driving churn for intervention design
- Segment customers by risk level (Low/Medium/High/Critical)
- Extract decision rules for retention targeting

---

### 🥈 SECONDARY RECOMMENDATION: Random Forest

**Why Random Forest for Your Use Case:**

✅ **Interpretability:** Feature importance rankings for business stakeholders  
✅ **Robustness:** Multiple trees reduce overfitting risk  
✅ **No Scaling Required:** Works directly with raw features  
✅ **Parallelizable:** Fast training on multi-core systems  
✅ **Stability:** Consistent results across different data samples  

**Model Advantages:**
- Achieves **80-86% accuracy** on churn datasets
- **AUC-ROC: 0.82-0.89**
- Excellent feature importance visualization
- Easy to explain to non-technical stakeholders
- Handles complex feature interactions

**Implementation:**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    class_weight='balanced',  # Handle imbalance
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Feature importance
importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
```

**When to Use Random Forest:**
- Need high interpretability for business teams
- Stakeholders prefer simpler explanations over marginal accuracy gains
- Want baseline model to compare against complex models
- Computational resources are limited

---

### 🥉 THIRD OPTION: LightGBM (Microsoft's Gradient Boosting)

**Why LightGBM for Your Use Case:**

✅ **Speed:** 10-20x faster training than traditional gbm  
✅ **Memory Efficient:** Handles large datasets efficiently  
✅ **Comparable Performance:** Similar accuracy to XGBoost  
✅ **Scalability:** Better for growing datasets  
✅ **GPU Support:** Can accelerate training significantly  

**Model Advantages:**
- Achieves **82-88% accuracy** (similar to XGBoost)
- **AUC-ROC: 0.84-0.91**
- Faster experimentation and iteration
- Better for real-time predictions at scale

**Implementation:**
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y[y==0])/len(y[y==1]),
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          eval_metric='auc',
          callbacks=[
              lgb.early_stopping(20),
              lgb.log_evaluation(100)
          ])
```

---

### 4️⃣ LOGISTIC REGRESSION (Baseline & Interpretable)

**Why Logistic Regression for Your Use Case:**

✅ **Interpretability:** Coefficients directly show feature impact  
✅ **Explainability:** Easy to explain to business stakeholders  
✅ **Probability Output:** Natural probability estimates  
✅ **Baseline Model:** Essential for model comparison  
✅ **Production Ready:** Simple, fast, reliable  

**Model Advantages:**
- Achieves **75-82% accuracy** on churn datasets
- **AUC-ROC: 0.78-0.85**
- Extremely fast training and prediction
- Built-in regularization (L1/L2) prevents overfitting
- Excellent for understanding linear relationships

**Implementation:**
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    C=0.1,  # Regularization strength
    penalty='l2',
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train_scaled, y_train)

# Coefficients show feature importance
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
```

**Use Case:**
- Production baseline model
- For stakeholder presentations (highly interpretable)
- A/B testing against complex models
- Regulatory compliance (explainability requirement)

---

### 5️⃣ NEURAL NETWORKS (Advanced Approach)

**Why Neural Networks:**

⚠️ **Use Only If:** You need maximum accuracy AND have model explainability tools  
✅ Can capture complex non-linear patterns  
✅ Potential for 85-90% accuracy with proper tuning  
❌ Less interpretable (black box)  
❌ Requires more data (you're at minimum for deep learning)  

**Implementation Note:**
```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

model.fit(X_train, y_train, epochs=30, batch_size=32,
          validation_split=0.2, callbacks=[
              keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
          ])
```

**Recommendation:** Use only if you pair with SHAP or LIME for explainability

---

## 🎯 RECOMMENDED SOLUTION: Ensemble Approach

### Best Practice: Stacked Ensemble

**Combine multiple models for optimal performance:**

```
┌─────────────────────────────────────────┐
│        META LEARNER (Logistic Reg)      │
│    (Learns optimal combination)          │
└──────────┬──────────┬──────────┬────────┘
           │          │          │
      ┌────▼──┐   ┌───▼────┐   ┌▼────────┐
      │ XGB   │   │ Random │   │LightGBM │
      │Model  │   │ Forest │   │ Model   │
      └────┬──┘   └───┬────┘   └┬────────┘
           │          │          │
           └──────────┼──────────┘
                      │
                ┌─────▼──────┐
                │  Train Data│
                └────────────┘
```

**Implementation:**
```python
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# Base learners
base_learners = [
    ('xgb', XGBClassifier(n_estimators=200, max_depth=6, subsample=0.8)),
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=15)),
    ('lgb', lgb.LGBMClassifier(n_estimators=200, max_depth=7))
]

# Meta learner
meta_learner = LogisticRegression(C=0.1)

# Stack
model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)

model.fit(X_train, y_train)
```

**Expected Performance:**
- **Accuracy:** 84-89%
- **AUC-ROC:** 0.87-0.93
- **Precision:** 80-85% (fewer false positives = better targeting)
- **Recall:** 75-82% (catches most churners)

---

## 📋 Model Comparison Table

| Model | Accuracy | AUC-ROC | Speed | Interpretability | Complexity | Recommendation |
|-------|----------|---------|-------|------------------|-----------|-----------------|
| **XGBoost** | 82-88% | 0.85-0.92 | ⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | **PRIMARY** |
| **Random Forest** | 80-86% | 0.82-0.89 | ⭐⭐ | ⭐⭐⭐⭐⭐ | Low | SECONDARY |
| **LightGBM** | 82-88% | 0.84-0.91 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | ALTERNATIVE |
| **Logistic Reg** | 75-82% | 0.78-0.85 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Low | BASELINE |
| **Neural Net** | 83-90% | 0.86-0.93 | ⭐⭐ | ⭐⭐ | High | ADVANCED |
| **Ensemble Stack** | 84-89% | 0.87-0.93 | ⭐ | ⭐⭐⭐ | High | **OPTIMAL** |

---

## ⚙️ Implementation Recommendations

### Step 1: Data Preparation
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data with your 48 engineered features
df = pd.read_csv('telecom_churn_engineered.csv')

# Separate features and target
X = df.drop('churn', axis=1)
y = df['churn']

# 80-20 split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} records")
print(f"Test set: {X_test.shape[0]} records")
print(f"Churn rate: {y.mean():.1%}")
```

### Step 2: Feature Preprocessing
```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Identify numeric and categorical columns
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), 
         categorical_cols)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

### Step 3: Handle Class Imbalance
```python
from imblearn.over_sampling import SMOTE

# If churn rate < 20%, use SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_processed, y_train
)
```

### Step 4: Model Training & Evaluation
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
    random_state=42
)

model.fit(X_train_processed, y_train)

# Predictions
y_pred = model.predict(X_test_processed)
y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X_train_processed, y_train, cv=5)
print(f"Cross-Val Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

---

## 🎯 Business Metric Optimization

### For Retention ROI Maximization

Instead of maximizing accuracy, optimize for **business value**:

```python
# Cost-benefit analysis
cost_false_negative = 300  # Cost of not retaining churner (revenue loss)
cost_false_positive = 50   # Cost of retention offer to non-churner
benefit_true_positive = 300 # Revenue saved from retention
benefit_true_negative = 0   # No cost for correct non-churn prediction

# Find optimal threshold
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Calculate business value at each threshold
business_values = []
for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred_threshold)
    tn, fp, fn, tp = cm.ravel()
    
    value = (tp * benefit_true_positive - fn * cost_false_negative 
             - fp * cost_false_positive)
    business_values.append(value)

# Find optimal threshold
optimal_idx = np.argmax(business_values)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Maximum business value: ${max(business_values)/1000:.1f}K")
```

---

## 🚀 Implementation Roadmap

### Week 1: Baseline & Preparation
- [ ] Data preprocessing and feature engineering validation
- [ ] Train Logistic Regression baseline
- [ ] Establish baseline metrics (75-82% accuracy)
- [ ] Prepare evaluation framework

### Week 2: Model Development
- [ ] Train Random Forest model
- [ ] Train XGBoost model
- [ ] Train LightGBM model
- [ ] Compare performance on test set

### Week 3: Advanced Models
- [ ] Implement neural network
- [ ] Develop stacking ensemble
- [ ] Hyperparameter tuning via GridSearch/RandomSearch
- [ ] Cross-validation on best models

### Week 4: Business Integration
- [ ] Feature importance analysis
- [ ] Business rule extraction
- [ ] ROI calculation and optimization
- [ ] Create churn propensity scoring system
- [ ] Build customer segmentation
- [ ] Develop retention recommendations

### Week 5: Deployment Preparation
- [ ] Model serialization (pickle/joblib)
- [ ] API development for predictions
- [ ] Real-time scoring implementation
- [ ] Model monitoring dashboard
- [ ] Documentation and handoff

---

## 💡 Key Considerations for Your Dataset

### Feature Engineering Quality
✅ Your 48 engineered features are excellent quality  
✅ Business-aligned categories (Usage, Payment, Satisfaction, etc.)  
✅ Risk scores and composite indices ready  
**Action:** Use these features as-is; they'll significantly boost model performance

### Class Imbalance Handling
⚠️ Typical telecom churn: 25-30% churn rate  
**Solutions:**
- Use `scale_pos_weight` in XGBoost/LightGBM
- Use `class_weight='balanced'` in sklearn models
- Implement SMOTE if churn rate < 15%
- Adjust classification threshold for business metrics

### Model Validation Strategy
✅ Use **Stratified K-Fold Cross-Validation** (maintain churn ratio in each fold)  
✅ Time-based validation (older data for training, recent for testing)  
✅ Business metric evaluation (not just accuracy)  

---

## 🏁 Final Recommendation

### For Your Telecom Churn Problem:

**PRIMARY CHOICE: XGBoost**
- Best accuracy (82-88%) with strong interpretability
- Company standard for churn prediction
- Natural probabilistic output for scoring
- Feature importance for business insights
- Fast prediction for real-time scoring

**SECONDARY: Random Forest**
- If stakeholders need maximum interpretability
- For comparison/validation against XGBoost
- Excellent feature importance visualization

**OPTIMAL: Ensemble (Stacked Random Forest + XGBoost + LightGBM)**
- Best performance (84-89% accuracy)
- Combines strengths of multiple models
- More robust to data variations
- Worth the complexity investment

### Next Steps:
1. Start with **XGBoost baseline**
2. Compare against **Random Forest**
3. If time permits, build **Stacked Ensemble**
4. Optimize threshold for **business ROI** (not just accuracy)
5. Extract **feature importance** for retention strategy
6. Create **risk segments** for targeting

---

## 📚 References & Resources

### XGBoost Parameter Tuning
- **n_estimators:** Start 200-500, monitor for overfitting
- **max_depth:** 5-10 for balanced bias-variance
- **learning_rate:** 0.05-0.2, lower = more iterations needed
- **subsample:** 0.7-0.9, prevents overfitting
- **colsample_bytree:** 0.7-0.9, feature sampling

### Performance Targets
- **Accuracy:** 82-88% (better than 75% baseline)
- **AUC-ROC:** 0.85-0.90 (separating churners well)
- **Precision:** 80%+ (accuracy of churn predictions)
- **Recall:** 75%+ (identify most potential churners)
- **F1-Score:** 77-83% (balance precision-recall)

---

**Date Updated:** February 8, 2026  
**Dataset Status:** ✓ Ready for modeling  
**Recommended Timeline:** 3-4 weeks for full implementation