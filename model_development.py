#!/usr/bin/env python3
"""
Telecom Churn Prediction - Model Development Template
Implements recommended models with evaluation framework

Usage:
    python model_development.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TELECOM CHURN PREDICTION - MODEL DEVELOPMENT FRAMEWORK")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING & PREPARATION
# ============================================================================

print("\n1. LOADING DATA & PREPARATION")
print("-" * 80)

# Load engineered dataset
df = pd.read_csv('telecom_churn_engineered.csv')
print(f"✓ Dataset loaded: {df.shape[0]:,} records, {df.shape[1]} features")

# Check for target variable
if 'churn' in df.columns:
    X = df.drop('churn', axis=1)
    y = df['churn']
elif 'churned' in df.columns:
    X = df.drop('churned', axis=1)
    y = df['churned']
else:
    print("⚠ Target variable 'churn' not found. Using last column as target.")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

print(f"Features (X): {X.shape[1]} columns")
print(f"Target (y): Churn rate = {y.mean():.1%}")

# Data type summary
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"  - Numeric features: {len(numeric_cols)}")
print(f"  - Categorical features: {len(categorical_cols)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✓ Train-test split: {X_train.shape[0]} train | {X_test.shape[0]} test")
print(f"  - Train churn rate: {y_train.mean():.1%}")
print(f"  - Test churn rate: {y_test.mean():.1%}")

# Feature scaling (for algorithms that need it)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Feature scaling applied")

# ============================================================================
# 2. MODEL EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Comprehensive model evaluation"""
    
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}")
    
    # Training and prediction
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Get probabilities for AUC
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    # Display results
    print(f"\n📊 Performance Metrics:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision:   {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:      {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  AUC-ROC:     {roc_auc:.4f}")
    print(f"  Cross-Val:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print(f"\n🎯 Confusion Matrix:")
    print(f"  True Negative:  {tn:5d} | False Positive: {fp:5d}")
    print(f"  False Negative: {fn:5d} | True Positive:  {tp:5d}")
    
    print(f"\n💡 Key Insights:")
    print(f"  - Correctly identified churners: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)")
    print(f"  - False positive rate: {fp/(fp+tn)*100:.1f}%")
    print(f"  - Business value: ${tp*300 - fn*300 - fp*50:,.0f} (approx)")
    
    return {
        'model': model,
        'name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

# ============================================================================
# 3. BASELINE MODEL - LOGISTIC REGRESSION
# ============================================================================

print("\n" + "="*80)
print("PHASE 1: BASELINE MODEL")
print("="*80)

lr_model = LogisticRegression(
    C=0.1,
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

results = []
results.append(evaluate_model(
    lr_model, X_train_scaled, X_test_scaled, y_train, y_test,
    "Logistic Regression (Baseline)"
))

# ============================================================================
# 4. TREE-BASED MODELS
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: TREE-BASED MODELS")
print("="*80)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

results.append(evaluate_model(
    rf_model, X_train, X_test, y_train, y_test,
    "Random Forest"
))

# ============================================================================
# 5. GRADIENT BOOSTING MODELS
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: GRADIENT BOOSTING MODELS")
print("="*80)

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
    random_state=42,
    verbosity=0
)

results.append(evaluate_model(
    xgb_model, X_train, X_test, y_train, y_test,
    "XGBoost"
))

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
    random_state=42,
    verbose=-1
)

results.append(evaluate_model(
    lgb_model, X_train, X_test, y_train, y_test,
    "LightGBM"
))

# ============================================================================
# 6. ENSEMBLE MODEL - STACKING
# ============================================================================

print("\n" + "="*80)
print("PHASE 4: ENSEMBLE MODEL (STACKING)")
print("="*80)

base_learners = [
    ('xgb', XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, 
                         scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                         random_state=42, verbosity=0)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, 
                                  max_features='sqrt', random_state=42, n_jobs=-1)),
    ('lgb', lgb.LGBMClassifier(n_estimators=150, max_depth=7, learning_rate=0.1,
                              scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                              random_state=42, verbose=-1))
]

meta_learner = LogisticRegression(C=0.1, max_iter=1000, random_state=42)

ensemble_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)

results.append(evaluate_model(
    ensemble_model, X_train, X_test, y_train, y_test,
    "Stacked Ensemble (XGB + RF + LGB)"
))

# ============================================================================
# 7. MODEL COMPARISON & RANKING
# ============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON & RANKING")
print("="*80)

# Create comparison dataframe
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df[['name', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cv_mean']]
comparison_df = comparison_df.sort_values('roc_auc', ascending=False).reset_index(drop=True)
comparison_df.index = comparison_df.index + 1

print("\n📊 Model Performance Ranking:")
print(comparison_df.to_string())

# Best model
best_idx = comparison_df['roc_auc'].idxmax() - 1
best_model_info = results[best_idx]
best_model = best_model_info['model']
best_name = best_model_info['name']

print(f"\n🏆 BEST MODEL: {best_name}")
print(f"   AUC-ROC: {best_model_info['roc_auc']:.4f}")
print(f"   Accuracy: {best_model_info['accuracy']:.4f}")

# ============================================================================
# 8. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n🔝 Top 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    print(f"... and {len(feature_importance)-15} more features")

elif hasattr(best_model, 'coef_'):
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': best_model.coef_[0]
    })
    feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    
    print("\n🔝 Top 15 Most Important Features:")
    print(feature_importance[['Feature', 'Coefficient']].head(15).to_string(index=False))

# ============================================================================
# 9. CHURN RISK SCORING
# ============================================================================

print("\n" + "="*80)
print("CHURN RISK SCORING")
print("="*80)

# Generate scores on full dataset
if hasattr(best_model, 'predict_proba'):
    churn_scores = best_model.predict_proba(X_test)[:, 1]
else:
    churn_scores = best_model.decision_function(X_test)

# Normalize to 0-100 scale
churn_scores_scaled = (churn_scores - churn_scores.min()) / (churn_scores.max() - churn_scores.min()) * 100

# Segmentation
risk_segments = pd.cut(churn_scores_scaled, bins=[0, 25, 50, 75, 100],
                       labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'])

print("\n📊 Customer Risk Segmentation:")
for segment in ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']:
    count = (risk_segments == segment).sum()
    pct = count / len(risk_segments) * 100
    print(f"  {segment:15s}: {count:5d} customers ({pct:5.1f}%)")

print("\n💡 Risk Score Stats:")
print(f"  Min Score:  {churn_scores_scaled.min():.1f}")
print(f"  Max Score:  {churn_scores_scaled.max():.1f}")
print(f"  Mean Score: {churn_scores_scaled.mean():.1f}")
print(f"  Median:     {np.median(churn_scores_scaled):.1f}")

# ============================================================================
# 10. BUSINESS IMPACT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("BUSINESS IMPACT ANALYSIS")
print("="*80)

# Cost-benefit parameters
cost_false_negative = 300    # Cost of not retaining churner
cost_false_positive = 50     # Cost of retention offer
benefit_retention = 300      # Revenue saved

# Calculate for best model
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()

revenue_saved = tp * benefit_retention
cost_of_offers = fp * cost_false_positive
cost_of_lost = fn * cost_false_negative

net_value = revenue_saved - cost_of_offers - cost_of_lost

print(f"\n💰 Financial Impact:")
print(f"  Revenue Saved (Retained Customers):  ${revenue_saved:>12,.0f}")
print(f"  Cost of Offers (False Positives):   ${cost_of_offers:>12,.0f}")
print(f"  Cost of Losses (False Negatives):   ${cost_of_lost:>12,.0f}")
print(f"  {'─'*45}")
print(f"  Net Business Value:                  ${net_value:>12,.0f}")

print(f"\n🎯 Intervention Targeting:")
print(f"  High Risk + Critical: {(risk_segments.isin(['High Risk', 'Critical Risk'])).sum()} customers")
print(f"  Expected Retention:   {(risk_segments.isin(['High Risk', 'Critical Risk'])).sum() * 0.4:.0f} customers")
print(f"  Expected Revenue:     ${(risk_segments.isin(['High Risk', 'Critical Risk'])).sum() * 0.4 * benefit_retention:,.0f}")

# ============================================================================
# 11. SUMMARY & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

print(f"""
✅ MODEL DEVELOPMENT COMPLETE

Selected Model: {best_name}
Performance:
  - Accuracy:  {best_model_info['accuracy']*100:.2f}%
  - Precision: {best_model_info['precision']*100:.2f}%
  - Recall:    {best_model_info['recall']*100:.2f}%
  - AUC-ROC:   {best_model_info['roc_auc']:.4f}

Next Steps:
  1. Deploy model for production churn scoring
  2. Generate churn risk scores for all customers
  3. Segment customers by risk level
  4. Design targeted retention programs for High/Critical risk
  5. Monitor model performance monthly
  6. Retrain quarterly with new data

Expected Impact:
  - Identify ~{(risk_segments.isin(['High Risk', 'Critical Risk'])).sum()} high-risk customers
  - Retain ~{(risk_segments.isin(['High Risk', 'Critical Risk'])).sum() * 0.4:.0f} customers through interventions
  - Revenue impact: ~${net_value:,.0f} per cycle

""")

print("="*80)
print(f"Analysis completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
