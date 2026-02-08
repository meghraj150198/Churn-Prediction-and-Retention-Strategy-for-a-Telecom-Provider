# 📋 OPERATIONAL INTEGRATION GUIDE

**Phase 6: Production Integration & Real-Time Monitoring**  
**Date:** February 8, 2026  
**Status:** Ready for Deployment

---

## Overview

This guide provides step-by-step instructions to integrate the churn prediction model into operational systems and set up real-time monitoring of predictions vs actual outcomes.

## Components

### 1. **integration_service.py** - Operational Integration Layer
Connects the model to business systems for real-time scoring.

**Key Classes:**
- `OperationalDatabase` - SQLite database for predictions, outcomes, and alerts
- `CRMIntegrator` - Connects to CRM systems
- `AlertSystem` - Generates alerts for high-risk customers
- `OperationalIntegration` - Main orchestrator

### 2. **monitoring_system.py** - Performance Monitoring
Tracks model predictions vs actual outcomes and calculates ROI.

**Key Features:**
- Prediction metrics calculation
- Model drift detection
- A/B testing framework
- Cumulative ROI tracking
- Performance dashboarding

### 3. **monitoring_dashboard.py** - Web Dashboard
Real-time visualization of model performance and business metrics.

**Endpoints:**
- `http://localhost:5000/` - Main dashboard
- `http://localhost:5000/api/dashboard` - Metrics API
- `http://localhost:5000/api/health` - Health check

---

## Installation & Setup

### Prerequisites
```bash
pip install flask flask-cors pandas numpy scipy sqlite3
```

### Step 1: Initialize Integration Service

```python
from integration_service import OperationalIntegration

# Initialize with model artifacts
integration = OperationalIntegration(
    model_path='ensemble_final_model_20260208_093741.pkl',
    scaler_path='feature_scaler_20260208_093741.pkl',
    features_path='feature_names_20260208_093741.pkl',
    db_path='operational_db.sqlite'
)
```

This creates:
- SQLite database for operational data
- CRM connectors
- Alert system
- Performance tracking

---

## Usage Examples

### Real-Time Customer Scoring

**Score a single customer:**
```python
customer_data = {
    'customer_id': 'CUST_12345',
    'monthly_charges': 65.50,
    'tenure_months': 24,
    'plan_type': 'Premium',
    # ... all other features
}

prediction = integration.score_customer(customer_data)

if prediction:
    print(f"Risk Score: {prediction.risk_score}")
    print(f"Risk Level: {prediction.risk_level.value}")
    print(f"Action: {prediction.intervention_strategy.value}")
```

**Returns:**
```python
ChurnPrediction(
    customer_id='CUST_12345',
    risk_score=72,
    risk_probability=0.72,
    risk_level=RiskLevel.HIGH,
    intervention_strategy=InterventionStrategy.HIGH,
    revenue_at_risk=469.6,
    intervention_cost=75.0,
    timestamp=datetime.now(),
    model_version='1.0-production'
)
```

### Batch Scoring

**Score multiple customers from CSV:**
```python
success_count, error_count = integration.batch_score_customers(
    'new_customers.csv'
)
print(f"Success: {success_count}, Errors: {error_count}")
```

The system automatically:
1. Loads customer data from CSV
2. Scores each customer
3. Saves to database
4. Pushes to CRM
5. Generates alerts for high-risk

### Record Actual Outcomes

**After retention intervention, track results:**
```python
# Customer churned despite intervention
integration.db.record_outcome(
    customer_id='CUST_12345',
    actual_churn=True,
    intervention_applied=True,
    intervention_cost_actual=75.0,
    revenue_impact=0  # Lost revenue
)

# Customer retained after intervention
integration.db.record_outcome(
    customer_id='CUST_12346',
    actual_churn=False,
    intervention_applied=True,
    intervention_cost_actual=50.0,
    revenue_impact=600  # Saved annual revenue
)
```

---

## Monitoring & Performance Tracking

### 1. Real-Time Dashboard

**Start the web dashboard:**
```bash
python monitoring_dashboard.py
```

**Access at:** `http://localhost:5000`

Dashboard shows:
- ✓ Total predictions (30-day period)
- ✓ Risk distribution (Critical/High/Medium/Low)
- ✓ Pending critical alerts
- ✓ Average risk score
- ✓ ROI metrics
- ✓ Revenue at risk vs saved
- ✓ Program costs and net benefits

### 2. Calculate Cohort Performance

```python
from monitoring_system import MonitoringSystem

monitor = MonitoringSystem('operational_db.sqlite')

# Analyze a cohort
metrics = monitor.calculate_prediction_metrics(
    cohort_name='Critical Risk',
    cohort_date=datetime(2026, 2, 8)
)

print(f"Accuracy: {metrics['accuracy']:.1%}")
print(f"Precision: {metrics['precision']:.1%}")
print(f"ROI: {metrics['roi']:.2f}x")
```

### 3. Detect Model Drift

```python
# Compare baseline vs current performance
baseline = {
    'accuracy': 0.852,
    'precision': 0.823,
    'recall': 0.789,
    'auc_roc': 0.8756
}

current = {
    'accuracy': 0.801,
    'precision': 0.795,
    'recall': 0.775,
    'auc_roc': 0.8401
}

drift_detections = monitor.detect_model_drift(
    baseline, current, threshold_pct=5.0
)

for drift in drift_detections:
    print(f"DRIFT: {drift['metric_name']} = {drift['deviation_pct']:.1f}% degradation")
```

### 4. A/B Test Results

```python
# Compare two retention strategies
ab_results = monitor.get_ab_test_results(
    control_cohort='Standard Retention',
    treatment_cohort='VIP Retention Program'
)

print(f"Control Churn Rate: {ab_results['control']['churn_rate']:.1%}")
print(f"Treatment Churn Rate: {ab_results['treatment']['churn_rate']:.1%}")
print(f"p-value: {ab_results['statistical_test']['p_value']:.4f}")
print(f"Significant: {ab_results['statistical_test']['is_significant']}")
```

### 5. Calculate Cumulative ROI

```python
from datetime import datetime, timedelta

# 30-day ROI
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

roi = monitor.calculate_cumulative_roi(start_date, end_date)

print("=" * 60)
print("CUMULATIVE ROI (Last 30 Days)")
print("=" * 60)
print(f"Revenue Saved: ${roi['financials']['revenue_saved']:,.0f}")
print(f"Program Cost: ${roi['financials']['program_cost']:,.0f}")
print(f"Net Benefit: ${roi['financials']['net_benefit']:,.0f}")
print(f"ROI: {roi['financials']['roi']:.2f}x")
print(f"Customers Retained: {roi['customers']['retained']:,}")
```

---

## CRM Integration

### Salesforce Integration

```python
from integration_service import OperationalIntegration, CRMIntegrator

# Initialize with Salesforce endpoint
crm = CRMIntegrator(
    api_endpoint='https://your-instance.salesforce.com/api/v57.0',
    api_key='your_api_key',
    crm_type='salesforce'
)

# Push churn score to CRM
crm.push_churn_score(prediction)

# Create retention task
crm.create_task(
    customer_id='CUST_12345',
    task_title='High-priority churn retention',
    task_description=f'Risk score: {prediction.risk_score}. Recommend VIP retention.',
    due_date=datetime.now() + timedelta(days=1)
)

# Bulk update multiple scores
crm.bulk_update_scores(predictions_list)
```

### HubSpot Integration

```python
# Similar pattern for HubSpot
crm = CRMIntegrator(
    api_endpoint='https://api.hubapi.com',
    api_key='your_hubspot_key',
    crm_type='hubspot'
)
```

---

## Database Schema

### Predictions Table
Stores all model predictions:
```sql
CREATE TABLE predictions (
    prediction_id TEXT,
    customer_id TEXT,
    risk_score INTEGER,
    risk_probability REAL,
    risk_level TEXT,
    intervention_strategy TEXT,
    revenue_at_risk REAL,
    intervention_cost REAL,
    model_version TEXT,
    timestamp DATETIME,
    created_at DATETIME
)
```

### Outcomes Table
Records actual churn outcomes:
```sql
CREATE TABLE outcomes (
    outcome_id TEXT,
    customer_id TEXT,
    actual_churn BOOLEAN,
    actual_churned_date DATETIME,
    intervention_applied BOOLEAN,
    intervention_date DATETIME,
    intervention_cost_actual REAL,
    revenue_impact REAL,
    recorded_at DATETIME
)
```

### Alerts Table
Tracks system alerts:
```sql
CREATE TABLE alerts (
    alert_id TEXT,
    customer_id TEXT,
    alert_type TEXT,
    alert_message TEXT,
    alert_level TEXT,
    is_acknowledged BOOLEAN,
    acknowledged_by TEXT,
    acknowledged_at DATETIME,
    created_at DATETIME
)
```

---

## Operational Workflows

### Week 1: Initial Integration

**Day 1-2: Setup**
```bash
1. Install integration components
2. Initialize operational database
3. Configure CRM connectors
4. Test with sample customers
```

**Day 3-4: Launch**
```bash
1. Import historical customer data
2. Run batch scoring
3. Verify CRM data population
4. Confirm alert system working
```

**Day 5-7: Monitor**
```bash
1. Track first predictions
2. Monitor alert response
3. Verify data quality
4. Gather team feedback
```

### Week 2+: Run Operations

**Daily Activities:**
```
1. Review new high-risk customers
2. Monitor intervention results with outcomes data
3. Check system health/alerts
4. Update retention campaign targeting
```

**Weekly Reviews:**
```
1. Run cohort performance analysis
2. Calculate weekly ROI
3. Review A/B test results
4. Identify model adjustments
```

**Monthly Activities:**
```
1. Calculate cumulative ROI
2. Detect model drift
3. Plan retention strategy adjustments
4. Prepare quarterly retraining
```

---

## Alert Management

### Alert Levels

| Level | Score | Action | Timeline |
|-------|-------|--------|----------|
| **CRITICAL** | 80+ | Immediate outreach | < 24 hours |
| **HIGH** | 60-79 | VIP retention program | < 48 hours |
| **MEDIUM** | 40-59 | Enhanced engagement | < 7 days |
| **LOW** | < 40 | Standard monitoring | Monitor |

### Acknowledge Alerts

```python
# Mark alert as acknowledged
integration.alerts.acknowledge_alert(
    alert_id='alert_CUST_12345_001',
    acknowledged_by='john.doe@company.com'
)

# Get pending alerts
pending = integration.alerts.get_pending_alerts(alert_level='CRITICAL')
for alert in pending:
    print(f"{alert['alert_level']}: {alert['alert_message']}")
```

---

## Performance Monitoring

### Key Metrics to Track

```
Model Performance:
├─ Accuracy: Target >82%
├─ Precision: Target >80%
├─ Recall: Target >75%
└─ AUC-ROC: Target >0.85

Business Metrics:
├─ Customers Retained: Target 4,000+/month
├─ Revenue Saved: Target $2M+/month
├─ Program Cost: Optimize <$50K/month
└─ ROI: Target 3.0x+ 

Operational:
├─ Predictions/day: Monitor growth
├─ Alert response time: Target <12h
├─ False positive rate: Minimize
└─ Model drift: Monitor for >5% deviation
```

### Setup Alerts

```python
# Alert if accuracy drops below 80%
if metrics['accuracy'] < 0.80:
    send_alert(
        "⚠️ Model accuracy degraded",
        f"Current: {metrics['accuracy']:.1%}, Threshold: 80%",
        severity='high'
    )

# Alert if ROI drops below 2.0x
if metrics['roi'] < 2.0:
    send_alert(
        "⚠️ ROI below target",
        f"Current: {metrics['roi']:.2f}x, Target: 2.0x+",
        severity='high'
    )
```

---

## Troubleshooting

### Issue: Predictions not saved to database

```python
# Check database connection
import sqlite3
conn = sqlite3.connect('operational_db.sqlite')
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM predictions")
print(cursor.fetchone())
```

### Issue: CRM push failing

```python
# Verify CRM credentials
crm = CRMIntegrator(api_endpoint=endpoint, api_key=api_key)
# Test with health check
try:
    crm.push_churn_score(prediction)
except Exception as e:
    print(f"CRM Error: {e}")
    # Check API key, endpoint, network connectivity
```

### Issue: Model drift detected

```python
# If drift detected:
# 1. Investigate data changes
# 2. Check for seasonal patterns
# 3. Consider model retraining if >5% deviation
# 4. Review recent retention program changes

# Trigger retraining
python deployment_script.py --retrain
```

---

## Next Steps

1. ✅ Complete Week 1 integration
2. ✅ Run 2-week pilot program
3. ✅ Collect customer outcome data
4. ✅ Calculate realized ROI
5. → Scale to full operations
6. → Plan quarterly retraining
7. → Expand to new customer segments
8. → Integrate advanced features

---

## Support

For issues or questions:
- Check `integration_service.py` docstrings
- Review monitoring system logs
- See `DEPLOYMENT_GUIDE.md` for deployment details
- Contact: Data Science Team

---

**Status: Ready for Production Integration** ✅

Generated: February 8, 2026
