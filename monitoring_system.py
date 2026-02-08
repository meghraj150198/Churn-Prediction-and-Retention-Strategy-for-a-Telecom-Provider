"""
REAL-TIME MONITORING SYSTEM
============================
Tracks model predictions vs actual outcomes and calculates ROI

This system provides:
1. Prediction accuracy tracking
2. ROI calculation
3. A/B testing framework  
4. Model drift detection
5. Performance dashboarding

Author: Monitoring & Analytics Team
Date: February 8, 2026
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Model performance metrics"""
    timestamp: datetime
    total_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    false_positive_rate: float
    false_negative_rate: float


class MonitoringSystem:
    """Real-time model monitoring and ROI tracking"""
    
    def __init__(self, db_path: str = 'operational_db.sqlite'):
        """Initialize monitoring system"""
        self.db_path = db_path
        self._init_monitoring_tables()
        logger.info(f"✓ Monitoring System initialized")
    
    def _init_monitoring_tables(self):
        """Create monitoring-specific tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Daily metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_metrics (
                metric_date DATE PRIMARY KEY,
                predictions_made INTEGER,
                avg_risk_score REAL,
                critical_risk_count INTEGER,
                high_risk_count INTEGER,
                medium_risk_count INTEGER,
                low_risk_count INTEGER,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                auc_roc REAL,
                recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Cohort performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cohort_performance (
                cohort_id TEXT PRIMARY KEY,
                cohort_name TEXT NOT NULL,
                cohort_date DATE NOT NULL,
                predicted_risk_level TEXT NOT NULL,
                total_customers INTEGER,
                actual_churns INTEGER,
                intervention_applied_count INTEGER,
                intervention_success_count INTEGER,
                revenue_at_risk REAL,
                revenue_saved REAL,
                program_cost REAL,
                net_benefit REAL,
                roi REAL,
                calibration_score REAL,
                recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model drift detection table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_drift (
                drift_id TEXT PRIMARY KEY,
                check_date DATE NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                baseline_value REAL NOT NULL,
                deviation_pct REAL NOT NULL,
                is_drift BOOLEAN NOT NULL,
                severity TEXT,
                recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_prediction_metrics(self, cohort_name: str, 
                                    cohort_date: datetime) -> Optional[Dict]:
        """Calculate metrics for a customer cohort"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get predictions for cohort
            cursor.execute('''
                SELECT p.customer_id, p.risk_score, p.risk_level, 
                       p.intervention_cost, p.revenue_at_risk,
                       o.actual_churn, o.intervention_applied,
                       o.intervention_cost_actual, o.revenue_impact
                FROM predictions p
                LEFT JOIN outcomes o ON p.customer_id = o.customer_id
                WHERE DATE(p.timestamp) = DATE(?)
            ''', (cohort_date,))
            
            results = cursor.fetchall()
            
            if not results:
                logger.warning(f"No predictions found for {cohort_date}")
                return None
            
            # Convert to DataFrame for analysis
            columns = ['customer_id', 'risk_score', 'risk_level', 'intervention_cost',
                      'revenue_at_risk', 'actual_churn', 'intervention_applied',
                      'intervention_cost_actual', 'revenue_impact']
            df = pd.DataFrame(results, columns=columns)
            
            # Calculate metrics
            metrics = self._compute_metrics(df)
            metrics['cohort_name'] = cohort_name
            metrics['cohort_date'] = cohort_date
            
            # Calculate ROI for this cohort
            total_revenue_at_risk = df['revenue_at_risk'].sum()
            actual_churns = df['actual_churn'].sum()
            retained = len(df) - actual_churns
            revenue_saved = retained * 600  # Avg annual value
            total_cost = df['intervention_cost'].sum()
            roi = (revenue_saved - total_cost) / total_cost if total_cost > 0 else 0
            
            metrics['roi'] = roi
            metrics['revenue_saved'] = revenue_saved
            metrics['revenue_at_risk'] = total_revenue_at_risk
            
            conn.close()
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating prediction metrics: {e}")
            return None
    
    def _compute_metrics(self, df: pd.DataFrame) -> Dict:
        """Compute classification metrics"""
        try:
            # For predictions with outcomes
            df_with_outcomes = df[df['actual_churn'].notna()]
            
            if len(df_with_outcomes) == 0:
                logger.warning("No outcomes available for metric calculation")
                return {}
            
            # Convert risk_score to binary prediction (threshold = 50)
            y_pred = (df_with_outcomes['risk_score'] >= 50).astype(int)
            y_true = df_with_outcomes['actual_churn'].astype(int)
            
            # Accuracy
            accuracy = (y_pred == y_true).mean()
            
            # Precision (TP / (TP + FP))
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Recall (TP / (TP + FN))
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # False positive and negative rates
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            return {
                'total_predictions': len(df),
                'predictions_with_outcomes': len(df_with_outcomes),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'false_positive_rate': fpr,
                'false_negative_rate': fnr,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            }
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {}
    
    def detect_model_drift(self, baseline_metrics: Dict, 
                          current_metrics: Dict,
                          threshold_pct: float = 5.0) -> List[Dict]:
        """Detect model drift by comparing metrics"""
        try:
            drifts = []
            drift_date = datetime.now().date()
            
            metrics_to_check = ['accuracy', 'precision', 'recall', 'auc_roc']
            
            for metric in metrics_to_check:
                if metric in baseline_metrics and metric in current_metrics:
                    baseline = baseline_metrics[metric]
                    current = current_metrics.get(metric, baseline)
                    
                    if baseline > 0:
                        deviation_pct = abs(current - baseline) / baseline * 100
                    else:
                        deviation_pct = 0
                    
                    is_drift = deviation_pct > threshold_pct
                    severity = 'critical' if deviation_pct > 10 else 'warning' if is_drift else 'normal'
                    
                    if is_drift:
                        drift_record = {
                            'drift_id': f"drift_{metric}_{drift_date}_{datetime.now().timestamp()}",
                            'check_date': drift_date,
                            'metric_name': metric,
                            'metric_value': current,
                            'baseline_value': baseline,
                            'deviation_pct': deviation_pct,
                            'is_drift': True,
                            'severity': severity
                        }
                        drifts.append(drift_record)
                        
                        logger.warning(f"⚠️ DRIFT DETECTED: {metric} = {current:.3f} (baseline: {baseline:.3f}, deviation: {deviation_pct:.1f}%)")
            
            return drifts
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return []
    
    def get_ab_test_results(self, control_cohort: str, treatment_cohort: str) -> Dict:
        """Compare performance between two cohorts (A/B testing)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            results = {}
            
            for cohort_type, cohort_name in [('control', control_cohort), 
                                             ('treatment', treatment_cohort)]:
                cursor.execute('''
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN o.actual_churn THEN 1 ELSE 0 END) as churned,
                           SUM(o.revenue_impact) as revenue_impact,
                           AVG(p.revenue_at_risk) as avg_revenue_at_risk
                    FROM predictions p
                    LEFT JOIN outcomes o ON p.customer_id = o.customer_id
                    WHERE p.risk_level = ?
                ''', (cohort_name,))
                
                row = cursor.fetchone()
                total, churned, revenue_impact, avg_risk = row
                
                churn_rate = (churned / total if total > 0 else 0)
                
                results[cohort_type] = {
                    'total_customers': total or 0,
                    'churned': churned or 0,
                    'churn_rate': churn_rate,
                    'revenue_impact': float(revenue_impact or 0),
                    'avg_revenue_at_risk': float(avg_risk or 0)
                }
            
            # Calculate significance
            control = results['control']
            treatment = results['treatment']
            
            if control['total_customers'] > 0 and treatment['total_customers'] > 0:
                # Chi-square test for churn rate difference
                contingency = [
                    [control['churned'], control['total_customers'] - control['churned']],
                    [treatment['churned'], treatment['total_customers'] - treatment['churned']]
                ]
                chi2, p_value = stats.chi2_contingency(contingency)[:2]
                
                results['statistical_test'] = {
                    'test': 'chi_square',
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'is_significant': p_value < 0.05
                }
            
            conn.close()
            return results
        except Exception as e:
            logger.error(f"Error calculating A/B test results: {e}")
            return {}
    
    def calculate_cumulative_roi(self, start_date: datetime, 
                                end_date: datetime) -> Dict:
        """Calculate cumulative ROI over time period"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(DISTINCT p.customer_id) as total_scored,
                    SUM(p.revenue_at_risk) as total_revenue_at_risk,
                    COUNT(DISTINCT CASE WHEN o.actual_churn THEN p.customer_id END) as actual_churned,
                    SUM(p.intervention_cost) as total_intervention_cost,
                    SUM(o.revenue_impact) as total_revenue_saved
                FROM predictions p
                LEFT JOIN outcomes o ON p.customer_id = o.customer_id
                WHERE DATE(p.timestamp) BETWEEN ? AND ?
            ''', (start_date.date(), end_date.date()))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                total_scored, total_revenue_at_risk, actual_churned, total_cost, total_saved = result
                
                # Calculate metrics
                customers_retained = (total_scored or 0) - (actual_churned or 0)
                revenue_saved = total_saved or (customers_retained * 600)  # Est. $600/customer/year
                net_benefit = (revenue_saved or 0) - (total_cost or 0)
                roi = (net_benefit / (total_cost or 1)) if total_cost else 0
                
                return {
                    'period': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    },
                    'customers': {
                        'total_scored': total_scored or 0,
                        'actual_churned': actual_churned or 0,
                        'retained': customers_retained
                    },
                    'financials': {
                        'revenue_at_risk': float(total_revenue_at_risk or 0),
                        'revenue_saved': float(revenue_saved or 0),
                        'program_cost': float(total_cost or 0),
                        'net_benefit': float(net_benefit or 0),
                        'roi': float(roi)
                    }
                }
            return {}
        except Exception as e:
            logger.error(f"Error calculating cumulative ROI: {e}")
            return {}
    
    def get_monitoring_dashboard_data(self) -> Dict:
        """Get all data for monitoring dashboard"""
        try:
            # Last 30 days ROI
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            roi_30d = self.calculate_cumulative_roi(start_date, end_date)
            
            # Last 7 days ROI
            start_date = end_date - timedelta(days=7)
            roi_7d = self.calculate_cumulative_roi(start_date, end_date)
            
            # Last 24 hours ROI
            start_date = end_date - timedelta(hours=24)
            roi_24h = self.calculate_cumulative_roi(start_date, end_date)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get latest metrics
            cursor.execute('''
                SELECT COUNT(*) as total_predictions,
                       AVG(risk_score) as avg_risk_score,
                       SUM(CASE WHEN risk_level = 'Critical Risk' THEN 1 ELSE 0 END) as critical_count
                FROM predictions
                WHERE timestamp > datetime('now', '-24 hours')
            ''')
            
            daily_stats = cursor.fetchone()
            conn.close()
            
            return {
                'roi': {
                    'last_24h': roi_24h,
                    'last_7d': roi_7d,
                    'last_30d': roi_30d
                },
                'daily_activity': {
                    'predictions_last_24h': daily_stats[0] or 0,
                    'avg_risk_score': float(daily_stats[1] or 0),
                    'critical_risk': daily_stats[2] or 0
                },
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}


def main():
    """Example usage"""
    print("=" * 80)
    print("MONITORING SYSTEM")
    print("=" * 80)
    
    monitoring = MonitoringSystem()
    
    # Get monitoring dashboard
    print("\n[1] Monitoring Dashboard Data:")
    dashboard = monitoring.get_monitoring_dashboard_data()
    print(f"  - Predictions (24h): {dashboard.get('daily_activity', {}).get('predictions_last_24h', 0)}")
    print(f"  - Avg Risk Score: {dashboard.get('daily_activity', {}).get('avg_risk_score', 0):.1f}")
    
    # Calculate cumulative ROI (last 30 days)
    print("\n[2] Cumulative ROI (Last 30 Days):")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    roi = monitoring.calculate_cumulative_roi(start_date, end_date)
    
    if roi:
        financials = roi.get('financials', {})
        print(f"  - Revenue Saved: ${financials.get('revenue_saved', 0):,.0f}")
        print(f"  - Program Cost: ${financials.get('program_cost', 0):,.0f}")
        print(f"  - Net Benefit: ${financials.get('net_benefit', 0):,.0f}")
        print(f"  - ROI: {financials.get('roi', 0):.2f}x")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
