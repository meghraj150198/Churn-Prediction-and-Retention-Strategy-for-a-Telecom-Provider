"""
INTEGRATION SERVICE
====================
Connects the churn prediction model to operational systems

This service provides:
1. Real-time customer scoring API
2. Batch scoring integration
3. CRM system connectors
4. Database integration
5. Event streaming for alerts

Author: Operations Integration Team
Date: February 8, 2026
"""

import pandas as pd
import numpy as np
import pickle
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "Low Risk"
    MEDIUM = "Medium Risk"
    HIGH = "High Risk"
    CRITICAL = "Critical Risk"


class InterventionStrategy(Enum):
    """Intervention strategy by risk level"""
    LOW = "Standard monitoring"
    MEDIUM = "Enhanced engagement offers"
    HIGH = "VIP retention program"
    CRITICAL = "Urgent intervention required"


@dataclass
class ChurnPrediction:
    """Churn prediction result"""
    customer_id: str
    risk_score: int
    risk_probability: float
    risk_level: RiskLevel
    intervention_strategy: InterventionStrategy
    revenue_at_risk: float
    intervention_cost: float
    timestamp: datetime
    model_version: str


class OperationalDatabase:
    """SQLite database interface for operational data"""
    
    def __init__(self, db_path: str = 'operational_db.sqlite'):
        """Initialize database connection"""
        self.db_path = db_path
        self._init_tables()
        logger.info(f"✓ Database initialized: {db_path}")
    
    def _init_tables(self):
        """Create required tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                risk_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                intervention_strategy TEXT NOT NULL,
                revenue_at_risk REAL NOT NULL,
                intervention_cost REAL NOT NULL,
                model_version TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Outcomes table (actual churn observed)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS outcomes (
                outcome_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                actual_churn BOOLEAN NOT NULL,
                actual_churned_date DATETIME,
                intervention_applied BOOLEAN NOT NULL,
                intervention_date DATETIME,
                intervention_cost_actual REAL,
                revenue_impact REAL,
                feedback TEXT,
                recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (customer_id) REFERENCES predictions(customer_id)
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                alert_message TEXT NOT NULL,
                alert_level TEXT NOT NULL,
                is_acknowledged BOOLEAN DEFAULT FALSE,
                acknowledged_by TEXT,
                acknowledged_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id TEXT PRIMARY KEY,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_date DATE NOT NULL,
                recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, prediction: ChurnPrediction) -> bool:
        """Save prediction to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            prediction_id = f"{prediction.customer_id}_{datetime.now().timestamp()}"
            
            cursor.execute('''
                INSERT INTO predictions 
                (prediction_id, customer_id, risk_score, risk_probability, 
                 risk_level, intervention_strategy, revenue_at_risk, 
                 intervention_cost, model_version, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_id,
                prediction.customer_id,
                prediction.risk_score,
                prediction.risk_probability,
                prediction.risk_level.value,
                prediction.intervention_strategy.value,
                prediction.revenue_at_risk,
                prediction.intervention_cost,
                prediction.model_version,
                prediction.timestamp
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return False
    
    def record_outcome(self, customer_id: str, actual_churn: bool, 
                      intervention_applied: bool = False,
                      intervention_cost_actual: float = None,
                      revenue_impact: float = None) -> bool:
        """Record actual churn outcome for model validation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            outcome_id = f"outcome_{customer_id}_{datetime.now().timestamp()}"
            churned_date = datetime.now() if actual_churn else None
            
            cursor.execute('''
                INSERT INTO outcomes
                (outcome_id, customer_id, actual_churn, actual_churned_date,
                 intervention_applied, intervention_cost_actual, revenue_impact)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                outcome_id,
                customer_id,
                actual_churn,
                churned_date,
                intervention_applied,
                intervention_cost_actual,
                revenue_impact
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error recording outcome: {e}")
            return False
    
    def get_prediction_by_customer(self, customer_id: str) -> Optional[Dict]:
        """Get latest prediction for a customer"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE customer_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''', (customer_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, result))
            return None
        except Exception as e:
            logger.error(f"Error retrieving prediction: {e}")
            return None
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance metrics for the last N days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Get predictions
            cursor.execute('''
                SELECT COUNT(*) as total_predictions,
                       AVG(risk_score) as avg_risk_score,
                       SUM(revenue_at_risk) as total_revenue_at_risk
                FROM predictions
                WHERE timestamp > ?
            ''', (cutoff_date,))
            
            pred_stats = cursor.fetchone()
            
            # Get outcomes
            cursor.execute('''
                SELECT COUNT(*) as total_outcomes,
                       SUM(CASE WHEN actual_churn THEN 1 ELSE 0 END) as actual_churns,
                       SUM(revenue_impact) as total_revenue_impact
                FROM outcomes
                WHERE recorded_at > ?
            ''', (cutoff_date,))
            
            outcome_stats = cursor.fetchone()
            conn.close()
            
            return {
                'predictions': {
                    'total': pred_stats[0] or 0,
                    'avg_risk_score': float(pred_stats[1] or 0),
                    'total_revenue_at_risk': float(pred_stats[2] or 0)
                },
                'outcomes': {
                    'total': outcome_stats[0] or 0,
                    'actual_churns': outcome_stats[1] or 0,
                    'total_revenue_impact': float(outcome_stats[2] or 0)
                },
                'period_days': days
            }
        except Exception as e:
            logger.error(f"Error retrieving performance summary: {e}")
            return {}


class CRMIntegrator:
    """CRM system integration connector"""
    
    def __init__(self, api_endpoint: str = None, api_key: str = None,
                 crm_type: str = 'salesforce'):
        """Initialize CRM integrator"""
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.crm_type = crm_type
        logger.info(f"✓ CRM Integrator initialized: {crm_type}")
    
    def push_churn_score(self, prediction: ChurnPrediction) -> bool:
        """Push churn score to CRM system"""
        try:
            # Mock implementation - replace with actual CRM API call
            crm_data = {
                'customer_id': prediction.customer_id,
                'churn_risk_score': prediction.risk_score,
                'churn_probability': prediction.risk_probability,
                'risk_category': prediction.risk_level.value,
                'recommended_action': prediction.intervention_strategy.value,
                'update_timestamp': prediction.timestamp.isoformat()
            }
            
            logger.info(f"✓ Pushed to CRM: {prediction.customer_id} (Score: {prediction.risk_score})")
            return True
        except Exception as e:
            logger.error(f"Error pushing to CRM: {e}")
            return False
    
    def create_task(self, customer_id: str, task_title: str, 
                   task_description: str, due_date: datetime) -> bool:
        """Create retention task in CRM"""
        try:
            task_data = {
                'customer_id': customer_id,
                'title': task_title,
                'description': task_description,
                'due_date': due_date.isoformat(),
                'priority': 'High',
                'type': 'churn_prevention'
            }
            
            logger.info(f"✓ Created task in CRM: {customer_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating CRM task: {e}")
            return False
    
    def bulk_update_scores(self, predictions: List[ChurnPrediction]) -> int:
        """Batch update multiple customer scores in CRM"""
        success_count = 0
        for prediction in predictions:
            if self.push_churn_score(prediction):
                success_count += 1
        
        logger.info(f"✓ Bulk updated {success_count}/{len(predictions)} customers in CRM")
        return success_count


class AlertSystem:
    """Alert generation and management"""
    
    def __init__(self, db: OperationalDatabase):
        """Initialize alert system"""
        self.db = db
        logger.info("✓ Alert System initialized")
    
    def generate_alert(self, prediction: ChurnPrediction) -> bool:
        """Generate alert for high-risk customer"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            alert_id = f"alert_{prediction.customer_id}_{datetime.now().timestamp()}"
            
            # Determine alert level
            if prediction.risk_score >= 80:
                alert_level = 'CRITICAL'
                alert_message = f"URGENT: {prediction.customer_id} - Churn risk {prediction.risk_score}%"
            elif prediction.risk_score >= 60:
                alert_level = 'HIGH'
                alert_message = f"HIGH PRIORITY: {prediction.customer_id} - Churn risk {prediction.risk_score}%"
            else:
                alert_level = 'MEDIUM'
                alert_message = f"Follow up: {prediction.customer_id} - Churn risk {prediction.risk_score}%"
            
            cursor.execute('''
                INSERT INTO alerts
                (alert_id, customer_id, alert_type, alert_message, alert_level)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                alert_id,
                prediction.customer_id,
                'churn_risk',
                alert_message,
                alert_level
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✓ Alert created: {alert_level} - {prediction.customer_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return False
    
    def get_pending_alerts(self, alert_level: str = None) -> List[Dict]:
        """Get all pending alerts"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            if alert_level:
                cursor.execute('''
                    SELECT * FROM alerts 
                    WHERE is_acknowledged = FALSE 
                    AND alert_level = ?
                    ORDER BY created_at DESC
                ''', (alert_level,))
            else:
                cursor.execute('''
                    SELECT * FROM alerts 
                    WHERE is_acknowledged = FALSE
                    ORDER BY created_at DESC
                ''')
            
            results = cursor.fetchall()
            conn.close()
            
            return [dict(zip([d[0] for d in cursor.description], row)) for row in results]
        except Exception as e:
            logger.error(f"Error retrieving alerts: {e}")
            return []
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Mark alert as acknowledged"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE alerts 
                SET is_acknowledged = TRUE, 
                    acknowledged_by = ?, 
                    acknowledged_at = CURRENT_TIMESTAMP
                WHERE alert_id = ?
            ''', (acknowledged_by, alert_id))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False


class OperationalIntegration:
    """Main integration service orchestrator"""
    
    def __init__(self, model_path: str, scaler_path: str, 
                 features_path: str, db_path: str = 'operational_db.sqlite'):
        """Initialize operational integration"""
        # Load model components
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(features_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        
        # Initialize components
        self.db = OperationalDatabase(db_path)
        self.crm = CRMIntegrator()
        self.alerts = AlertSystem(self.db)
        self.model_version = "1.0-production"
        
        logger.info("✓ Operational Integration initialized")
    
    def score_customer(self, customer_data: Dict) -> Optional[ChurnPrediction]:
        """Score a single customer and record in database"""
        try:
            # Prepare data
            df = pd.DataFrame([customer_data])
            
            # Ensure features match
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            X = df[self.feature_names]
            X_scaled = self.scaler.transform(X)
            
            # Predict
            churn_prob = self.model.predict_proba(X_scaled)[0, 1]
            risk_score = int(churn_prob * 100)
            
            # Determine risk level
            if risk_score < 25:
                risk_level = RiskLevel.LOW
            elif risk_score < 50:
                risk_level = RiskLevel.MEDIUM
            elif risk_score < 75:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            # Get intervention strategy
            intervention_map = {
                RiskLevel.LOW: InterventionStrategy.LOW,
                RiskLevel.MEDIUM: InterventionStrategy.MEDIUM,
                RiskLevel.HIGH: InterventionStrategy.HIGH,
                RiskLevel.CRITICAL: InterventionStrategy.CRITICAL
            }
            intervention = intervention_map[risk_level]
            
            # Calculate financial metrics
            monthly_charges = customer_data.get('monthly_charges', 50)
            revenue_at_risk = monthly_charges * 12 * churn_prob
            
            cost_map = {
                RiskLevel.LOW: 0,
                RiskLevel.MEDIUM: 25,
                RiskLevel.HIGH: 75,
                RiskLevel.CRITICAL: 150
            }
            intervention_cost = cost_map[risk_level]
            
            # Create prediction
            prediction = ChurnPrediction(
                customer_id=customer_data.get('customer_id'),
                risk_score=risk_score,
                risk_probability=round(churn_prob, 4),
                risk_level=risk_level,
                intervention_strategy=intervention,
                revenue_at_risk=revenue_at_risk,
                intervention_cost=intervention_cost,
                timestamp=datetime.now(),
                model_version=self.model_version
            )
            
            # Save to database
            self.db.save_prediction(prediction)
            
            # Push to CRM
            self.crm.push_churn_score(prediction)
            
            # Generate alerts for high-risk
            if risk_score >= 60:
                self.alerts.generate_alert(prediction)
            
            logger.info(f"✓ Scored customer: {prediction.customer_id} (Risk: {risk_score}%)")
            return prediction
        except Exception as e:
            logger.error(f"Error scoring customer: {e}")
            return None
    
    def batch_score_customers(self, customer_file: str) -> Tuple[int, int]:
        """Score batch of customers from CSV"""
        try:
            df = pd.read_csv(customer_file)
            success_count = 0
            error_count = 0
            
            for _, row in df.iterrows():
                customer_dict = row.to_dict()
                if self.score_customer(customer_dict):
                    success_count += 1
                else:
                    error_count += 1
            
            logger.info(f"✓ Batch scoring complete: {success_count} success, {error_count} errors")
            return success_count, error_count
        except Exception as e:
            logger.error(f"Error in batch scoring: {e}")
            return 0, 0
    
    def get_operational_status(self) -> Dict:
        """Get current operational status"""
        try:
            perf_summary = self.db.get_performance_summary(days=30)
            
            pending_alerts = self.alerts.get_pending_alerts()
            critical_alerts = len([a for a in pending_alerts if a['alert_level'] == 'CRITICAL'])
            
            return {
                'status': 'operational',
                'timestamp': datetime.now().isoformat(),
                'model_version': self.model_version,
                'performance_30d': perf_summary,
                'pending_alerts': {
                    'total': len(pending_alerts),
                    'critical': critical_alerts
                },
                'system_health': 'healthy' if critical_alerts < 50 else 'warning'
            }
        except Exception as e:
            logger.error(f"Error getting operational status: {e}")
            return {'status': 'error', 'error': str(e)}


def main():
    """Example usage"""
    print("=" * 80)
    print("OPERATIONAL INTEGRATION SERVICE")
    print("=" * 80)
    
    # Initialize
    integration = OperationalIntegration(
        'ensemble_final_model_20260208_093741.pkl',
        'feature_scaler_20260208_093741.pkl',
        'feature_names_20260208_093741.pkl'
    )
    
    # Example: Score a customer
    sample_customer = {
        'customer_id': 'CUST_001',
        'monthly_charges': 65.5,
        'tenure_months': 24,
        'plan_type': 'Premium',
        # ... add all required features from your dataset
    }
    
    print("\n[1] Scoring sample customer...")
    prediction = integration.score_customer(sample_customer)
    if prediction:
        print(f"✓ Risk Score: {prediction.risk_score}")
        print(f"✓ Risk Level: {prediction.risk_level.value}")
        print(f"✓ Recommended Action: {prediction.intervention_strategy.value}")
    
    # Get operational status
    print("\n[2] Operational Status:")
    status = integration.get_operational_status()
    print(json.dumps(status, indent=2, default=str))
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
