"""
PRODUCTION CHURN SCORER
========================
Reusable module for scoring new/batch customers in production

This module provides:
1. ChurnScorer class - Load model and score new customers
2. Batch scoring functionality - Score CSV files
3. Real-time scoring - Score individual customers
4. Risk categorization - Assign risk levels and actions
5. Model management - Load/reload trained models

Usage Examples:
    # Load scorer
    scorer = ChurnScorer('ensemble_final_model_20260208_120000.pkl',
                        'feature_scaler_20260208_120000.pkl',
                        'feature_names_20260208_120000.pkl')
    
    # Score batch
    scores = scorer.score_batch('new_customers.csv')
    scores.to_csv('new_customers_with_scores.csv', index=False)
    
    # Score single customer
    score = scorer.score_customer(customer_dict)
    print(f"Risk: {score['risk_category']} ({score['risk_score']}/100)")

Author: Churn Prediction Team
Date: February 8, 2026
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ChurnScorer:
    """Production-ready churn risk scorer"""
    
    def __init__(self, model_path: str, scaler_path: str, features_path: str):
        """
        Initialize scorer with trained model and preprocessing objects
        
        Args:
            model_path: Path to trained ensemble model (.pkl)
            scaler_path: Path to feature scaler (.pkl)
            features_path: Path to feature names (.pkl)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.features_path = features_path
        
        # Load artifacts
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(features_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        
        print(f"✓ Scorer initialized")
        print(f"  - Model: {model_path}")
        print(f"  - Features: {len(self.feature_names)} dimensions")
    
    def _risk_score_to_category(self, score: int) -> str:
        """Convert numeric score to risk category"""
        if score < 25:
            return 'Low Risk'
        elif score < 50:
            return 'Medium Risk'
        elif score < 75:
            return 'High Risk'
        else:
            return 'Critical Risk'
    
    def _risk_score_to_action(self, score: int) -> str:
        """Convert score to recommended action"""
        if score < 25:
            return 'Standard monitoring'
        elif score < 50:
            return 'Enhanced engagement offers'
        elif score < 75:
            return 'VIP retention program'
        else:
            return 'Urgent intervention required'
    
    def _risk_score_to_cost(self, score: int) -> float:
        """Estimate intervention cost based on score"""
        if score < 25:
            return 0.0
        elif score < 50:
            return 25.0
        elif score < 75:
            return 75.0
        else:
            return 150.0
    
    def score_customer(self, customer_data: Dict) -> Dict:
        """
        Score a single customer
        
        Args:
            customer_data: Dictionary with customer features
        
        Returns:
            Dictionary with scoring results including:
            - risk_probability: Churn probability (0-1)
            - risk_score: Score 0-100
            - risk_category: Category (Low/Medium/High/Critical)
            - retention_action: Recommended action
            - estimated_cost: Intervention cost estimate
        """
        try:
            # Prepare dataframe with single row
            customer_df = pd.DataFrame([customer_data])
            
            # Ensure all features present
            for feature in self.feature_names:
                if feature not in customer_df.columns:
                    customer_df[feature] = 0
            
            # Select and order features
            X = customer_df[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get prediction
            churn_prob = self.model.predict_proba(X_scaled)[0, 1]
            risk_score = int(churn_prob * 100)
            
            # Generate all outputs
            return {
                'risk_probability': round(churn_prob, 4),
                'risk_score': risk_score,
                'risk_category': self._risk_score_to_category(risk_score),
                'retention_action': self._risk_score_to_action(risk_score),
                'estimated_intervention_cost': self._risk_score_to_cost(risk_score)
            }
        
        except Exception as e:
            print(f"❌ Error scoring customer: {str(e)}")
            return None
    
    def score_batch(self, csv_path: str, sample: int = None) -> pd.DataFrame:
        """
        Score batch of customers from CSV
        
        Args:
            csv_path: Path to CSV file with customer features
            sample: Optional sample size (for testing)
        
        Returns:
            DataFrame with original data + scoring results
        """
        print(f"\n📊 Loading batch file: {csv_path}")
        
        # Load data
        df = pd.read_csv(csv_path)
        
        if sample:
            df = df.sample(n=min(sample, len(df)), random_state=42)
        
        print(f"  - Records to score: {len(df):,}")
        
        # Ensure all features present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Prepare features
        X = df[self.feature_names].copy()
        
        print(f"  - Features: {len(self.feature_names)}")
        
        # Scale features
        print(f"  - Scaling features...")
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions
        print(f"  - Scoring batch...")
        churn_probabilities = self.model.predict_proba(X_scaled)[:, 1]
        risk_scores = (churn_probabilities * 100).astype(int)
        
        # Add scoring results
        df['churn_probability'] = churn_probabilities
        df['risk_score'] = risk_scores
        df['risk_category'] = risk_scores.apply(self._risk_score_to_category)
        df['retention_action'] = risk_scores.apply(self._risk_score_to_action)
        df['estimated_intervention_cost'] = risk_scores.apply(self._risk_score_to_cost)
        
        # Summary statistics
        print(f"\n✓ Batch scoring complete:")
        print(f"  - Critical Risk: {(risk_scores >= 75).sum():,}")
        print(f"  - High Risk: {((risk_scores >= 50) & (risk_scores < 75)).sum():,}")
        print(f"  - Medium Risk: {((risk_scores >= 25) & (risk_scores < 50)).sum():,}")
        print(f"  - Low Risk: {(risk_scores < 25).sum():,}")
        print(f"  - Avg Score: {risk_scores.mean():.1f}")
        
        return df
    
    def segment_batch(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Segment scored customers into risk categories
        
        Args:
            df: DataFrame with 'risk_category' column
        
        Returns:
            Dictionary with DataFrames for each segment
        """
        segments = {
            'critical': df[df['risk_category'] == 'Critical Risk'],
            'high': df[df['risk_category'] == 'High Risk'],
            'medium': df[df['risk_category'] == 'Medium Risk'],
            'low': df[df['risk_category'] == 'Low Risk']
        }
        
        print(f"\n📊 Segmentation Summary:")
        for segment, data in segments.items():
            print(f"  - {segment.upper()}: {len(data):,} customers")
        
        return segments
    
    def generate_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary report from scored data
        
        Args:
            df: DataFrame with scoring results
        
        Returns:
            Dictionary with summary statistics
        """
        risk_scores = df['risk_score']
        
        report = {
            'total_customers': len(df),
            'avg_risk_score': float(risk_scores.mean()),
            'min_risk_score': int(risk_scores.min()),
            'max_risk_score': int(risk_scores.max()),
            'std_dev': float(risk_scores.std()),
            'patients_by_category': df['risk_category'].value_counts().to_dict(),
            'total_estimated_cost': float(df['estimated_intervention_cost'].sum()),
            'priority_interventions': int((risk_scores >= 75).sum()),
            'total_at_risk': int((risk_scores >= 50).sum())
        }
        
        return report
    
    def export_for_crm(self, df: pd.DataFrame, output_path: str, 
                       segment: str = None) -> str:
        """
        Export data in CRM-friendly format
        
        Args:
            df: DataFrame with scoring results
            output_path: Output file path
            segment: Optional segment filter (critical/high/medium/low)
        
        Returns:
            Path to exported file
        """
        if segment:
            risk_map = {
                'critical': 'Critical Risk',
                'high': 'High Risk',
                'medium': 'Medium Risk',
                'low': 'Low Risk'
            }
            df = df[df['risk_category'] == risk_map.get(segment, segment)]
        
        # Select relevant columns for CRM
        export_cols = ['churn_probability', 'risk_score', 'risk_category', 
                      'retention_action', 'estimated_intervention_cost']
        
        # Add any ID/name columns if present
        id_cols = [c for c in df.columns if 'id' in c.lower() or 'name' in c.lower()]
        export_cols = id_cols + export_cols
        
        df_export = df[[c for c in export_cols if c in df.columns]]
        df_export.to_csv(output_path, index=False)
        
        print(f"✓ Exported {len(df_export):,} records to {output_path}")
        return output_path


def main():
    """Example usage"""
    print("=" * 80)
    print("PRODUCTION CHURN SCORER - EXAMPLE USAGE")
    print("=" * 80)
    
    # Example 1: Load scorer
    print("\n[EXAMPLE 1] Loading scorer...")
    try:
        scorer = ChurnScorer(
            'ensemble_final_model_20260208_120000.pkl',
            'feature_scaler_20260208_120000.pkl',
            'feature_names_20260208_120000.pkl'
        )
    except FileNotFoundError:
        print("❌ Model files not found. Run deployment_script.py first.")
        return
    
    # Example 2: Score batch from main customer file
    print("\n[EXAMPLE 2] Scoring batch from main dataset...")
    try:
        scored_df = scorer.score_batch('telecom_churn_engineered.csv', sample=100)
        print(scored_df[['risk_score', 'risk_category', 'retention_action']].head(10))
    except FileNotFoundError:
        print("❌ Data file not found.")
        return
    
    # Example 3: Segment and export
    print("\n[EXAMPLE 3] Exporting by segment...")
    segments = scorer.segment_batch(scored_df)
    for segment_name, segment_data in segments.items():
        if len(segment_data) > 0:
            output_file = f'crm_export_{segment_name}_risk.csv'
            scorer.export_for_crm(scored_df, output_file, segment_name)
    
    # Example 4: Generate report
    print("\n[EXAMPLE 4] Generating summary report...")
    report = scorer.generate_report(scored_df)
    print(f"  - Total customers: {report['total_customers']:,}")
    print(f"  - Avg risk score: {report['avg_risk_score']:.1f}")
    print(f"  - Priority interventions: {report['priority_interventions']:,}")
    print(f"  - Total estimated cost: ${report['total_estimated_cost']:,.0f}")


if __name__ == '__main__':
    main()
