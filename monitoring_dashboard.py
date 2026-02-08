"""
MONITORING DASHBOARD WEB SERVICE
=================================
Serves real-time monitoring dashboard via HTTP

Author: Operations Team
Date: February 8, 2026
"""

from flask import Flask, render_template_string, jsonify
from flask_cors import CORS
import json
from datetime import datetime, timedelta
import sqlite3
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = 'operational_db.sqlite'


def get_dashboard_metrics():
    """Get metrics for dashboard"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get predictions stats
        cursor.execute('''
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN risk_level = 'Critical Risk' THEN 1 ELSE 0 END) as critical,
                   SUM(CASE WHEN risk_level = 'High Risk' THEN 1 ELSE 0 END) as high,
                   SUM(CASE WHEN risk_level = 'Medium Risk' THEN 1 ELSE 0 END) as medium,
                   SUM(CASE WHEN risk_level = 'Low Risk' THEN 1 ELSE 0 END) as low,
                   AVG(risk_score) as avg_score,
                   SUM(revenue_at_risk) as total_revenue_at_risk,
                   SUM(intervention_cost) as total_intervention_cost
            FROM predictions
            WHERE timestamp > datetime('now', '-30 days')
        ''')
        
        pred_stats = cursor.fetchone()
        
        # Get outcome stats
        cursor.execute('''
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN actual_churn THEN 1 ELSE 0 END) as churned,
                   SUM(revenue_impact) as revenue_saved
            FROM outcomes
            WHERE recorded_at > datetime('now', '-30 days')
        ''')
        
        outcome_stats = cursor.fetchone()
        
        # Get pending alerts
        cursor.execute('''
            SELECT COUNT(*) as critical_alerts
            FROM alerts
            WHERE is_acknowledged = FALSE
            AND alert_level = 'CRITICAL'
        ''')
        
        alerts = cursor.fetchone()
        
        conn.close()
        
        total_pred, critical, high, medium, low, avg_score, total_revenue, total_cost = pred_stats
        total_outcomes, churned, revenue_saved = outcome_stats
        critical_alerts = alerts[0] if alerts else 0
        
        # Calculate ROI
        revenue_saved = revenue_saved or 0
        net_benefit = revenue_saved - (total_cost or 0)
        roi = (net_benefit / (total_cost or 1)) if total_cost else 0
        
        return {
            'predictions': {
                'total': total_pred or 0,
                'critical': critical or 0,
                'high': high or 0,
                'medium': medium or 0,
                'low': low or 0,
                'avg_score': float(avg_score or 0)
            },
            'outcomes': {
                'total': total_outcomes or 0,
                'churned': churned or 0,
                'saved': (total_pred or 0) - (churned or 0)
            },
            'financials': {
                'revenue_at_risk': float(total_revenue or 0),
                'program_cost': float(total_cost or 0),
                'revenue_saved': float(revenue_saved or 0),
                'net_benefit': float(net_benefit or 0),
                'roi': float(roi)
            },
            'alerts': {
                'critical': critical_alerts
            },
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {e}")
        return {}


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Churn Prediction - Operational Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }
        
        .header p {
            color: #666;
            font-size: 14px;
        }
        
        .status-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-top: 10px;
        }
        
        .status-operational {
            background: #10b981;
            color: white;
        }
        
        .status-warning {
            background: #f59e0b;
            color: white;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .metric-card h3 {
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 15px;
            letter-spacing: 1px;
        }
        
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        
        .metric-change {
            font-size: 12px;
            color: #10b981;
        }
        
        .metric-change.negative {
            color: #ef4444;
        }
        
        .risk-distribution {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        
        .risk-bar {
            flex: 1;
            height: 30px;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 11px;
            font-weight: bold;
        }
        
        .risk-critical { background: #dc2626; }
        .risk-high { background: #f97316; }
        .risk-medium { background: #eab308; }
        .risk-low { background: #10b981; }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .chart-container {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .chart-title {
            font-size: 16px;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
        }
        
        .alerts-panel {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .alert-item {
            padding: 15px;
            border-left: 4px solid #dc2626;
            background: #fef2f2;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        
        .alert-item.high {
            border-left-color: #f97316;
            background: #fffbeb;
        }
        
        .alert-item.medium {
            border-left-color: #eab308;
            background: #fffef3;
        }
        
        .alert-message {
            font-size: 14px;
            color: #333;
            margin-bottom: 5px;
        }
        
        .alert-time {
            font-size: 12px;
            color: #999;
        }
        
        .refresh-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            margin-top: 20px;
        }
        
        .refresh-btn:hover {
            background: #5568d3;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: white;
            font-size: 18px;
        }
        
        .roi-highlight {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .roi-stat {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.3);
        }
        
        .roi-stat:last-child {
            border-bottom: none;
        }
        
        .roi-label {
            opacity: 0.9;
        }
        
        .roi-value {
            font-weight: bold;
            font-size: 16px;
        }
        
        .timestamp {
            text-align: right;
            font-size: 12px;
            color: #999;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Churn Prediction - Operational Monitoring</h1>
            <p>Real-time tracking of model performance, ROI, and customer risk segmentation</p>
            <span class="status-badge status-operational" id="status">● OPERATIONAL</span>
        </div>
        
        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Predictions (30d)</h3>
                <div class="metric-value" id="metric-totalPred">-</div>
                <div class="metric-change">Last update: <span id="update-time">-</span></div>
            </div>
            
            <div class="metric-card">
                <h3>Critical Risk Customers</h3>
                <div class="metric-value" id="metric-critical">-</div>
                <div style="margin-top: 15px;">
                    <div style="font-size: 12px; color: #666; margin-bottom: 10px;">Risk Distribution:</div>
                    <div class="risk-distribution">
                        <div class="risk-bar risk-critical">
                            <span id="risk-critical-pct">-</span>%
                        </div>
                        <div class="risk-bar risk-high">
                            <span id="risk-high-pct">-</span>%
                        </div>
                        <div class="risk-bar risk-medium">
                            <span id="risk-medium-pct">-</span>%
                        </div>
                        <div class="risk-bar risk-low">
                            <span id="risk-low-pct">-</span>%
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Pending Alerts</h3>
                <div class="metric-value" id="metric-alerts">-</div>
                <div class="metric-change negative">Action Required</div>
            </div>
            
            <div class="metric-card">
                <h3>Avg Risk Score</h3>
                <div class="metric-value" id="metric-avgScore">-</div>
                <div class="metric-change">Model Confidence: High</div>
            </div>
        </div>
        
        <!-- ROI Highlight -->
        <div class="roi-highlight">
            <div style="font-size: 20px; font-weight: bold; margin-bottom: 20px;">💰 Program ROI Analysis (Last 30 Days)</div>
            <div class="roi-stat">
                <span class="roi-label">Revenue at Risk</span>
                <span class="roi-value">$<span id="roi-revenueAtRisk">0</span></span>
            </div>
            <div class="roi-stat">
                <span class="roi-label">Program Cost</span>
                <span class="roi-value">$<span id="roi-cost">0</span></span>
            </div>
            <div class="roi-stat">
                <span class="roi-label">Revenue Saved</span>
                <span class="roi-value">$<span id="roi-saved">0</span></span>
            </div>
            <div class="roi-stat">
                <span class="roi-label">Net Benefit</span>
                <span class="roi-value">$<span id="roi-netBenefit">0</span></span>
            </div>
            <div class="roi-stat" style="border-bottom: none;">
                <span class="roi-label">ROI Multiple</span>
                <span class="roi-value"><span id="roi-multiple">0.0</span>x</span>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Risk Distribution</div>
                <canvas id="riskChart"></canvas>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Outcome Analysis</div>
                <canvas id="outcomeChart"></canvas>
            </div>
        </div>
        
        <!-- Alerts Panel -->
        <div class="alerts-panel">
            <div class="chart-title">Recent Alerts</div>
            <div id="alerts-container">
                <div class="alert-item">
                    <div class="alert-message">No critical alerts at this time</div>
                    <div class="alert-time">System is running smoothly</div>
                </div>
            </div>
        </div>
        
        <button class="refresh-btn" onclick="loadDashboardData()">🔄 Refresh Data</button>
        
        <div class="timestamp">
            Last updated: <span id="last-updated">-</span>
        </div>
    </div>
    
    <script>
        let riskChart = null;
        let outcomeChart = null;
        
        async function loadDashboardData() {
            try {
                const response = await axios.get('/api/dashboard');
                const data = response.data;
                
                // Update metric cards
                document.getElementById('metric-totalPred').textContent = 
                    data.predictions.total.toLocaleString();
                document.getElementById('metric-critical').textContent = 
                    data.predictions.critical.toLocaleString();
                document.getElementById('metric-alerts').textContent = 
                    data.alerts.critical.toLocaleString();
                document.getElementById('metric-avgScore').textContent = 
                    data.predictions.avg_score.toFixed(1);
                
                // Update risk distribution percentages
                const totalPred = data.predictions.total;
                document.getElementById('risk-critical-pct').textContent = 
                    ((data.predictions.critical / totalPred * 100) || 0).toFixed(0);
                document.getElementById('risk-high-pct').textContent = 
                    ((data.predictions.high / totalPred * 100) || 0).toFixed(0);
                document.getElementById('risk-medium-pct').textContent = 
                    ((data.predictions.medium / totalPred * 100) || 0).toFixed(0);
                document.getElementById('risk-low-pct').textContent = 
                    ((data.predictions.low / totalPred * 100) || 0).toFixed(0);
                
                // Update ROI section
                document.getElementById('roi-revenueAtRisk').textContent = 
                    (data.financials.revenue_at_risk / 1000000).toFixed(2) + 'M';
                document.getElementById('roi-cost').textContent = 
                    (data.financials.program_cost / 1000).toFixed(0) + 'K';
                document.getElementById('roi-saved').textContent = 
                    (data.financials.revenue_saved / 1000000).toFixed(2) + 'M';
                document.getElementById('roi-netBenefit').textContent = 
                    (data.financials.net_benefit / 1000000).toFixed(2) + 'M';
                document.getElementById('roi-multiple').textContent = 
                    data.financials.roi.toFixed(2);
                
                // Update timestamp
                const updateTime = new Date(data.timestamp).toLocaleTimeString();
                document.getElementById('update-time').textContent = updateTime;
                document.getElementById('last-updated').textContent = updateTime;
                
                // Update charts
                updateRiskChart(data.predictions);
                updateOutcomeChart(data.outcomes);
                
                // Update status
                const status = data.alerts.critical > 50 ? 'warning' : 'operational';
                const statusEl = document.getElementById('status');
                statusEl.textContent = status === 'warning' ? 
                    '⚠ WARNING' : '● OPERATIONAL';
                statusEl.className = `status-badge status-${status}`;
                
            } catch (error) {
                console.error('Error loading dashboard data:', error);
            }
        }
        
        function updateRiskChart(predictions) {
            const ctx = document.getElementById('riskChart').getContext('2d');
            
            if (riskChart) {
                riskChart.destroy();
            }
            
            riskChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Critical', 'High', 'Medium', 'Low'],
                    datasets: [{
                        data: [predictions.critical, predictions.high, 
                               predictions.medium, predictions.low],
                        backgroundColor: ['#dc2626', '#f97316', '#eab308', '#10b981']
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }
        
        function updateOutcomeChart(outcomes) {
            const ctx = document.getElementById('outcomeChart').getContext('2d');
            
            if (outcomeChart) {
                outcomeChart.destroy();
            }
            
            const saved = outcomes.saved || 0;
            const churned = outcomes.churned || 0;
            
            outcomeChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Retained', 'Churned'],
                    datasets: [{
                        label: 'Customers',
                        data: [saved, churned],
                        backgroundColor: ['#10b981', '#dc2626']
                    }]
                },
                options: {
                    responsive: true,
                    indexAxis: 'y',
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
        
        // Load data on page load and refresh every 30 seconds
        loadDashboardData();
        setInterval(loadDashboardData, 30000);
    </script>
</body>
</html>
"""


@app.route('/')
def dashboard():
    """Serve monitoring dashboard"""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/dashboard')
def api_dashboard():
    """API endpoint for dashboard metrics"""
    metrics = get_dashboard_metrics()
    return jsonify(metrics)


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


if __name__ == '__main__':
    print("=" * 80)
    print("MONITORING DASHBOARD WEB SERVER")
    print("=" * 80)
    print("\n✓ Dashboard available at: http://localhost:5000")
    print("✓ API endpoint: http://localhost:5000/api/dashboard")
    print("✓ Health check: http://localhost:5000/api/health")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
