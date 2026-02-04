/**
 * Fraud Detection Dashboard - JavaScript
 * Real-time data visualization and WebSocket handling
 */

class FraudDashboard {
    constructor() {
        this.ws = null;
        this.charts = {};
        this.transactions = [];
        this.maxTransactions = 50;
        this.stats = {
            total: 0,
            approved: 0,
            denied: 0,
            review: 0,
        };
        
        this.init();
    }
    
    async init() {
        await this.loadInitialData();
        this.initCharts();
        this.connectWebSocket();
        this.startPolling();
    }
    
    async loadInitialData() {
        try {
            const [statsRes, modelRes, geoRes] = await Promise.all([
                fetch('/api/stats/realtime'),
                fetch('/api/stats/model-performance'),
                fetch('/api/stats/geographic'),
            ]);
            
            const stats = await statsRes.json();
            const models = await modelRes.json();
            const geo = await geoRes.json();
            
            this.updateStats(stats);
            this.updateModelCards(models);
            this.updateRiskFactors(stats.top_risk_factors);
        } catch (error) {
            console.error('Failed to load initial data:', error);
        }
    }
    
    initCharts() {
        this.initTransactionChart();
        this.initRiskDistributionChart();
        this.initHeatmap();
    }
    
    initTransactionChart() {
        const ctx = document.getElementById('transactionChart');
        if (!ctx) return;
        
        const labels = Array.from({length: 60}, (_, i) => `${59-i}s ago`).reverse();
        const data = Array.from({length: 60}, () => Math.floor(Math.random() * 100) + 50);
        
        this.charts.transactions = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Transactions/sec',
                    data: data,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                },
                scales: {
                    x: {
                        display: false,
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)',
                        },
                        ticks: {
                            color: '#6b6b7b',
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index',
                },
            }
        });
    }
    
    initRiskDistributionChart() {
        const ctx = document.getElementById('riskDistributionChart');
        if (!ctx) return;
        
        this.charts.riskDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Low', 'Medium', 'High', 'Critical'],
                datasets: [{
                    data: [75, 15, 8, 2],
                    backgroundColor: [
                        '#10b981',
                        '#f59e0b',
                        '#ef4444',
                        '#dc2626',
                    ],
                    borderWidth: 0,
                    spacing: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            color: '#a0a0b0',
                            padding: 15,
                            usePointStyle: true,
                        }
                    }
                }
            }
        });
    }
    
    initHeatmap() {
        const container = document.getElementById('heatmapContainer');
        if (!container) return;
        
        // Create 24x7 heatmap (24 hours x 7 days)
        const hours = 24;
        const days = 7;
        
        for (let d = 0; d < days; d++) {
            for (let h = 0; h < hours; h++) {
                const cell = document.createElement('div');
                cell.className = 'heatmap-cell';
                const intensity = Math.random();
                cell.style.backgroundColor = this.getHeatmapColor(intensity);
                cell.title = `Day ${d + 1}, Hour ${h}: ${Math.floor(intensity * 100)}% activity`;
                container.appendChild(cell);
            }
        }
    }
    
    getHeatmapColor(intensity) {
        if (intensity < 0.25) return 'rgba(16, 185, 129, 0.3)';
        if (intensity < 0.5) return 'rgba(16, 185, 129, 0.6)';
        if (intensity < 0.75) return 'rgba(245, 158, 11, 0.6)';
        return 'rgba(239, 68, 68, 0.6)';
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/transactions`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
        };
        
        this.ws.onmessage = (event) => {
            const transaction = JSON.parse(event.data);
            this.handleNewTransaction(transaction);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
            // Attempt reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    handleNewTransaction(transaction) {
        // Update stats
        this.stats.total++;
        if (transaction.decision === 'approve') this.stats.approved++;
        else if (transaction.decision === 'deny') this.stats.denied++;
        else this.stats.review++;
        
        // Add to transactions list
        this.transactions.unshift(transaction);
        if (this.transactions.length > this.maxTransactions) {
            this.transactions.pop();
        }
        
        // Update UI
        this.updateTransactionTable(transaction);
        this.updateChartData(transaction);
        this.updateStatCounters();
    }
    
    updateTransactionTable(transaction) {
        const tbody = document.getElementById('transactionTableBody');
        if (!tbody) return;
        
        const row = document.createElement('tr');
        row.className = 'animate-fade-in';
        
        const riskClass = transaction.risk_level;
        const decisionClass = transaction.decision;
        
        row.innerHTML = `
            <td><span class="transaction-id">${transaction.id}</span></td>
            <td>$${transaction.amount.toLocaleString()}</td>
            <td>${transaction.location}</td>
            <td>
                <div class="risk-meter">
                    <div class="risk-bar">
                        <div class="risk-bar-fill ${riskClass}" style="width: ${transaction.risk_score}%"></div>
                    </div>
                    <span class="risk-value">${transaction.risk_score.toFixed(1)}</span>
                </div>
            </td>
            <td><span class="risk-badge ${riskClass}">${transaction.risk_level}</span></td>
            <td><span class="decision-badge ${decisionClass}">${transaction.decision}</span></td>
            <td>${transaction.device_known ? '✓' : '✗'}</td>
        `;
        
        tbody.insertBefore(row, tbody.firstChild);
        
        // Remove old rows
        while (tbody.children.length > this.maxTransactions) {
            tbody.removeChild(tbody.lastChild);
        }
    }
    
    updateChartData(transaction) {
        if (this.charts.transactions) {
            const chart = this.charts.transactions;
            chart.data.datasets[0].data.shift();
            chart.data.datasets[0].data.push(Math.floor(Math.random() * 50) + 80);
            chart.update('none');
        }
    }
    
    updateStatCounters() {
        const elements = {
            totalTransactions: this.stats.total,
            approvedCount: this.stats.approved,
            deniedCount: this.stats.denied,
            reviewCount: this.stats.review,
        };
        
        for (const [id, value] of Object.entries(elements)) {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = value.toLocaleString();
            }
        }
    }
    
    updateStats(stats) {
        const tpsEl = document.getElementById('tps');
        if (tpsEl) tpsEl.textContent = stats.transactions_per_second;
        
        const latencyEl = document.getElementById('avgLatency');
        if (latencyEl) latencyEl.textContent = `${stats.avg_latency_ms}ms`;
        
        const fraudEl = document.getElementById('fraudDetected');
        if (fraudEl) fraudEl.textContent = stats.fraud_detected;
        
        // Update risk distribution chart
        if (this.charts.riskDistribution && stats.risk_distribution) {
            const dist = stats.risk_distribution;
            this.charts.riskDistribution.data.datasets[0].data = [
                dist.low, dist.medium, dist.high, dist.critical
            ];
            this.charts.riskDistribution.update();
        }
    }
    
    updateModelCards(models) {
        const container = document.getElementById('modelCards');
        if (!container || !models.models) return;
        
        container.innerHTML = models.models.map(model => `
            <div class="model-card">
                <div class="model-name">${model.name}</div>
                <div class="model-metrics">
                    <div class="model-metric">
                        <div class="model-metric-value">${(model.accuracy * 100).toFixed(2)}%</div>
                        <div class="model-metric-label">Accuracy</div>
                    </div>
                    <div class="model-metric">
                        <div class="model-metric-value">${(model.precision * 100).toFixed(2)}%</div>
                        <div class="model-metric-label">Precision</div>
                    </div>
                    <div class="model-metric">
                        <div class="model-metric-value">${(model.recall * 100).toFixed(2)}%</div>
                        <div class="model-metric-label">Recall</div>
                    </div>
                    <div class="model-metric">
                        <div class="model-metric-value">${model.avg_latency_ms}ms</div>
                        <div class="model-metric-label">Latency</div>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    updateRiskFactors(factors) {
        const container = document.getElementById('riskFactors');
        if (!container || !factors) return;
        
        container.innerHTML = factors.map(factor => `
            <div class="risk-factor-item">
                <span class="risk-factor-name">${factor.factor}</span>
                <span class="risk-factor-count">${factor.count}</span>
            </div>
        `).join('');
    }
    
    updateConnectionStatus(connected) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        if (statusDot) {
            statusDot.style.background = connected ? 'var(--success)' : 'var(--danger)';
        }
        if (statusText) {
            statusText.textContent = connected ? 'Live' : 'Disconnected';
        }
    }
    
    startPolling() {
        // Poll for stats every 5 seconds
        setInterval(async () => {
            try {
                const res = await fetch('/api/stats/realtime');
                const stats = await res.json();
                this.updateStats(stats);
            } catch (error) {
                console.error('Polling error:', error);
            }
        }, 5000);
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new FraudDashboard();
});
