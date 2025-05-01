// Initialize the charts
const attitudeCtx = document.getElementById('attitudeChart').getContext('2d');
const imuCtx = document.getElementById('imuChart').getContext('2d');

// Attitude chart (Roll, Pitch, Yaw)
const attitudeChart = new Chart(attitudeCtx, {
    type: 'line',
    data: {
        labels: Array(30).fill(''),
        datasets: [
            {
                label: 'Roll',
                data: Array(30).fill(null),
                borderColor: '#3498db',
                tension: 0.4
            },
            {
                label: 'Pitch',
                data: Array(30).fill(null),
                borderColor: '#2ecc71',
                tension: 0.4
            },
            {
                label: 'Yaw',
                data: Array(30).fill(null),
                borderColor: '#e74c3c',
                tension: 0.4
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#aaa'
                }
            },
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#aaa'
                }
            }
        },
        plugins: {
            legend: {
                labels: {
                    color: '#aaa'
                }
            }
        }
    }
});

// IMU chart (Accelerometer X, Y, Z)
const imuChart = new Chart(imuCtx, {
    type: 'line',
    data: {
        labels: Array(30).fill(''),
        datasets: [
            {
                label: 'Accel X',
                data: Array(30).fill(null),
                borderColor: '#e74c3c',
                tension: 0.4
            },
            {
                label: 'Accel Y',
                data: Array(30).fill(null),
                borderColor: '#2ecc71',
                tension: 0.4
            },
            {
                label: 'Accel Z',
                data: Array(30).fill(null),
                borderColor: '#3498db',
                tension: 0.4
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#aaa'
                }
            },
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: '#aaa'
                }
            }
        },
        plugins: {
            legend: {
                labels: {
                    color: '#aaa'
                }
            }
        }
    }
});

// Connect to socket.io server
const socket = io();

socket.on('connect', function() {
    console.log('Connected to server');
});

socket.on('fpv_data', function(data) {
    // Update numerical displays
    document.getElementById('roll').textContent = `${data.orientation.roll.toFixed(1)}°`;
    document.getElementById('pitch').textContent = `${data.orientation.pitch.toFixed(1)}°`;
    document.getElementById('yaw').textContent = `${data.orientation.yaw.toFixed(1)}°`;
    document.getElementById('altitude').textContent = `${data.altitude.toFixed(1)} m`;
    
    // Update attitude chart
    updateChart(attitudeChart, data.timestamp, [
        data.orientation.roll,
        data.orientation.pitch,
        data.orientation.yaw
    ]);
    
    // Update IMU chart
    updateChart(imuChart, data.timestamp, [
        data.accel.x,
        data.accel.y,
        data.accel.z
    ]);
});

// Helper function to update charts
function updateChart(chart, label, values) {
    // Remove oldest data point
    chart.data.labels.shift();
    // Add new timestamp
    chart.data.labels.push(label);
    
    // Update each dataset
    for (let i = 0; i < values.length; i++) {
        chart.data.datasets[i].data.shift();
        chart.data.datasets[i].data.push(values[i]);
    }
    
    // Update the chart
    chart.update();
}