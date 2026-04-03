"""
nadi_main.py - 100% Real-time Ayurvedic Nadi Pariksha Monitor
(No Stuttering, No Lag, Perfect Queue Draining)
"""

import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
import pyqtgraph as pg

from nadi_generator import VirtualSensor
from nadi_dsp import NadiDSP

class NadiMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ayurvedic Nadi Pariksha - True Realtime Monitor")
        self.setGeometry(100, 100, 1400, 900)
        
        self.sensor = VirtualSensor(sampling_rate=1000, batch_size=50, 
                                    vata_strength=0.8, pitta_strength=0.5, kapha_strength=0.4)
        self.dsp = NadiDSP(sampling_rate=1000)
        
        self.max_samples = 3000
        self.raw_buffer = np.full(self.max_samples, np.nan)
        self.velocity_buffer = np.full(self.max_samples, np.nan)
        self.displacement_buffer = np.full(self.max_samples, np.nan)
        
        self.setup_ui()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        # पोलिंग को थोड़ा तेज़ किया गया है (40ms) ताकि कतार (Queue) कभी जाम न हो
        self.timer.setInterval(40) 
        self.is_running = False
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("▶ Start Monitoring")
        self.start_button.setFixedSize(200, 50)
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px;")
        self.start_button.clicked.connect(self.start_simulation)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setFixedSize(200, 50)
        self.stop_button.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; font-size: 14px;")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_simulation)
        control_layout.addWidget(self.stop_button)
        
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Arial", 12))
        control_layout.addWidget(self.status_label)
        main_layout.addLayout(control_layout)
        
        def create_plot(title, y_label, pen_color):
            label = QLabel(title)
            label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            main_layout.addWidget(label)
            plot = pg.PlotWidget()
            plot.setBackground('#121212')
            plot.setLabel('bottom', 'Time', units='samples')
            plot.setLabel('left', y_label)
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.enableAutoRange('y', True)
            main_layout.addWidget(plot)
            curve = plot.plot(pen=pg.mkPen(pen_color, width=2.5))
            return curve

        self.raw_curve = create_plot("Raw Arterial Pulse (Sensor Data) - पीला", "Amplitude", '#FFD700')
        self.velocity_curve = create_plot("Velocity Wave (Blood Flow Rate) - स्यान", "Velocity", '#00FFFF')
        self.displacement_curve = create_plot("Displacement / Vata-Pitta-Kapha - हरा", "Displacement", '#00FF00')

    def update_data(self):
        data_found = False
        
        # PERMANENT FIX: QUEUE DRAINING LOOP
        # यह लूप फँसा हुआ सारा डेटा एक साथ निकाल लेगा। ग्राफ कभी स्लो-मोशन में नहीं चलेगा!
        while not self.sensor.data_queue.empty():
            try:
                batch = self.sensor.data_queue.get_nowait()
                results = self.dsp.process_batch(batch)
                
                self.raw_buffer = np.concatenate([self.raw_buffer, results['raw_filtered']])
                self.velocity_buffer = np.concatenate([self.velocity_buffer, results['velocity']])
                self.displacement_buffer = np.concatenate([self.displacement_buffer, results['displacement']])
                data_found = True
            except Exception:
                break
                
        if not data_found:
            return
            
        if len(self.raw_buffer) > self.max_samples:
            self.raw_buffer = self.raw_buffer[-self.max_samples:]
            self.velocity_buffer = self.velocity_buffer[-self.max_samples:]
            self.displacement_buffer = self.displacement_buffer[-self.max_samples:]
            
        self.raw_curve.setData(self.raw_buffer, connect="finite")
        self.velocity_curve.setData(self.velocity_buffer, connect="finite")
        self.displacement_curve.setData(self.displacement_buffer, connect="finite")
        
    def start_simulation(self):
        if self.is_running: return
        self.sensor.start()
        self.timer.start()
        self.is_running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Status: ● Live Monitoring")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        
        self.dsp.reset_state()
        self.raw_buffer = np.full(self.max_samples, np.nan)
        self.velocity_buffer = np.full(self.max_samples, np.nan)
        self.displacement_buffer = np.full(self.max_samples, np.nan)
        
    def stop_simulation(self):
        if not self.is_running: return
        self.timer.stop()
        self.sensor.stop()
        self.is_running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: ■ Stopped")
        self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NadiMainWindow()
    window.show()
    sys.exit(app.exec())
