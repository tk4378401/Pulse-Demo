"""
nadi_main.py - आयुर्वेदिक नाड़ी परीक्षण के लिए डेस्कटॉप GUI एप्लिकेशन
Desktop GUI Application for Ayurvedic Nadi Pariksha

यह मॉड्यूल तीनों मॉड्यूल को एक साथ लाता है और 
Zero-bug फिक्स (np.nan initialization) के साथ स्मूथ लाइव प्लॉटिंग करता है।
"""

import sys                          # सिस्टम पैरामीटर्स और एग्जिट कोड के लिए
import numpy as np                  # न्यूमेरिकल कंप्यूटेशन के लिए
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import QTimer     # टाइमर के लिए
from PyQt6.QtGui import QFont       # फ़ॉन्ट स्टाइलिंग के लिए
import pyqtgraph as pg              # रियल-टाइम ग्राफ़ प्लॉटिंग के लिए

# कस्टम मॉड्यूल इंपोर्ट करें
from nadi_generator import VirtualSensor
from nadi_dsp import NadiDSP


class NadiMainWindow(QMainWindow):
    """
    नाड़ी मेन विंडो - मुख्य GUI एप्लिकेशन
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Ayurvedic Nadi Pariksha DSP Sandbox - आयुर्वेदिक नाड़ी परीक्षा")
        self.setGeometry(100, 100, 1400, 900)
        
        # VirtualSensor और NadiDSP इनिशियलाइज करें
        self.sensor = VirtualSensor(sampling_rate=1000, batch_size=50)
        self.dsp = NadiDSP(sampling_rate=1000, highpass_cutoff=0.1)
        
        # Data Buffers (2000 samples = 2 seconds)
        self.max_samples = 2000
        
        # === ZERO-BUG FIX ===
        # Initialize buffers with NaN (Not a Number) to fix auto-range squash bug
        # इससे शुरूआती ग्राफ़ 0 से ज़ूम-आउट नहीं होगा (No squash)
        self.raw_buffer = np.full(self.max_samples, np.nan)
        self.velocity_buffer = np.full(self.max_samples, np.nan)
        self.displacement_buffer = np.full(self.max_samples, np.nan)
        
        self.setup_ui()
        
        # Timer सेटअप (हर 50ms में डेटा पोल करने के लिए)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.setInterval(50)
        
        self.is_running = False
    
    def setup_ui(self):
        """
        यूजर इंटरफेस सेटअप - सभी UI एलिमेंट्स बनाएं
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # --- Control Panel ---
        control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("▶ Start Simulation")
        self.start_button.setFixedSize(200, 50)
        self.start_button.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.start_button.clicked.connect(self.start_simulation)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹ Stop Simulation")
        self.stop_button.setFixedSize(200, 50)
        self.stop_button.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.stop_button.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_simulation)
        control_layout.addWidget(self.stop_button)
        
        self.status_label = QLabel("Status: Ready to Start")
        self.status_label.setFont(QFont("Arial", 12))
        control_layout.addWidget(self.status_label)
        
        main_layout.addLayout(control_layout)
        
        # --- Graph 1: Raw Sensor Wave ---
        raw_label = QLabel("Raw Sensor Wave (Filtered) - पीला ग्राफ़")
        raw_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        main_layout.addWidget(raw_label)
        
        self.raw_plot = pg.PlotWidget()
        self.raw_plot.setBackground('w')
        self.raw_plot.setTitle("Raw Filtered Signal", color='k', size='12pt')
        self.raw_plot.setLabel('bottom', 'Time', units='ms')
        self.raw_plot.setLabel('left', 'Amplitude', units='AU')
        self.raw_plot.showGrid(x=True, y=True, alpha=0.3)
        main_layout.addWidget(self.raw_plot)
        self.raw_curve = self.raw_plot.plot(pen=pg.mkPen('y', width=2))
        
        # --- Graph 2: Velocity Wave ---
        velocity_label = QLabel("Velocity Wave (First Integration) - नीला ग्राफ़")
        velocity_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        main_layout.addWidget(velocity_label)
        
        self.velocity_plot = pg.PlotWidget()
        self.velocity_plot.setBackground('w')
        self.velocity_plot.setTitle("Velocity Signal", color='k', size='12pt')
        self.velocity_plot.setLabel('bottom', 'Time', units='ms')
        self.velocity_plot.setLabel('left', 'Velocity', units='AU/ms')
        self.velocity_plot.showGrid(x=True, y=True, alpha=0.3)
        main_layout.addWidget(self.velocity_plot)
        self.velocity_curve = self.velocity_plot.plot(pen=pg.mkPen('b', width=2))
        
        # --- Graph 3: Displacement Wave ---
        displacement_label = QLabel("Displacement Wave / Vata-Pitta-Kapha Morphology - हरा ग्राफ़")
        displacement_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        main_layout.addWidget(displacement_label)
        
        self.displacement_plot = pg.PlotWidget()
        self.displacement_plot.setBackground('w')
        self.displacement_plot.setTitle("Displacement Signal (Tridosha Morphology)", color='k', size='12pt')
        self.displacement_plot.setLabel('bottom', 'Time', units='ms')
        self.displacement_plot.setLabel('left', 'Displacement', units='AU·ms')
        self.displacement_plot.showGrid(x=True, y=True, alpha=0.3)
        main_layout.addWidget(self.displacement_plot)
        self.displacement_curve = self.displacement_plot.plot(pen=pg.mkPen('g', width=2))
    
    def update_data(self):
        """
        QTimer callback - हर 50ms में डेटा पोल और अपडेट करें
        """
        try:
            batch = self.sensor.data_queue.get_nowait()
        except Exception:
            return
        
        results = self.dsp.process_batch(batch)
        
        # डेटा बफर्स में नया डेटा जोड़ें (Append new data)
        self.raw_buffer = np.concatenate([self.raw_buffer, results['raw_filtered']])
        self.velocity_buffer = np.concatenate([self.velocity_buffer, results['velocity']])
        self.displacement_buffer = np.concatenate([self.displacement_buffer, results['displacement']])
        
        # बफर साइज को 2000 सैंपल्स तक सीमित रखें
        if len(self.raw_buffer) > self.max_samples:
            self.raw_buffer = self.raw_buffer[-self.max_samples:]
            self.velocity_buffer = self.velocity_buffer[-self.max_samples:]
            self.displacement_buffer = self.displacement_buffer[-self.max_samples:]
        
        # Graph Curves अपडेट करें
        # connect="finite" का मतलब है कि NaN values प्लॉट नहीं होंगी (Zero-bug fix!)
        self.raw_curve.setData(self.raw_buffer, connect="finite")
        self.velocity_curve.setData(self.velocity_buffer, connect="finite")
        self.displacement_curve.setData(self.displacement_buffer, connect="finite")
    
    def start_simulation(self):
        if self.is_running:
            return
            
        self.sensor.start()
        self.timer.start()
        self.is_running = True
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Status: ● Running")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        
        self.dsp.reset_state()
        
        # === ZERO-BUG FIX (Clear states with NaN) ===
        self.raw_buffer = np.full(self.max_samples, np.nan)
        self.velocity_buffer = np.full(self.max_samples, np.nan)
        self.displacement_buffer = np.full(self.max_samples, np.nan)
        
        print("Simulation started - सिमुलेशन शुरू हो गया")
    
    def stop_simulation(self):
        if not self.is_running:
            return
            
        self.timer.stop()
        self.sensor.stop()
        self.is_running = False
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: ■ Stopped")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        print("Simulation stopped - सिमुलेशन रुक गया")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NadiMainWindow()
    window.show()
    sys.exit(app.exec())
