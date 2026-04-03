"""
nadi_main.py - आयुर्वेदिक नाड़ी परीक्षण के लिए डेस्कटॉप GUI एप्लिकेशन
Desktop GUI Application for Ayurvedic Nadi Pariksha

इसमें वात, पित्त, कफ इनपुट वेव्स को स्विच करने के लिए कंट्रोल पैनल जोड़ा गया है।
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
        self.setWindowTitle("Ayurvedic Nadi Pariksha - Tridosha Analyzer")
        self.setGeometry(100, 100, 1400, 900)
        
        self.sensor = VirtualSensor(sampling_rate=1000, batch_size=50)
        self.dsp = NadiDSP(sampling_rate=1000)
        
        self.max_samples = 2000
        self.raw_buffer = np.array([])
        self.velocity_buffer = np.array([])
        self.displacement_buffer = np.array([])
        
        self.setup_ui()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.setInterval(50)
        self.is_running = False
        self.current_dosha_name = "संतुलित (Balanced)"
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # ====================
        # Control Panel (Start/Stop)
        # ====================
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
        
        self.status_label = QLabel("Status: Ready to Start | Active: संतुलित (Balanced)")
        self.status_label.setFont(QFont("Arial", 12))
        control_layout.addWidget(self.status_label)
        
        main_layout.addLayout(control_layout)
        
        # ====================
        # Dosha Selection Panel (वात, पित्त, कफ Buttons)
        # ====================
        dosha_layout = QHBoxLayout()
        
        self.btn_vata = QPushButton("💨 वात (Vata) - सर्प गति")
        self.btn_vata.setFixedSize(250, 40)
        self.btn_vata.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.btn_vata.setStyleSheet("background-color: #5D4037; color: white; border-radius: 5px;")
        self.btn_vata.clicked.connect(lambda: self.change_dosha('vata', 'वात (Vata) - तेज़'))
        dosha_layout.addWidget(self.btn_vata)
        
        self.btn_pitta = QPushButton("🔥 पित्त (Pitta) - मंडूक गति")
        self.btn_pitta.setFixedSize(250, 40)
        self.btn_pitta.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.btn_pitta.setStyleSheet("background-color: #E64A19; color: white; border-radius: 5px;")
        self.btn_pitta.clicked.connect(lambda: self.change_dosha('pitta', 'पित्त (Pitta) - मध्यम'))
        dosha_layout.addWidget(self.btn_pitta)
        
        self.btn_kapha = QPushButton("💧 कफ (Kapha) - हंस गति")
        self.btn_kapha.setFixedSize(250, 40)
        self.btn_kapha.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.btn_kapha.setStyleSheet("background-color: #0288D1; color: white; border-radius: 5px;")
        self.btn_kapha.clicked.connect(lambda: self.change_dosha('kapha', 'कफ (Kapha) - धीमी'))
        dosha_layout.addWidget(self.btn_kapha)
        
        # लेफ्ट अलाइनमेंट के लिए स्ट्रेच जोड़ें
        dosha_layout.addStretch()
        main_layout.addLayout(dosha_layout)
        
        # ====================
        # Graph Widgets
        # ====================
        # 1. Raw Wave
        main_layout.addWidget(QLabel("Raw Arterial Pulse (Sensor Data) - पीला"))
        self.raw_plot = pg.PlotWidget()
        self.raw_plot.setBackground('#1e1e1e')
        self.raw_plot.showGrid(x=True, y=True, alpha=0.3)
        main_layout.addWidget(self.raw_plot)
        self.raw_curve = self.raw_plot.plot(pen=pg.mkPen('#FFD700', width=2))
        
        # 2. Velocity Wave
        main_layout.addWidget(QLabel("Velocity Wave (Blood Flow Rate) - स्यान"))
        self.velocity_plot = pg.PlotWidget()
        self.velocity_plot.setBackground('#1e1e1e')
        self.velocity_plot.showGrid(x=True, y=True, alpha=0.3)
        main_layout.addWidget(self.velocity_plot)
        self.velocity_curve = self.velocity_plot.plot(pen=pg.mkPen('#00FFFF', width=2))
        
        # 3. Displacement Wave
        main_layout.addWidget(QLabel("Displacement / Vata-Pitta-Kapha Morphology - हरा"))
        self.displacement_plot = pg.PlotWidget()
        self.displacement_plot.setBackground('#1e1e1e')
        self.displacement_plot.showGrid(x=True, y=True, alpha=0.3)
        main_layout.addWidget(self.displacement_plot)
        self.displacement_curve = self.displacement_plot.plot(pen=pg.mkPen('#00FF00', width=2))
    
    def change_dosha(self, dosha_type, display_name):
        """दोष प्रोफाइल बदलें और UI अपडेट करें"""
        self.sensor.set_dosha_profile(dosha_type)
        self.current_dosha_name = display_name
        
        if self.is_running:
            self.status_label.setText(f"Status: ● Running | Active: {self.current_dosha_name}")
        else:
            self.status_label.setText(f"Status: ■ Stopped | Active: {self.current_dosha_name}")
            
        # स्विच करते समय ग्राफ़ को साफ़ करें ताकि नई वेव साफ़ दिखे
        self.raw_buffer = np.array([])
        self.velocity_buffer = np.array([])
        self.displacement_buffer = np.array([])
    
    def update_data(self):
        # Queue Draining Loop (रुका हुआ डेटा एक साथ निकालने के लिए)
        data_found = False
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
            
        self.raw_curve.setData(self.raw_buffer)
        self.velocity_curve.setData(self.velocity_buffer)
        self.displacement_curve.setData(self.displacement_buffer)
    
    def start_simulation(self):
        if self.is_running: return
        self.sensor.start()
        self.timer.start()
        self.is_running = True
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(f"Status: ● Running | Active: {self.current_dosha_name}")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        
        self.dsp.reset_state()
        self.raw_buffer = np.array([])
        self.velocity_buffer = np.array([])
        self.displacement_buffer = np.array([])
    
    def stop_simulation(self):
        if not self.is_running: return
        self.timer.stop()
        self.sensor.stop()
        self.is_running = False
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText(f"Status: ■ Stopped | Active: {self.current_dosha_name}")
        self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NadiMainWindow()
    window.show()
    sys.exit(app.exec())
