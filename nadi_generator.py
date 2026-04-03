"""
nadi_generator.py - 100% Medical Grade Pulse Generator
(Continuous Phase Tracking के साथ - कोई गैप या टाइम-वार्प नहीं)
"""

import numpy as np
import threading
import queue
import time

class VirtualSensor(threading.Thread):
    def __init__(self, sampling_rate=1000, batch_size=50, vata_strength=0.8, pitta_strength=0.5, kapha_strength=0.4):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.vata_strength = vata_strength
        self.pitta_strength = pitta_strength
        self.kapha_strength = kapha_strength
        
        self._stop_event = threading.Event()
        self.data_queue = queue.Queue()
        self.current_time = 0.0
        
        # PERMANENT FIX: Continuous Phase Tracking 
        # (यह पल्स को 1.7s दूर भागने से रोकेगा)
        self.phase = 0.0  
        
    def generate_batch(self):
        dt = 1.0 / self.sampling_rate
        batch_data = np.zeros(self.batch_size)
        
        for i in range(self.batch_size):
            # हार्ट रेट को 70 BPM पर सेट करें (साँस के साथ हल्का बदलाव)
            current_hr = 70.0 + 3.0 * np.sin(2 * np.pi * 0.25 * self.current_time)
            freq = current_hr / 60.0
            
            # Phase को लगातार आगे बढ़ाएं (कोई टाइम-वार्प जंप नहीं)
            self.phase += freq * dt
            self.phase = self.phase % 1.0
            
            # 1. वात (Vata) - Systolic Peak
            vata = self.vata_strength * 1.0 * np.exp(-((self.phase - 0.15) ** 2) / (2 * 0.04 ** 2))
            
            # 2. पित्त (Pitta) - Dicrotic Notch
            pitta = self.pitta_strength * 0.45 * np.exp(-((self.phase - 0.38) ** 2) / (2 * 0.06 ** 2))
            
            # 3. कफ (Kapha) - Diastolic Base
            kapha = self.kapha_strength * 0.25 * np.exp(-((self.phase - 0.65) ** 2) / (2 * 0.12 ** 2))
            
            base_wave = vata + pitta + kapha
            
            # साँस का बेसलाइन प्रभाव
            respiration = 40.0 * np.sin(2 * np.pi * 0.25 * self.current_time)
            
            # 2048 DC Offset + 600 Amplitude
            batch_data[i] = 2048.0 + (base_wave * 600.0) + respiration
            
            self.current_time += dt
            
        # हल्का सा क्लिनिकल नॉइज़
        batch_data += 2.0 * np.random.randn(self.batch_size)
        return batch_data

    def run(self):
        batch_duration = self.batch_size / self.sampling_rate
        while not self._stop_event.is_set():
            batch_data = self.generate_batch()
            try:
                self.data_queue.put(batch_data, block=False)
            except queue.Full:
                pass
            time.sleep(batch_duration)
            
    def start(self):
        super().start()
        print("Bio-Realistic Virtual Sensor Started")
        
    def stop(self):
        self._stop_event.set()
        self.join(timeout=1.0)
        
    def get_latest_batch(self, timeout=1.0):
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
