"""
nadi_generator.py - The Flawless Multi-Gaussian Pulse Generator
(No Auto-Gain Bugs, Perfect Phase Tracking)
"""

import numpy as np
import threading
import queue
import time

class VirtualSensor(threading.Thread):
    def __init__(self, sampling_rate=1000, batch_size=50):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        
        self._stop_event = threading.Event()
        self.data_queue = queue.Queue()
        
        # Continuous Phase Tracking (यह ग्राफ़ को कभी टूटने या ज़िगज़ैग नहीं होने देगा)
        self.phase = 0.0  
        self.current_time = 0.0
        
        self.set_dosha_profile('balanced')
        
    def set_dosha_profile(self, dosha_type):
        """बिना ग्राफ़ तोड़े, चलते हुए सिमुलेशन में दोष बदलें"""
        if dosha_type == 'vata':
            self.v_str, self.p_str, self.k_str = 1.0, 0.2, 0.2
            self.heart_rate = 85.0
        elif dosha_type == 'pitta':
            self.v_str, self.p_str, self.k_str = 0.4, 0.8, 0.3
            self.heart_rate = 75.0
        elif dosha_type == 'kapha':
            self.v_str, self.p_str, self.k_str = 0.2, 0.3, 0.8
            self.heart_rate = 60.0
        else: # balanced
            self.v_str, self.p_str, self.k_str = 0.6, 0.5, 0.5
            self.heart_rate = 70.0
            
        self.heart_frequency = self.heart_rate / 60.0
        print(f"Dosha Profile: {dosha_type.upper()} | Heart Rate: {self.heart_rate} BPM")

    def generate_batch(self):
        batch_data = np.zeros(self.batch_size)
        dt = 1.0 / self.sampling_rate
        
        for i in range(self.batch_size):
            # 1. समय (Phase) को लगातार आगे बढ़ाएं - कोई जंप या रिसेट नहीं!
            self.phase += self.heart_frequency * dt
            if self.phase >= 1.0:
                self.phase -= 1.0
                
            # 2. Multi-Gaussian मॉडल (यह अपने आप 0 से 1 के बीच रहता है, Normalization की ज़रूरत नहीं)
            vata = self.v_str * 1.0 * np.exp(-((self.phase - 0.15) ** 2) / (2 * 0.04 ** 2))
            pitta = self.p_str * 0.5 * np.exp(-((self.phase - 0.38) ** 2) / (2 * 0.06 ** 2))
            kapha = self.k_str * 0.3 * np.exp(-((self.phase - 0.65) ** 2) / (2 * 0.12 ** 2))
            
            base_wave = vata + pitta + kapha
            
            # 3. साँस का प्रभाव और 2048 DC Offset
            respiration = 20.0 * np.sin(2 * np.pi * 0.25 * self.current_time)
            batch_data[i] = 2048.0 + (base_wave * 600.0) + respiration
            
            self.current_time += dt
            
        # 4. बहुत ही हल्का क्लिनिकल नॉइज़
        batch_data += 2.0 * np.random.randn(self.batch_size)
        
        return batch_data
    
    def run(self):
        batch_duration = self.batch_size / self.sampling_rate
        while not self._stop_event.is_set():
            try:
                self.data_queue.put(self.generate_batch(), block=False)
            except queue.Full:
                pass
            time.sleep(batch_duration)
            
    def start(self):
        super().start()
        print(f"Virtual Sensor started (1000Hz, {self.batch_size} samples/batch)")
    
    def stop(self):
        self._stop_event.set()
        self.join(timeout=1.0)
        
    def get_latest_batch(self, timeout=1.0):
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_queue_size(self):
        return self.data_queue.qsize()

if __name__ == "__main__":
    sensor = VirtualSensor()
    sensor.start()
    time.sleep(1)
    sensor.stop()
