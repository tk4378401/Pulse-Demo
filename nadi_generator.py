"""
nadi_generator.py - आयुर्वेदिक नाड़ी परीक्षण के लिए सिंथेटिक पल्स वेव जनरेटर
Synthetic Pulse Wave Generator for Ayurvedic Nadi Pariksha

यह मॉड्यूल 1000Hz सैम्पलिंग रेट पर आर्टिरियल पल्स वेव बनाता है 
और क्लिनिकल एक्यूरेसी के लिए 2048 DC Offset + 0.005 Noise को सिमुलेट करता है।
"""

import numpy as np              # न्यूमेरिकल कंप्यूटेशन के लिए
import scipy.signal as signal   # वेव शेपिंग के लिए
import threading                # बैकग्राउंड थ्रेड में डेटा जनरेट करने के लिए
import queue                    # थ्रेड-सेफ कतार
import time                     # टाइमिंग कंट्रोल के लिए


class VirtualSensor(threading.Thread):
    """
    वर्चुअल सेंसर क्लास - एक थ्रेड जो लगातार आयुर्वेदिक पल्स वेव डेटा जनरेट करता है
    """
    
    def __init__(self, sampling_rate=1000, batch_size=50, vata_strength=0.4, pitta_strength=0.3, kapha_strength=0.3):
        super().__init__()
        
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        
        self.vata_strength = vata_strength
        self.pitta_strength = pitta_strength
        self.kapha_strength = kapha_strength
        
        self._stop_event = threading.Event()
        self.data_queue = queue.Queue()
        self.current_time = 0.0
        
        self.heart_rate = 65  # 65 BPM
        self.heart_frequency = self.heart_rate / 60.0
        self.cardiac_cycle_duration = 1.0 / self.heart_frequency
        self.previous_value = 2048.0  # ADC offset से शुरू
    
    def generate_ayurvedic_pulse_wave(self, time_array):
        """
        आयुर्वेदिक पल्स वेव जनरेट करें - त्रिदोष सिद्धांत पर आधारित
        """
        normalized_time = np.mod(time_array, self.cardiac_cycle_duration) / self.cardiac_cycle_duration
        
        # 1. वत (VATA) - तेज़ चढ़ाई
        vata_component = 0.5 * (1 + np.tanh(normalized_time * 20 * (1 - self.vata_strength)))
        vata_decay = np.exp(-normalized_time * 3)
        vata_wave = vata_component * vata_decay
        
        # 2. पित्त (PITTA) - Dicrotic Notch
        notch_position = 0.65
        notch_width = 0.15 * (1 - self.pitta_strength)
        pitta_component = -np.exp(-((normalized_time - notch_position) ** 2) / (2 * notch_width ** 2))
        
        tidal_position = 0.75
        tidal_width = 0.1
        tidal_bounce = 0.3 * np.exp(-((normalized_time - tidal_position) ** 2) / (2 * tidal_width ** 2))
        pitta_wave = pitta_component + tidal_bounce
        
        # 3. कफ (KAPHA) - चौड़ा बेस
        kapha_decay_rate = 2 * (1 - self.kapha_strength)
        kapha_wave = np.exp(-normalized_time * kapha_decay_rate)
        
        combined_wave = (
            self.vata_strength * vata_wave +
            self.pitta_strength * pitta_wave +
            self.kapha_strength * kapha_wave
        )
        
        wave_min = np.min(combined_wave)
        wave_max = np.max(combined_wave)
        
        if wave_max > wave_min:
            normalized_pulse = (combined_wave - wave_min) / (wave_max - wave_min)
        else:
            normalized_pulse = np.zeros_like(combined_wave)
            
        return normalized_pulse
    
    def generate_batch(self):
        """
        50 सैंपल्स का एक बैच जनरेट करें (DC Offset और Noise के साथ)
        """
        dt = 1.0 / self.sampling_rate
        batch_time = np.arange(0, self.batch_size * dt, dt)
        actual_time = batch_time + self.current_time
        
        # 0.0 से 1.0 के बीच वेव
        base_wave = self.generate_ayurvedic_pulse_wave(actual_time)
        
        # === 12-bit ADC सिमुलेशन ===
        # असली सेंसर की तरह 2048 का DC Offset + 500 का एम्पलीट्यूड 
        # (यह आपके DSP के Transient Shockwave फिक्स को टेस्ट करने के लिए परफेक्ट है)
        batch_data = 2048.0 + (base_wave * 500.0)
        
        # === NOISE BUG FIX ===
        # Reduce noise to 0.5% for a clean, clinical Ayurvedic wave
        # शोर (Noise) को 0.05 से 0.005 कर दिया गया है
        noise_amplitude = 500.0 * 0.005 
        noise = noise_amplitude * np.random.randn(self.batch_size)
        batch_data += noise
        
        # लि니어 इंटरपोलेशन - स्मूथ ट्रांजिशन
        transition_samples = 5
        for i in range(transition_samples):
            alpha = i / transition_samples
            batch_data[i] = (1 - alpha) * self.previous_value + alpha * batch_data[i]
        
        self.previous_value = batch_data[-1]
        self.current_time += self.batch_size * dt
        
        if self.current_time > 10 * self.cardiac_cycle_duration:
            self.current_time = 0.0
            
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
        print("Virtual Sensor stopped - वर्चुअल सेंसर रुक गया")
    
    def start(self):
        super().start()
        print(f"Virtual Sensor started - वर्चुअल सेंसर शुरू हो गया")
        print(f"Tridosha - Vata: {self.vata_strength:.2f}, Pitta: {self.pitta_strength:.2f}, Kapha: {self.kapha_strength:.2f}")
    
    def stop(self):
        self._stop_event.set()
        self.join(timeout=1.0)
        print("Virtual Sensor stopped successfully - वर्चुअल सेंसर सफलतापूर्वक रुक गया")
    
    def get_latest_batch(self, timeout=1.0):
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_queue_size(self):
        return self.data_queue.qsize()


if __name__ == "__main__":
    print("=" * 70)
    print("Ayurvedic Nadi Pariksha Generator Test (with ADC offset)")
    print("=" * 70)
    
    sensor = VirtualSensor()
    sensor.start()
    
    try:
        batch = sensor.get_latest_batch(timeout=1.0)
        if batch is not None:
            print(f"\nTested First Batch Shape: {batch.shape}")
            print(f"Min Val: {np.min(batch):.2f} (Should be around 2048)")
            print(f"Max Val: {np.max(batch):.2f} (Should be around 2548)")
    finally:
        sensor.stop()
