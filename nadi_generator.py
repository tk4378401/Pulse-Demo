"""
nadi_generator.py - आयुर्वेदिक नाड़ी परीक्षण के लिए सिंथेटिक पल्स वेव जनरेटर
Synthetic Pulse Wave Generator for Ayurvedic Nadi Pariksha

यह मॉड्यूल 1000Hz सैम्पलिंग रेट पर आर्टिरियल पल्स वेव बनाता है। 
इसमें वात, पित्त और कफ दोषों के लिए अलग-अलग प्रोफाइल (Heart Rate + Wave Shape) जोड़े गए हैं।
"""

import numpy as np
import scipy.signal as signal
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
        self.current_time = 0.0
        self.previous_value = 0.0
        
        # डिफ़ॉल्ट रूप से 'संतुलित' (Balanced) प्रोफाइल सेट करें
        self.set_dosha_profile('balanced')
        
    def set_dosha_profile(self, dosha_type):
        """
        आयुर्वेदिक सिद्धांत के अनुसार हार्ट रेट और वेव का आकार बदलें
        """
        if dosha_type == 'vata':
            # वात (सर्प गति): तेज़, तीखी, अनियमित
            self.vata_strength = 0.95
            self.pitta_strength = 0.15
            self.kapha_strength = 0.15
            self.heart_rate = 85.0  # तेज़ धड़कन
        elif dosha_type == 'pitta':
            # पित्त (मंडूक गति): स्पष्ट उछाल (Dicrotic Notch)
            self.vata_strength = 0.30
            self.pitta_strength = 0.95
            self.kapha_strength = 0.20
            self.heart_rate = 75.0  # मध्यम धड़कन
        elif dosha_type == 'kapha':
            # कफ (हंस गति): धीमी, चौड़ी और शांत
            self.vata_strength = 0.15
            self.pitta_strength = 0.20
            self.kapha_strength = 0.95
            self.heart_rate = 60.0  # धीमी धड़कन
        else: # balanced
            # त्रिदोष संतुलन
            self.vata_strength = 0.5
            self.pitta_strength = 0.5
            self.kapha_strength = 0.5
            self.heart_rate = 70.0
            
        self.heart_frequency = self.heart_rate / 60.0
        self.cardiac_cycle_duration = 1.0 / self.heart_frequency
        
        # स्विच करते समय वेव को टूटने से बचाने के लिए टाइमर रीसेट करें
        self.current_time = 0.0
        print(f"Dosha Profile Changed: {dosha_type.upper()} | HR: {self.heart_rate} BPM")

    def generate_ayurvedic_pulse_wave(self, time_array):
        # समय सरणी को नॉर्मलाइज करें
        normalized_time = np.mod(time_array, self.cardiac_cycle_duration) / self.cardiac_cycle_duration
        
        # 1. वात (VATA) - Anacrotic Limb (तेज़ चढ़ाई)
        vata_component = 0.5 * (1 + np.tanh(normalized_time * 20 * (1 - self.vata_strength)))
        vata_decay = np.exp(-normalized_time * 3)
        vata_wave = vata_component * vata_decay
        
        # 2. पित्त (PITTA) - Dicrotic Notch (मंडूक उछाल)
        notch_position = 0.65
        notch_width = 0.15 * (1 - self.pitta_strength)
        pitta_component = -np.exp(-((normalized_time - notch_position) ** 2) / (2 * notch_width ** 2))
        
        tidal_position = 0.75
        tidal_width = 0.1
        tidal_bounce = 0.3 * np.exp(-((normalized_time - tidal_position) ** 2) / (2 * tidal_width ** 2))
        pitta_wave = pitta_component + tidal_bounce
        
        # 3. कफ (KAPHA) - Diastolic Runoff (हंस जैसी चौड़ी वेव)
        kapha_decay_rate = 2 * (1 - self.kapha_strength)
        kapha_wave = np.exp(-normalized_time * kapha_decay_rate)
        
        # तीनों दोषों को मिलाएं
        combined_wave = (
            self.vata_strength * vata_wave +
            self.pitta_strength * pitta_wave +
            self.kapha_strength * kapha_wave
        )
        
        # नॉर्मलाइजेशन
        wave_min = np.min(combined_wave)
        wave_max = np.max(combined_wave)
        if wave_max > wave_min:
            normalized_pulse = 0.2 + 0.8 * (combined_wave - wave_min) / (wave_max - wave_min)
        else:
            normalized_pulse = np.ones_like(combined_wave) * 0.6
            
        return normalized_pulse
    
    def generate_batch(self):
        dt = 1.0 / self.sampling_rate
        batch_time = np.arange(0, self.batch_size * dt, dt)
        actual_time = batch_time + self.current_time
        
        # 0.0 से 1.0 के बीच वेव
        base_wave = self.generate_ayurvedic_pulse_wave(actual_time)
        
        # 2048 DC Offset + 600 Amplitude
        batch_data = 2048.0 + (base_wave * 600.0)
        
        # नॉइज़ और बेसलाइन वांडर
        respiration = 20.0 * np.sin(2 * np.pi * 0.25 * actual_time)
        noise = 2.0 * np.random.randn(self.batch_size)
        batch_data += respiration + noise
        
        # स्मूथ ट्रांजिशन
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
