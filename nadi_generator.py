"""
==============================================================================
प्रोजेक्ट (Project): Ayurvedic Nadi Pariksha
फ़ाइल (File): nadi_generator.py
विवरण: 100% बायोलॉजिकल और क्लिनिकल आर्टिरियल पल्स वेव जनरेटर (Real Human Pulse)
==============================================================================
"""

import numpy as np
import threading
import queue
import time

class VirtualSensor(threading.Thread):
    """
    वर्चुअल सेंसर - असली इंसान जैसी (Bio-Realistic) पल्स वेव बनाता है
    """
    def __init__(self, sampling_rate=1000, batch_size=50, vata_strength=0.8, pitta_strength=0.5, kapha_strength=0.4):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        
        # त्रिदोष तीव्रता (असली पल्स के अनुपात में)
        self.vata_strength = vata_strength      # सिस्टोलिक पीक (Systolic Peak)
        self.pitta_strength = pitta_strength    # डिक्रोटिक/टाइडल वेव (Dicrotic Notch)
        self.kapha_strength = kapha_strength    # डायस्टोलिक रनऑफ (Diastolic Runoff)
        
        self._stop_event = threading.Event()
        self.data_queue = queue.Queue()
        self.current_time = 0.0
        
        self.base_heart_rate = 70.0  # 70 BPM (सामान्य दिल की धड़कन)
        self.previous_value = 2048.0
        
    def generate_bio_realistic_wave(self, time_array):
        """
        मल्टी-गॉसियन मॉडल (Multi-Gaussian Model) का उपयोग करके
        एकदम असली मेडिकल PPG/Arterial वेव जनरेट करें।
        """
        # हार्ट रेट वैरियबिलिटी (HRV) - असली इंसान का दिल मशीन की तरह फिक्स नहीं धड़कता
        # साँस लेने (Respiration) के साथ धड़कन थोड़ी बदलती है (RSA)
        current_hr = self.base_heart_rate + 3.0 * np.sin(2 * np.pi * 0.25 * time_array)
        cardiac_cycle_duration = 60.0 / current_hr
        
        # कार्डियक साइकिल के अंदर समय को 0 से 1 तक नॉर्मलाइज करें
        normalized_time = np.mod(time_array, cardiac_cycle_duration) / cardiac_cycle_duration
        
        # 1. वात (Vata) - मुख्य सिस्टोलिक पीक (तेज़ और तीखा)
        # खून का पहला तेज़ बहाव
        vata_peak = self.vata_strength * 1.0 * np.exp(-((normalized_time - 0.15) ** 2) / (2 * 0.04 ** 2))
        
        # 2. पित्त (Pitta) - टाइडल वेव / डिक्रोटिक नॉच (मध्यम)
        # आर्टरी (धमनी) से टकराकर वापस आने वाला खून
        pitta_peak = self.pitta_strength * 0.45 * np.exp(-((normalized_time - 0.38) ** 2) / (2 * 0.06 ** 2))
        
        # 3. कफ (Kapha) - डायस्टोलिक रनऑफ (धीमा और चौड़ा)
        # खून का धीमा बहाव जब दिल आराम कर रहा होता है
        kapha_peak = self.kapha_strength * 0.25 * np.exp(-((normalized_time - 0.65) ** 2) / (2 * 0.12 ** 2))
        
        # तीनों को मिलाकर असली पल्स शेप बनाएं
        bio_wave = vata_peak + pitta_peak + kapha_peak
        return bio_wave
        
    def generate_batch(self):
        dt = 1.0 / self.sampling_rate
        batch_time = np.arange(0, self.batch_size * dt, dt)
        actual_time = batch_time + self.current_time
        
        # 0.0 से 1.0 के बीच मेडिकल वेव
        base_wave = self.generate_bio_realistic_wave(actual_time)
        
        # 1. साँस लेने का प्रभाव (Respiratory Baseline Wander)
        # इंसान जब साँस लेता है तो ग्राफ थोड़ा ऊपर-नीचे होता है (15 साँस प्रति मिनट = 0.25Hz)
        respiration_wander = 40.0 * np.sin(2 * np.pi * 0.25 * actual_time)
        
        # 2. असली सेंसर जैसा DC Offset (2048) और एम्पलीट्यूड
        batch_data = 2048.0 + (base_wave * 600.0) + respiration_wander
        
        # 3. हल्का सा मेडिकल सेंसर नॉइज़ (बहुत कम, ताकि वेव गंदी न लगे)
        noise = 3.0 * np.random.randn(self.batch_size)
        batch_data += noise
        
        # लि니어 इंटरपोलेशन - बैच के बीच कोई कट (Jump) न लगे
        transition_samples = 3
        for i in range(transition_samples):
            alpha = (i + 1) / (transition_samples + 1)
            batch_data[i] = (1 - alpha) * self.previous_value + alpha * batch_data[i]
            
        self.previous_value = batch_data[-1]
        self.current_time += self.batch_size * dt
        
        # समय को बहुत बड़ा होने से रोकें
        if self.current_time > 3600:
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
        print("Bio-Realistic Virtual Sensor Started - असली पल्स सिमुलेशन शुरू")
        
    def stop(self):
        self._stop_event.set()
        self.join(timeout=1.0)
        print("Sensor stopped.")
        
    def get_latest_batch(self, timeout=1.0):
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
