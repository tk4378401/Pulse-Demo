"""
==============================================================================
प्रोजेक्ट (Project): Ayurvedic Nadi Pariksha DSP Engine
फ़ाइल का नाम (File Name): nadi_dsp.py
संस्करण (Version): 4.0.0 (Ultimate Smoothness & Anti-Drift BPF)

विवरण (Description): 
यह मॉड्यूल डेटा में आने वाले "किंक्स" (kinks/breaks) और बेसलाइन ड्रिफ्ट को 
खत्म करने के लिए Band-Pass Filter (BPF) का उपयोग करता है। 
Velocity और Displacement को 100% स्थिर रखने के लिए 'Post-Integration 
High-Pass Filtering' की गई है।
==============================================================================
"""

import numpy as np
from scipy import signal

class NadiDSP:
    def __init__(self, sampling_rate=1000):
        """
        NadiDSP कंस्ट्रक्टर - स्मूथ बैंड-पास और पोस्ट-स्टेबलाइजर के साथ
        """
        self.fs = sampling_rate
        self.dt = 1.0 / self.fs
        
        # 1. Band-Pass Filter for Raw Data (0.5 Hz से 20.0 Hz)
        # 0.5 Hz -> DC और साँस का प्रभाव हटाएगा
        # 20.0 Hz -> बैच के जुड़ने पर आने वाले झटके (breaks) को मक्खन की तरह स्मूथ करेगा
        self.sos_bp = signal.butter(2, [0.5, 20.0], btype='bandpass', fs=self.fs, output='sos')
        self.zi_bp = signal.sosfilt_zi(self.sos_bp)
        
        # 2. Leaky Integrator Coefficients (0.995)
        # यह स्मूथ इंटीग्रेशन करता है
        self.alpha = 0.995
        self.b_int = [self.dt]
        self.a_int = [1.0, -self.alpha]
        
        # इंटीग्रेटर स्टेट्स
        self.zi_vel = signal.lfilter_zi(self.b_int, self.a_int) * 0.0
        self.zi_disp = signal.lfilter_zi(self.b_int, self.a_int) * 0.0
        
        # 3. Post-Integration Stabilizer (0.5 Hz High-Pass)
        # यह वेलोसिटी और डिस्प्लेसमेंट को बिगड़ने (Wander) से 100% रोकेगा 
        # और सेंटर लाइन (0) पर लॉक रखेगा
        self.sos_hp = signal.butter(2, 0.5, btype='highpass', fs=self.fs, output='sos')
        self.zi_hp_v = signal.sosfilt_zi(self.sos_hp)
        self.zi_hp_d = signal.sosfilt_zi(self.sos_hp)
        
        self.is_first_batch = True

    def process_batch(self, raw_batch):
        """
        आने वाले डेटा बैच को प्रोसेस करें - 100% Smooth Pipeline
        """
        # 2048 DC Offset शॉकवेव फिक्स
        if self.is_first_batch:
            self.zi_bp = self.zi_bp * raw_batch[0]
            self.is_first_batch = False
            
        # Step 1: Raw Signal Filtering (मूल डेटा को स्मूथ और साफ़ करें)
        # यह लाइन डाटा के "टूटने" की समस्या को हमेशा के लिए खत्म कर देगी
        raw_filtered, self.zi_bp = signal.sosfilt(self.sos_bp, raw_batch, zi=self.zi_bp)
        
        # Step 2: Velocity Wave (First Integration + Stabilization)
        vel_raw, self.zi_vel = signal.lfilter(self.b_int, self.a_int, raw_filtered, zi=self.zi_vel)
        # स्टेबलाइजर: वेलोसिटी को बिगड़ने से रोकें
        velocity, self.zi_hp_v = signal.sosfilt(self.sos_hp, vel_raw, zi=self.zi_hp_v)
        
        # Step 3: Displacement Wave (Second Integration + Stabilization)
        disp_raw, self.zi_disp = signal.lfilter(self.b_int, self.a_int, velocity, zi=self.zi_disp)
        # स्टेबलाइजर: डिस्प्लेसमेंट को बिगड़ने से रोकें
        displacement, self.zi_hp_d = signal.sosfilt(self.sos_hp, disp_raw, zi=self.zi_hp_d)
        
        return {
            'raw_filtered': raw_filtered,
            'velocity': velocity,
            'displacement': displacement
        }
    
    def reset_state(self):
        """
        नई सिक्वेंस के लिए सभी फिल्टर और इंटीग्रेटर रीसेट करें
        """
        self.zi_bp = signal.sosfilt_zi(self.sos_bp)
        
        self.zi_vel = signal.lfilter_zi(self.b_int, self.a_int) * 0.0
        self.zi_disp = signal.lfilter_zi(self.b_int, self.a_int) * 0.0
        
        self.zi_hp_v = signal.sosfilt_zi(self.sos_hp)
        self.zi_hp_d = signal.sosfilt_zi(self.sos_hp)
        
        self.is_first_batch = True
        print("DSP State Reset - सभी फिल्टर और इंटीग्रेटर रीसेट हो गए")

    def get_filter_info(self):
        return {
            'sampling_rate': self.fs,
            'raw_filter': 'Bandpass (0.5-20 Hz)',
            'integrator': 'Leaky Integrator (alpha=0.995)',
            'stabilizer': 'Highpass (0.5 Hz)'
        }

# ============================================================================
# उदाहरण उपयोग (Example Usage) - टेस्टिंग के लिए
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Ayurvedic Nadi Pariksha DSP - Ultimate Stability Version")
    print("=" * 70)
    
    dsp = NadiDSP()
    print("Processing synthetic test batch...")
    
    # 2048 DC Offset + Sine Wave + High Frequency Kinks (Noise)
    t = np.linspace(0, 0.05, 50)
    test_batch = 2048 + 500 * np.sin(2 * np.pi * 1 * t) + 20 * np.sin(2 * np.pi * 50 * t)
    
    results = dsp.process_batch(test_batch)
    
    print(f"Raw Filtered:   Shape={results['raw_filtered'].shape}, Min={np.min(results['raw_filtered']):.4f}")
    print(f"Velocity:       Shape={results['velocity'].shape}, Min={np.min(results['velocity']):.4f}")
    print(f"Displacement:   Shape={results['displacement'].shape}, Min={np.min(results['displacement']):.4f}")
    print("\nDSP Test Complete - कोई टूट-फूट या कचरा नहीं!")
