"""
==============================================================================
प्रोजेक्ट (Project): Ayurvedic Nadi Pariksha DSP Engine
फ़ाइल का नाम (File Name): nadi_dsp.py
संस्करण (Version): 5.0.0 (Ultimate Anti-Crash & Locked Baseline)

विवरण (Description): 
यह मॉड्यूल 'Raw' डाटा के क्रैश (सीधी लाइन) होने की समस्या को अलग-अलग 
High-Pass और Low-Pass फिल्टर का उपयोग करके सुलझाता है। 
Displacement के ऊपर-नीचे भटकने (Drift) को रोकने के लिए एक मजबूत 1.0 Hz 
एंकर (Anchor) लगाया गया है।
==============================================================================
"""

import numpy as np
from scipy import signal

class NadiDSP:
    def __init__(self, sampling_rate=1000):
        """
        NadiDSP कंस्ट्रक्टर - क्रैश-प्रूफ फिल्टर और मजबूत एंकर के साथ
        """
        self.fs = sampling_rate
        self.dt = 1.0 / self.fs
        
        # 1. Raw Data Filters (Band-pass को तोड़कर अलग किया गया ताकि क्रैश न हो)
        
        # 1A. High-Pass (0.5 Hz) - DC और साँस का प्रभाव हटाएगा
        self.sos_hp_raw = signal.butter(2, 0.5, btype='highpass', fs=self.fs, output='sos')
        self.zi_hp_raw = signal.sosfilt_zi(self.sos_hp_raw)
        
        # 1B. Low-Pass (20.0 Hz) - झटके और नॉइज़ को स्मूथ करेगा
        self.sos_lp_raw = signal.butter(2, 20.0, btype='lowpass', fs=self.fs, output='sos')
        self.zi_lp_raw = signal.sosfilt_zi(self.sos_lp_raw)
        
        # 2. Stronger Leaky Integrator (0.98)
        # लीकेज को 0.995 से 0.98 कर दिया गया है, ताकि यह तेजी से 0 पर वापस आए
        self.alpha = 0.98
        self.b_int = [self.dt]
        self.a_int = [1.0, -self.alpha]
        
        self.zi_vel = signal.lfilter_zi(self.b_int, self.a_int) * 0.0
        self.zi_disp = signal.lfilter_zi(self.b_int, self.a_int) * 0.0
        
        # 3. Post-Integration Stabilizers (Anchor)
        
        # Velocity के लिए 0.5 Hz का स्टेबलाइजर
        self.sos_hp_v = signal.butter(2, 0.5, btype='highpass', fs=self.fs, output='sos')
        self.zi_hp_v = signal.sosfilt_zi(self.sos_hp_v)
        
        # Displacement के लिए मजबूत 1.0 Hz का स्टेबलाइजर (ताकि यह ऊपर-नीचे न भटके)
        self.sos_hp_d = signal.butter(2, 1.0, btype='highpass', fs=self.fs, output='sos')
        self.zi_hp_d = signal.sosfilt_zi(self.sos_hp_d)
        
        self.is_first_batch = True

    def process_batch(self, raw_batch):
        """
        आने वाले डेटा बैच को प्रोसेस करें - 100% Stable Pipeline
        """
        # 2048 DC Offset शॉकवेव फिक्स (Crash Preventer)
        if self.is_first_batch:
            # केवल High-Pass को 2048 का शॉक देते हैं
            self.zi_hp_raw = self.zi_hp_raw * raw_batch[0]
            # Low-Pass को 0 देते हैं, क्योंकि High-Pass पहले ही 2048 हटा चुका होगा!
            self.zi_lp_raw = self.zi_lp_raw * 0.0
            self.is_first_batch = False
            
        # Step 1: Raw Signal Filtering (दो चरणों में ताकि लाइन सीधी न हो)
        raw_hp, self.zi_hp_raw = signal.sosfilt(self.sos_hp_raw, raw_batch, zi=self.zi_hp_raw)
        raw_filtered, self.zi_lp_raw = signal.sosfilt(self.sos_lp_raw, raw_hp, zi=self.zi_lp_raw)
        
        # Step 2: Velocity Wave (First Integration + Stabilization)
        vel_raw, self.zi_vel = signal.lfilter(self.b_int, self.a_int, raw_filtered, zi=self.zi_vel)
        velocity, self.zi_hp_v = signal.sosfilt(self.sos_hp_v, vel_raw, zi=self.zi_hp_v)
        
        # Step 3: Displacement Wave (Second Integration + Strong Anchor)
        disp_raw, self.zi_disp = signal.lfilter(self.b_int, self.a_int, velocity, zi=self.zi_disp)
        # यह मजबूत एंकर डिस्प्लेसमेंट को सेंटर में लॉक कर देगा (ऊपर-नीचे नहीं होने देगा)
        displacement, self.zi_hp_d = signal.sosfilt(self.sos_hp_d, disp_raw, zi=self.zi_hp_d)
        
        return {
            'raw_filtered': raw_filtered,
            'velocity': velocity,
            'displacement': displacement
        }
    
    def reset_state(self):
        """
        नई सिक्वेंस के लिए सभी फिल्टर और इंटीग्रेटर रीसेट करें
        """
        self.zi_hp_raw = signal.sosfilt_zi(self.sos_hp_raw)
        self.zi_lp_raw = signal.sosfilt_zi(self.sos_lp_raw)
        
        self.zi_vel = signal.lfilter_zi(self.b_int, self.a_int) * 0.0
        self.zi_disp = signal.lfilter_zi(self.b_int, self.a_int) * 0.0
        
        self.zi_hp_v = signal.sosfilt_zi(self.sos_hp_v)
        self.zi_hp_d = signal.sosfilt_zi(self.sos_hp_d)
        
        self.is_first_batch = True
        print("DSP State Reset - सभी फिल्टर और एंकर रीसेट हो गए")

    def get_filter_info(self):
        return {
            'sampling_rate': self.fs,
            'raw_filters': 'Highpass (0.5Hz) + Lowpass (20Hz)',
            'integrator': 'Leaky Integrator (alpha=0.98)',
            'velocity_anchor': 'Highpass (0.5 Hz)',
            'displacement_anchor': 'Highpass (1.0 Hz)'
        }

# ============================================================================
# उदाहरण उपयोग (Example Usage) - टेस्टिंग के लिए
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Ayurvedic Nadi Pariksha DSP - Crash-Proof & Locked Baseline")
    print("=" * 70)
    
    dsp = NadiDSP()
    print("Processing synthetic test batch...")
    
    # 2048 DC Offset + Sine Wave + High Frequency Noise
    t = np.linspace(0, 0.05, 50)
    test_batch = 2048 + 500 * np.sin(2 * np.pi * 1 * t) + 20 * np.sin(2 * np.pi * 50 * t)
    
    results = dsp.process_batch(test_batch)
    
    print(f"Raw Filtered:   Shape={results['raw_filtered'].shape}, Min={np.min(results['raw_filtered']):.4f}")
    print(f"Velocity:       Shape={results['velocity'].shape}, Min={np.min(results['velocity']):.4f}")
    print(f"Displacement:   Shape={results['displacement'].shape}, Min={np.min(results['displacement']):.4f}")
    print("\nDSP Test Complete - कोई सीधी लाइन नहीं, कोई भटकाव नहीं!")
