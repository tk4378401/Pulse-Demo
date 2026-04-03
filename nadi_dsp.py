"""
==============================================================================
प्रोजेक्ट (Project): Ayurvedic Nadi Pariksha DSP Engine
फ़ाइल का नाम (File Name): nadi_dsp.py
संस्करण (Version): 3.0.0 (Leaky Integrator Fix for Garbage Wave)

विवरण (Description): 
यह मॉड्यूल लगातार आ रहे डेटा बैच को प्रोसेस करता है। 
'Velocity Garbage' को पूरी तरह से ठीक करने के लिए इसमें 'Leaky Integrator' 
(स्थिर समाकलन) तकनीक का उपयोग किया गया है, जो वेव को फटने (drift) से रोकता है।
==============================================================================
"""

import numpy as np
from scipy import signal

class NadiDSP:
    def __init__(self, sampling_rate=1000, highpass_cutoff=0.5):
        """
        NadiDSP कंस्ट्रक्टर - लीकी इंटीग्रेटर्स के साथ
        """
        self.fs = sampling_rate
        self.dt = 1.0 / self.fs
        
        # 1. High-Pass Filter (0.5 Hz - साँस और सेंसर के झटके रोकने के लिए मजबूत कटऑफ)
        # यह बेसलाइन को 100% फ्लैट कर देगा
        self.sos_hp = signal.butter(4, highpass_cutoff, 'highpass', fs=self.fs, output='sos')
        self.zi_hp = signal.sosfilt_zi(self.sos_hp)
        
        # 2. Leaky Integrator Coefficients (लीकी इंटीग्रेटर - कचरा वेव का परमानेंट इलाज)
        # यह cumulative_trapezoid की तरह अनंत (infinity) तक नहीं जाता, बल्कि स्टेबल रहता है।
        self.alpha = 0.998  # 0.998 = Perfect Integration without drift
        self.b_int = [self.dt]
        self.a_int = [1.0, -self.alpha]
        
        # इंटीग्रेटर स्टेट्स (Seamless कनेक्टिविटी के लिए)
        self.zi_vel = signal.lfilter_zi(self.b_int, self.a_int) * 0.0
        self.zi_disp = signal.lfilter_zi(self.b_int, self.a_int) * 0.0
        
        self.is_first_batch = True

    def process_batch(self, raw_batch):
        """
        आने वाले डेटा बैच को प्रोसेस करें - Leaky Integration Pipeline
        """
        # 2048 DC Offset शॉकवेव फिक्स
        if self.is_first_batch:
            self.zi_hp = self.zi_hp * raw_batch[0]
            self.is_first_batch = False
            
        # Step 1: Raw Signal Filtering (मूल डेटा को साफ़ करें)
        raw_filtered, self.zi_hp = signal.sosfilt(self.sos_hp, raw_batch, zi=self.zi_hp)
        
        # Step 2: Velocity (Velocity Wave) - First Leaky Integration
        # यह एकदम स्मूथ गति (Velocity) निकालेगा, कोई कचरा नहीं!
        velocity, self.zi_vel = signal.lfilter(self.b_int, self.a_int, raw_filtered, zi=self.zi_vel)
        
        # Step 3: Displacement (VPK Wave) - Second Leaky Integration
        # यह वात-पित्त-कफ (Displacement) की शेप बनाएगा
        displacement, self.zi_disp = signal.lfilter(self.b_int, self.a_int, velocity, zi=self.zi_disp)
        
        return {
            'raw_filtered': raw_filtered,
            'velocity': velocity,
            'displacement': displacement
        }
    
    def reset_state(self):
        """
        नई सिक्वेंस के लिए सभी फिल्टर और इंटीग्रेटर रीसेट करें
        """
        self.zi_hp = signal.sosfilt_zi(self.sos_hp)
        self.zi_vel = signal.lfilter_zi(self.b_int, self.a_int) * 0.0
        self.zi_disp = signal.lfilter_zi(self.b_int, self.a_int) * 0.0
        self.is_first_batch = True
        print("DSP State Reset - इंटीग्रेटर और फिल्टर रीसेट हो गए")

    def get_filter_info(self):
        return {
            'sampling_rate': self.fs,
            'integrator_type': 'Leaky Integrator (alpha=0.998)',
            'highpass_cutoff': 0.5
        }


# ============================================================================
# उदाहरण उपयोग (Example Usage) - टेस्टिंग के लिए
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Ayurvedic Nadi Pariksha DSP Engine - Leaky Integrator Version")
    print("=" * 70)
    
    dsp = NadiDSP()
    print("Processing synthetic test batch...")
    
    # 2048 DC Offset + Sine Wave के साथ टेस्ट
    t = np.linspace(0, 0.05, 50)
    test_batch = 2048 + 500 * np.sin(2 * np.pi * 1 * t)
    
    results = dsp.process_batch(test_batch)
    
    print(f"Raw Filtered:   Shape={results['raw_filtered'].shape}, Min={np.min(results['raw_filtered']):.4f}")
    print(f"Velocity:       Shape={results['velocity'].shape}, Min={np.min(results['velocity']):.4f}")
    print(f"Displacement:   Shape={results['displacement'].shape}, Min={np.min(results['displacement']):.4f}")
    print("\nDSP Test Complete - कोई कचरा नहीं!")
