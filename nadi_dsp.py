"""
==============================================================================
प्रोजेक्ट (Project): Ayurvedic Nadi Pariksha DSP Engine
फ़ाइल का नाम (File Name): nadi_dsp.py
संस्करण (Version): 2.0.1 (Seamless Integration Fixed)

विवरण (Description): 
यह मॉड्यूल लगातार आ रहे डेटा बैच को प्रोसेस करता है और डबल इंटीग्रेशन के माध्यम से
वेवफॉर्म की वेलोसिटी (Velocity) और डिस्प्लेसमेंट (Displacement - वात, पित्त, कफ)
कैलकुलेट करता है। यह 100% सीमलेस बाउंड्री और शॉकवेव-फ्री (Drift-Free) है।
==============================================================================
"""

import numpy as np                      # न्यूमेरिकल कंप्यूटेशन के लिए (Numerical computation)
from scipy import signal                # सिग्नल प्रोसेसिंग और फिल्टर डिज़ाइन के लिए (Signal processing & filter design)
from scipy import integrate             # इंटीग्रेशन (समाकलन) के लिए (Integration calculus)


class NadiDSP:
    """
    नाड़ी DSP क्लास - आयुर्वेदिक पल्स वेव का डिजिटल सिग्नल प्रोसेसिंग
    Nadi DSP Class - Digital Signal Processing of Ayurvedic Pulse Wave
    """
    
    def __init__(self, sampling_rate=1000, highpass_cutoff=0.1):
        """
        NadiDSP का कंस्ट्रक्टर - फिल्टर और स्टेट वेरिएबल्स को इनिशियलाइज करें
        """
        self.fs = sampling_rate
        self.dt = 1.0 / self.fs
        self.highpass_cutoff = highpass_cutoff
        
        # 0.1Hz Highpass Filter (2nd order, SOS format)
        self.sos_hp = signal.butter(2, self.highpass_cutoff, 'highpass', fs=self.fs, output='sos')
        
        # Filter Initial Conditions (zi)
        self.zi_hp1 = signal.sosfilt_zi(self.sos_hp)
        self.zi_hp2 = signal.sosfilt_zi(self.sos_hp)
        self.zi_hp3 = signal.sosfilt_zi(self.sos_hp)
        
        # Boundary States (पिछली वेव को नई वेव से जोड़ने के लिए - Seamless Integration)
        self.last_raw_filtered = 0.0
        self.last_velocity_raw = 0.0
        self.last_velocity_filtered = 0.0
        self.last_displacement_raw = 0.0
        
        # शॉकवेव को रोकने के लिए फ्लैग
        self.is_first_batch = True

    def process_batch(self, raw_batch):
        """
        आने वाले डेटा बैच को प्रोसेस करें - 100% Drift-Free DSP Pipeline
        """
        # 1. 2048 का शॉकवेव रोकें (Scale Initial State to DC Offset)
        # जब पहला डेटा आए, तो फिल्टर की इनिशियल स्टेट को डेटा के लेवल पर सेट करें
        if self.is_first_batch:
            self.zi_hp1 = self.zi_hp1 * raw_batch[0]
            
        # Step 1: Raw Data को फिल्टर करें (DC Offset हटाता है)
        raw_filtered, self.zi_hp1 = signal.sosfilt(self.sos_hp, raw_batch, zi=self.zi_hp1)
        
        # Step 2: Velocity (First Integration) - 100% Seamless
        if self.is_first_batch:
            vel_integ = integrate.cumulative_trapezoid(raw_filtered, dx=self.dt, initial=0)
            velocity_raw = vel_integ
        else:
            # पिछले पैकेट के आखिरी सैंपल को जोड़कर ट्रैपेज़ॉइड बनाएं
            arr_for_vel = np.insert(raw_filtered, 0, self.last_raw_filtered)
            vel_integ = integrate.cumulative_trapezoid(arr_for_vel, dx=self.dt, initial=0)
            # एक्स्ट्रा सैंपल हटाकर पिछली टोटल वैल्यू जोड़ दें
            velocity_raw = vel_integ[1:] + self.last_velocity_raw
            
        # स्टेट्स अपडेट करें (अगले बैच के लिए)
        self.last_raw_filtered = raw_filtered[-1]
        self.last_velocity_raw = velocity_raw[-1]
        
        # Velocity को फिल्टर करें (Drift हटाने के लिए)
        velocity, self.zi_hp2 = signal.sosfilt(self.sos_hp, velocity_raw, zi=self.zi_hp2)
        
        # Step 3: Displacement (Second Integration) - 100% Seamless
        if self.is_first_batch:
            disp_integ = integrate.cumulative_trapezoid(velocity, dx=self.dt, initial=0)
            displacement_raw = disp_integ
        else:
            # पिछले पैकेट के आखिरी सैंपल को जोड़कर ट्रैपेज़ॉइड बनाएं
            arr_for_disp = np.insert(velocity, 0, self.last_velocity_filtered)
            disp_integ = integrate.cumulative_trapezoid(arr_for_disp, dx=self.dt, initial=0)
            # एक्स्ट्रा सैंपल हटाकर पिछली टोटल वैल्यू जोड़ दें
            displacement_raw = disp_integ[1:] + self.last_displacement_raw
            
        # स्टेट्स अपडेट करें (अगले बैच के लिए)
        self.last_velocity_filtered = velocity[-1]
        self.last_displacement_raw = displacement_raw[-1]
        
        # Displacement को फिल्टर करें (Final Drift Removal)
        displacement, self.zi_hp3 = signal.sosfilt(self.sos_hp, displacement_raw, zi=self.zi_hp3)
        
        self.is_first_batch = False
        
        return {
            'raw_filtered': raw_filtered,  # Baseline corrected raw wave
            'velocity': velocity,          # Smooth Velocity
            'displacement': displacement   # The Vata-Pitta-Kapha Wave
        }
    
    def reset_state(self):
        """
        सभी स्टेट्स को रीसेट करें - नई सिक्वेंस के लिए
        """
        self.zi_hp1 = signal.sosfilt_zi(self.sos_hp)
        self.zi_hp2 = signal.sosfilt_zi(self.sos_hp)
        self.zi_hp3 = signal.sosfilt_zi(self.sos_hp)
        
        self.last_raw_filtered = 0.0
        self.last_velocity_raw = 0.0
        self.last_velocity_filtered = 0.0
        self.last_displacement_raw = 0.0
        
        self.is_first_batch = True
        print("DSP State reset - DSP स्टेट रीसेट हो गया (Seamless state clear)")
    
    def get_filter_info(self):
        """
        फिल्टर की जानकारी प्राप्त करें
        """
        info = {
            'sampling_rate': self.fs,
            'cutoff_frequency': self.highpass_cutoff,
            'filter_type': 'Butterworth Highpass (SOS)',
            'filter_order': 2,
            'nyquist': self.fs / 2.0
        }
        return info


# ============================================================================
# उदाहरण उपयोग (Example Usage) - टेस्टिंग के लिए
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Ayurvedic Nadi Pariksha DSP Engine - 100% Seamless Version")
    print("=" * 70)
    
    dsp = NadiDSP()
    print("Processing synthetic DC offset test batch...")
    
    # 2048 DC Offset + Noise के साथ टेस्ट
    t = np.linspace(0, 0.05, 50)
    test_batch = 2048 + 500 * np.sin(2 * np.pi * 1 * t) + 2.5 * np.random.randn(50)
    
    results = dsp.process_batch(test_batch)
    
    print(f"Raw Filtered:   Shape={results['raw_filtered'].shape}")
    print(f"Velocity:       Shape={results['velocity'].shape}")
    print(f"Displacement:   Shape={results['displacement'].shape}")
    print("\nDSP Test Complete - DSP परीक्षण पूर्ण")
