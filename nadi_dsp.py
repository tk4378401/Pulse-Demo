"""
==============================================================================
प्रोजेक्ट (Project): Ayurvedic Nadi Pariksha DSP Engine
फ़ाइल का नाम (File Name): nadi_dsp.py
संस्करण (Version): 7.0.0 (True Mathematical Double Integration)

विवरण:
यह मॉड्यूल अब कोई 'फेक एंकर' इस्तेमाल नहीं करता। यह असली 'cumulative_trapezoid' 
का उपयोग करके 100% शुद्ध डबल इंटीग्रेशन करता है। DC स्पाइक को रोकने के लिए 
इसमें Dynamic Baseline Tracking का उपयोग किया गया है।
==============================================================================
"""

import numpy as np
from scipy import signal
from scipy import integrate

class NadiDSP:
    def __init__(self, sampling_rate=1000):
        self.fs = sampling_rate
        self.dt = 1.0 / self.fs
        
        # 0.5 Hz का स्टैण्डर्ड Butterworth हाई-पास फिल्टर (ड्रिफ्ट को धीरे से रोकने के लिए)
        self.sos_hp = signal.butter(2, 0.5, btype='highpass', fs=self.fs, output='sos')
        
        self.zi_raw = signal.sosfilt_zi(self.sos_hp)
        self.zi_vel = signal.sosfilt_zi(self.sos_hp)
        self.zi_disp = signal.sosfilt_zi(self.sos_hp)
        
        self.last_vel = 0.0
        self.last_disp = 0.0
        
        self.is_first_batch = True
        self.dc_baseline = 0.0

    def process_batch(self, raw_batch):
        # ==========================================================
        # 1. DYNAMIC DC REMOVAL (बिना स्पाइक के 2048 को हटाना)
        # ==========================================================
        if self.is_first_batch:
            self.dc_baseline = raw_batch[0]
            self.is_first_batch = False
            
        # डेटा को सेंटर (0) पर लाएं
        raw_centered = raw_batch - self.dc_baseline
        
        # बेसलाइन को धीरे-धीरे अपडेट करें (साँस के प्रभाव को ट्रैक करने के लिए)
        self.dc_baseline = 0.999 * self.dc_baseline + 0.001 * np.mean(raw_batch)

        # ==========================================================
        # STEP 1: RAW SIGNAL
        # ==========================================================
        raw_hp, self.zi_raw = signal.sosfilt(self.sos_hp, raw_centered, zi=self.zi_raw)

        # ==========================================================
        # STEP 2: VELOCITY (TRUE FIRST INTEGRATION)
        # ==========================================================
        # असली गणितीय समाकलन (True Integration)
        vel_int = integrate.cumulative_trapezoid(raw_hp, dx=self.dt, initial=0)
        vel_raw = self.last_vel + vel_int
        self.last_vel = vel_raw[-1]
        
        # गति को 0 लाइन पर स्थिर रखने के लिए फिल्टर
        vel_hp, self.zi_vel = signal.sosfilt(self.sos_hp, vel_raw, zi=self.zi_vel)

        # ==========================================================
        # STEP 3: DISPLACEMENT (TRUE SECOND INTEGRATION)
        # ==========================================================
        # असली गणितीय समाकलन (True Double Integration)
        disp_int = integrate.cumulative_trapezoid(vel_hp, dx=self.dt, initial=0)
        disp_raw = self.last_disp + disp_int
        self.last_disp = disp_raw[-1]
        
        # विस्थापन को 0 लाइन पर स्थिर रखने के लिए फिल्टर
        disp_hp, self.zi_disp = signal.sosfilt(self.sos_hp, disp_raw, zi=self.zi_disp)
        
        # गणित का नियम: दो बार इंटीग्रेट करने से सिग्नल उल्टा (180 phase shift) हो जाता है।
        # इसलिए हम इसे सीधा (-1 से गुणा) कर रहे हैं ताकि ग्राफ़ सही दिखे।
        disp_hp = -disp_hp

        return {
            'raw_filtered': raw_hp,
            'velocity': vel_hp,
            'displacement': disp_hp
        }
    
    def reset_state(self):
        self.zi_raw = signal.sosfilt_zi(self.sos_hp)
        self.zi_vel = signal.sosfilt_zi(self.sos_hp)
        self.zi_disp = signal.sosfilt_zi(self.sos_hp)
        self.last_vel = 0.0
        self.last_disp = 0.0
        self.is_first_batch = True
        self.dc_baseline = 0.0
        print("DSP State Reset - True Integrators Ready")

    def get_filter_info(self):
        return {
            'sampling_rate': self.fs,
            'integration': 'True Cumulative Trapezoid',
            'baseline_correction': 'Dynamic EMA + 0.5Hz Butterworth'
        }
