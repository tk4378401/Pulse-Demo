"""
nadi_dsp.py - आयुर्वेदिक नाड़ी परीक्षण के लिए DSP इंजन
(Fixed: Morphological Envelope Tracker & 0.75Hz Respiration Killer)
"""
import numpy as np
import scipy.signal as signal

class NadiDSP:
    def __init__(self, sampling_rate=1000):
        self.fs = sampling_rate
        self.dt = 1.0 / self.fs

        # ==========================================================
        # 1. MORPHOLOGICAL ENVELOPE TRACKER (पीले ग्राफ़ के लिए)
        # ==========================================================
        # यह स्मार्ट बेसलाइन साँस के झटकों को सोख लेगी। 
        # पीला ग्राफ़ कभी हवा में नहीं तैरेगा, हमेशा 0 की लाइन पर चिपका रहेगा।
        self.envelope_baseline = 0.0
        
        # ==========================================================
        # 2. RESPIRATION KILLER (इंटीग्रेशन/मैथ के लिए)
        # ==========================================================
        # 0.75Hz का मजबूत हाई-पास फ़िल्टर: 
        # यह 'साँस' (0.25Hz) को पूरी तरह खत्म कर देगा ताकि हरी लाइन बेकाबू न हो।
        self.sos_math = signal.butter(2, 0.75, btype='highpass', fs=self.fs, output='sos')
        self.zi_math = signal.sosfilt_zi(self.sos_math)

        # ==========================================================
        # 3. LEAKY INTEGRATORS (गति और विस्थापन)
        # ==========================================================
        self.leak_v = 0.985
        self.b_v = [self.dt]
        self.a_v = [1.0, -self.leak_v]
        self.zi_v = signal.lfilter_zi(self.b_v, self.a_v)

        self.leak_d = 0.97
        self.b_d = [self.dt]
        self.a_d = [1.0, -self.leak_d]
        self.zi_d = signal.lfilter_zi(self.b_d, self.a_d)

        # ==========================================================
        # 4. STABILIZERS (सेंटर लॉक)
        # ==========================================================
        # दोनों इंटीग्रेशन के बाद 0.5Hz का स्टेबलाइजर ताकि वे 0 पर टिके रहें।
        self.sos_stab = signal.butter(1, 0.5, btype='highpass', fs=self.fs, output='sos')
        self.zi_sv = signal.sosfilt_zi(self.sos_stab)
        self.zi_sd = signal.sosfilt_zi(self.sos_stab)

        self.is_first_batch = True

    def process_batch(self, raw_batch):
        # शुरुआत में 2048 DC Offset को सेट करें
        if self.is_first_batch:
            self.envelope_baseline = raw_batch[0]
            self.zi_math = self.zi_math * raw_batch[0]
            self.is_first_batch = False

        # ==========================================================
        # PATH 1: DISPLAY PATH (पीले ग्राफ़ के लिए 100% फ्लैट बेसलाइन)
        # ==========================================================
        raw_display = np.zeros_like(raw_batch)
        for i in range(len(raw_batch)):
            val = raw_batch[i]
            
            # अगर पल्स नीचे आती है, तो बेसलाइन तुरंत नीचे आ जाएगी
            if val < self.envelope_baseline:
                self.envelope_baseline = val
            else:
                # Slew Rate (0.04): यह साँस के उठने की स्पीड (0.031) से थोड़ा तेज़ है, 
                # इसलिए यह साँस को काट देगा, लेकिन पल्स को बिल्कुल नहीं बिगाड़ेगा!
                self.envelope_baseline += 0.04 
            
            # असली पल्स = कच्चा डेटा - स्मार्ट बेसलाइन
            raw_display[i] = val - self.envelope_baseline

        # ==========================================================
        # PATH 2: MATH PATH (साँस को मारकर इंटीग्रेशन करना)
        # ==========================================================
        # साँस और DC दोनों को पूरी तरह साफ करें
        raw_math, self.zi_math = signal.sosfilt(self.sos_math, raw_batch, zi=self.zi_math)

        # ==========================================================
        # INTEGRATION (Velocity & Displacement)
        # ==========================================================
        # Velocity (First Integration)
        vel_int, self.zi_v = signal.lfilter(self.b_v, self.a_v, raw_math, zi=self.zi_v)
        velocity, self.zi_sv = signal.sosfilt(self.sos_stab, vel_int, zi=self.zi_sv)

        # Displacement (Second Integration)
        disp_int, self.zi_d = signal.lfilter(self.b_d, self.a_d, velocity, zi=self.zi_d)
        displacement, self.zi_sd = signal.sosfilt(self.sos_stab, disp_int, zi=self.zi_sd)

        # विस्थापन को सीधा करें (Inversion fix)
        displacement = -displacement

        return {
            'raw_filtered': raw_display,  
            'velocity': velocity,
            'displacement': displacement
        }

    def reset_state(self):
        self.envelope_baseline = 0.0
        self.zi_math = signal.sosfilt_zi(self.sos_math)
        self.zi_v = signal.lfilter_zi(self.b_v, self.a_v)
        self.zi_d = signal.lfilter_zi(self.b_d, self.a_d)
        self.zi_sv = signal.sosfilt_zi(self.sos_stab)
        self.zi_sd = signal.sosfilt_zi(self.sos_stab)
        self.is_first_batch = True
        print("DSP State Reset - Ready for new signal")

    def get_filter_info(self):
        return {
            'sampling_rate': self.fs,
            'raw_display': 'Morphological Envelope Tracker',
            'integration': 'Leaky Integrator + 0.75Hz Respiration Killer'
        }
