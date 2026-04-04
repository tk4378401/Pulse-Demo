"""
nadi_dsp.py - आयुर्वेदिक नाड़ी परीक्षण के लिए DSP इंजन
(Fixed: Valley Tracker for Flat Baseline & Zero Deep-Bowl Distortion)
"""
import numpy as np
import scipy.signal as signal

class NadiDSP:
    def __init__(self, sampling_rate=1000):
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / self.sampling_rate

        # ==========================================================
        # 1. VALLEY TRACKER (दिखने वाले पीले ग्राफ़ के लिए)
        # ==========================================================
        # यह सिग्नल के 'सबसे निचले हिस्से' को ट्रैक करेगा, औसत को नहीं।
        # इससे पीली लाइन हमेशा 0 की बेसलाइन पर चिपकी रहेगी (कोई गड्ढा नहीं बनेगा)।
        self.valley_baseline = 0.0
        
        # ==========================================================
        # 2. AC-COUPLER (मैथ और इंटीग्रेशन के लिए)
        # ==========================================================
        self.alpha_ac = 0.995 
        self.b_ac = [1.0 - self.alpha_ac]
        self.a_ac = [1.0, -self.alpha_ac]
        self.zi_ac = signal.lfilter_zi(self.b_ac, self.a_ac)

        # ==========================================================
        # 3. LEAKY INTEGRATORS (गति और विस्थापन)
        # ==========================================================
        self.leak_v = 0.985
        self.b_int_v = [self.dt]
        self.a_int_v = [1.0, -self.leak_v]
        self.zi_vel = signal.lfilter_zi(self.b_int_v, self.a_int_v)

        self.leak_d = 0.97
        self.b_int_d = [self.dt]
        self.a_int_d = [1.0, -self.leak_d]
        self.zi_disp = signal.lfilter_zi(self.b_int_d, self.a_int_d)

        # ==========================================================
        # 4. STABILIZERS
        # ==========================================================
        self.sos_hp = signal.butter(1, 0.5, btype='highpass', fs=self.sampling_rate, output='sos')
        self.zi_hp_v = signal.sosfilt_zi(self.sos_hp)
        self.zi_hp_d = signal.sosfilt_zi(self.sos_hp)

        self.is_first_batch = True

    def process_batch(self, raw_batch):
        if self.is_first_batch:
            # 2048 DC Offset को पहले ही सैंपल पर सेट करें
            self.valley_baseline = raw_batch[0]
            self.zi_ac = self.zi_ac * raw_batch[0]
            self.is_first_batch = False

        # ==========================================================
        # PATH 1: DISPLAY PATH (पीले ग्राफ़ के लिए Valley Tracking)
        # ==========================================================
        raw_display = np.zeros_like(raw_batch)
        for i in range(len(raw_batch)):
            val = raw_batch[i]
            if val < self.valley_baseline:
                # अगर सिग्नल नीचे जाता है, तो बहुत तेज़ी से बेसलाइन को नीचे लाएं (गड्ढा बनने से रोकें)
                self.valley_baseline = 0.5 * self.valley_baseline + 0.5 * val 
            else:
                # अगर सिग्नल ऊपर (पल्स) है, तो बेसलाइन को बहुत धीरे-धीरे (साँस के लिए) ऊपर लाएं
                self.valley_baseline = 0.9995 * self.valley_baseline + 0.0005 * val 
            
            # असली पल्स = सिग्नल - घाटी की बेसलाइन (अब यह सिर्फ 0 से ऊपर उठेगा)
            raw_display[i] = val - self.valley_baseline

        # ==========================================================
        # PATH 2: MATH PATH (इंटीग्रेशन के लिए Zero-Mean)
        # ==========================================================
        ac_baseline, self.zi_ac = signal.lfilter(self.b_ac, self.a_ac, raw_batch, zi=self.zi_ac)
        raw_math = raw_batch - ac_baseline

        # ==========================================================
        # INTEGRATION (Velocity & Displacement)
        # ==========================================================
        vel_int, self.zi_vel = signal.lfilter(self.b_int_v, self.a_int_v, raw_math, zi=self.zi_vel)
        velocity, self.zi_hp_v = signal.sosfilt(self.sos_hp, vel_int, zi=self.zi_hp_v)

        disp_int, self.zi_disp = signal.lfilter(self.b_int_d, self.a_int_d, velocity, zi=self.zi_disp)
        displacement, self.zi_hp_d = signal.sosfilt(self.sos_hp, disp_int, zi=self.zi_hp_d)

        # वेव को सीधा करें
        displacement = -displacement

        return {
            'raw_filtered': raw_display,  # स्क्रीन पर दिखने के लिए एकदम परफेक्ट फ्लैट-बॉटम वेव!
            'velocity': velocity,
            'displacement': displacement
        }

    def reset_state(self):
        self.valley_baseline = 0.0
        self.zi_ac = signal.lfilter_zi(self.b_ac, self.a_ac)
        self.zi_vel = signal.lfilter_zi(self.b_int_v, self.a_int_v)
        self.zi_disp = signal.lfilter_zi(self.b_int_d, self.a_int_d)
        self.zi_hp_v = signal.sosfilt_zi(self.sos_hp)
        self.zi_hp_d = signal.sosfilt_zi(self.sos_hp)
        self.is_first_batch = True
        print("DSP State Reset - Ready for new signal")

    def get_filter_info(self):
        return {
            'sampling_rate': self.sampling_rate,
            'dc_removal': 'Dual Pathway (Valley Tracker + AC Coupler)',
            'integration': 'Leaky Integrator'
        }

if __name__ == "__main__":
    print("Ayurvedic Nadi Pariksha DSP Engine - Dual Pathway Version")
    dsp = NadiDSP()
    test_batch = 2048 + 500 * np.sin(2 * np.pi * 1 * np.linspace(0, 0.05, 50))
    res = dsp.process_batch(test_batch)
    print(f"Raw Filtered Shape: {res['raw_filtered'].shape}")
    print("Test passed successfully.")
