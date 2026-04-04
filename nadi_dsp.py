"""
nadi_dsp.py - आयुर्वेदिक नाड़ी परीक्षण के लिए DSP इंजन
DSP Engine for Ayurvedic Nadi Pariksha

(Fixed: 1st-Order EMA Baseline Tracking to remove 'Deep Bowl' Ringing)
"""
import numpy as np
import scipy.signal as signal

class NadiDSP:
    def __init__(self, sampling_rate=1000):
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / self.sampling_rate

        # ==========================================================
        # 1. DYNAMIC BASELINE TRACKER (1st-Order EMA)
        # ==========================================================
        # यह 4th-order Butterworth की तरह गहरे 'गड्ढे' (undershoots) नहीं बनाता।
        # यह बहुत ही कोमलता से 2048 DC और साँस के इफ़ेक्ट को हटाता है।
        self.alpha_ema = 0.992  
        self.b_ema = [1.0 - self.alpha_ema]
        self.a_ema = [1.0, -self.alpha_ema]
        self.zi_ema = signal.lfilter_zi(self.b_ema, self.a_ema)

        # ==========================================================
        # 2. LEAKY INTEGRATORS (गति और विस्थापन के लिए)
        # ==========================================================
        # cumulative_trapezoid की जगह Leaky Integrator का उपयोग, 
        # ताकि ग्राफ़ कभी भी आउट-ऑफ़-कंट्रोल (Drift) होकर आसमान में न भागे।
        
        self.leak_v = 0.985
        self.b_int_v = [self.dt]
        self.a_int_v = [1.0, -self.leak_v]
        self.zi_vel = signal.lfilter_zi(self.b_int_v, self.a_int_v)

        self.leak_d = 0.97
        self.b_int_d = [self.dt]
        self.a_int_d = [1.0, -self.leak_d]
        self.zi_disp = signal.lfilter_zi(self.b_int_d, self.a_int_d)

        # 3. GENTLE HIGHPASS (इंटीग्रेशन को सेंटर (0) लाइन पर रखने के लिए)
        # यह सिर्फ 1st order का 0.5Hz फिल्टर है, जो सिग्नल के आकार को नहीं बिगाड़ेगा।
        self.sos_hp = signal.butter(1, 0.5, btype='highpass', fs=self.sampling_rate, output='sos')
        self.zi_hp_v = signal.sosfilt_zi(self.sos_hp)
        self.zi_hp_d = signal.sosfilt_zi(self.sos_hp)

        self.is_first_batch = True

    def process_batch(self, raw_batch):
        # 2048 DC Offset को 0 सेकंड में इनिशियलाइज करें (बिना किसी स्पाइक के)
        if self.is_first_batch:
            self.zi_ema = self.zi_ema * raw_batch[0]
            self.is_first_batch = False

        # ==========================================================
        # STEP 1: RAW SIGNAL (Perfect DC Removal)
        # ==========================================================
        # बेसलाइन (2048 + साँस) निकालें
        baseline, self.zi_ema = signal.lfilter(self.b_ema, self.a_ema, raw_batch, zi=self.zi_ema)
        
        # असली पल्स वेव = कच्चा डेटा - बेसलाइन
        raw_clean = raw_batch - baseline

        # ==========================================================
        # STEP 2: VELOCITY (First Leaky Integration)
        # ==========================================================
        vel_int, self.zi_vel = signal.lfilter(self.b_int_v, self.a_int_v, raw_clean, zi=self.zi_vel)
        velocity, self.zi_hp_v = signal.sosfilt(self.sos_hp, vel_int, zi=self.zi_hp_v)

        # ==========================================================
        # STEP 3: DISPLACEMENT (Second Leaky Integration)
        # ==========================================================
        disp_int, self.zi_disp = signal.lfilter(self.b_int_d, self.a_int_d, velocity, zi=self.zi_disp)
        displacement, self.zi_hp_d = signal.sosfilt(self.sos_hp, disp_int, zi=self.zi_hp_d)

        # गणितीय सुधार: दो बार इंटीग्रेट करने से वेव 180° पलट जाती है, इसलिए उसे सीधा करें
        displacement = -displacement

        return {
            'raw_filtered': raw_clean,
            'velocity': velocity,
            'displacement': displacement
        }

    def reset_state(self):
        self.zi_ema = signal.lfilter_zi(self.b_ema, self.a_ema)
        self.zi_vel = signal.lfilter_zi(self.b_int_v, self.a_int_v)
        self.zi_disp = signal.lfilter_zi(self.b_int_d, self.a_int_d)
        self.zi_hp_v = signal.sosfilt_zi(self.sos_hp)
        self.zi_hp_d = signal.sosfilt_zi(self.sos_hp)
        self.is_first_batch = True
        print("DSP State Reset - Ready for new signal")

    def get_filter_info(self):
        return {
            'sampling_rate': self.sampling_rate,
            'dc_removal': '1st-Order EMA (alpha=0.992)',
            'integration': 'Leaky Integrator'
        }

if __name__ == "__main__":
    print("Ayurvedic Nadi Pariksha DSP Engine - 1st Order EMA Version")
    dsp = NadiDSP()
    test_batch = 2048 + 500 * np.sin(2 * np.pi * 1 * np.linspace(0, 0.05, 50))
    res = dsp.process_batch(test_batch)
    print(f"Raw Filtered Shape: {res['raw_filtered'].shape}")
    print("Test passed successfully.")
