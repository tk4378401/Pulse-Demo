"""
nadi_generator.py - आयुर्वेदिक नाड़ी परीक्षण के लिए सिंथेटिक पल्स वेव जनरेटर
Synthetic Pulse Wave Generator for Ayurvedic Nadi Pariksha

यह मॉड्यूल 1000Hz सैम्पलिंग रेट पर आर्टिरियल पल्स वेव बनाता है जिसमें
त्रिदोष (वात, पित्त, कफ) के लक्षण स्पष्ट रूप से दिखाई देते हैं।
"""

import numpy as np              # न्यूमेरिकल कंप्यूटेशन के लिए (Numerical computation)
import scipy.signal as signal   # सिग्नल प्रोसेसिंग और वेव शेपिंग के लिए (Signal processing & wave shaping)
import threading                # बैकग्राउंड थ्रेड में डेटा जनरेट करने के लिए (Background data generation)
import queue                    # थ्रेड-सेफ कतार जहाँ डेटा स्टोर होगा (Thread-safe data storage)
import time                     # टाइमिंग और डिले कंट्रोल के लिए (Timing and delay control)


class VirtualSensor(threading.Thread):
    """
    वर्चुअल सेंसर क्लास - एक थ्रेड जो लगातार आयुर्वेदिक पल्स वेव डेटा जनरेट करता है
    
    Virtual Sensor Class - A thread that continuously generates Ayurvedic pulse wave data
    
    यह क्लास threading.Thread को इनहेरिट करती है ताकि यह मेन प्रोग्राम के साथ
    बैकग्राउंड में चल सके बिना मुख्य प्रोग्राम को ब्लॉक किए।
    
    Inherits from threading.Thread to run in background without blocking main program
    """
    
    def __init__(self, sampling_rate=1000, batch_size=50, vata_strength=0.4, pitta_strength=0.3, kapha_strength=0.3):
        """
        वर्चुअल सेंसर का कंस्ट्रक्टर -初始化 सेंसर पैरामीटर्स
        
        Virtual Sensor Constructor - Initialize sensor parameters
        
        Parameters:
        -----------
        sampling_rate : int, default=1000
            सैम्पलिंग रेट हर्ट्ज में (Sampling rate in Hertz)
            1000Hz = प्रति सेकंड 1000 सैंपल्स (1000 samples per second)
            
        batch_size : int, default=50
            हर बैच में सैंपल्स की संख्या (Number of samples per batch)
            50 सैंपल्स @ 1000Hz = 50ms का डेटा (50ms worth of data)
            
        vata_strength : float, default=0.4
            वात दोष की तीव्रता (Intensity of Vata dosha)
            रेंज: 0.0 से 1.0 (Range: 0.0 to 1.0)
            उच्च मान = तेज़ चढ़ाई (Higher value = steeper rise)
            
        pitta_strength : float, default=0.3
            पित्त दोष की तीव्रता (Intensity of Pitta dosha)
            रेंज: 0.0 से 1.0 (Range: 0.0 to 1.0)
            उच्च मान = गहरा डिक्रोटिक नॉच (Higher value = deeper dicrotic notch)
            
        kapha_strength : float, default=0.3
            कफ दोष की तीव्रता (Intensity of Kapha dosha)
            रेंज: 0.0 से 1.0 (Range: 0.0 to 1.0)
            उच्च मान = चौड़ा बेस (Higher value = wider base)
        """
        
        # पेरेंट क्लास (threading.Thread) को इनिशियलाइज करें
        # Initialize parent class (threading.Thread)
        super().__init__()
        
        # सैंपलिंग रेट स्टोर करें - 1000Hz का मतलब 1ms में एक सैंपल
        # Store sampling rate - 1000Hz means one sample every 1ms
        self.sampling_rate = sampling_rate
        
        # बैच साइज स्टोर करें - 50 सैंपल्स = 50ms का डेटा
        # Store batch size - 50 samples = 50ms of data
        self.batch_size = batch_size
        
        # त्रिदोष स्ट्रेंथ स्टोर करें - ये आयुर्वेदिक पल्स की प्रकृति तय करते हैं
        # Store tridosha strengths - these determine Ayurvedic pulse nature
        self.vata_strength = vata_strength      # वात: वेव की चढ़ाई (Wave ascent)
        self.pitta_strength = pitta_strength    # पित्त: डिक्रोटिक नॉच (Dicrotic notch)
        self.kapha_strength = kapha_strength    # कफ: वेव का बेस (Wave base)
        
        # थ्रेड स्टॉप फ्लैग - जब True होगा तो थ्रेड रुक जाएगा
        # Thread stop flag - when True, thread will stop
        self._stop_event = threading.Event()
        
        # थ्रेड-सेफ कतार जहाँ जनरेटेड डेटा स्टोर होगा
        # Thread-safe queue where generated data will be stored
        self.data_queue = queue.Queue()
        
        # समय ट्रैकिंग वेरिएबल - सिमुलेशन टाइम
        # Time tracking variable - simulation time
        self.current_time = 0.0
        
        # हृदय गति - सामान्य आराम दर (60-70 BPM)
        # Heart rate - normal resting rate (60-70 BPM)
        self.heart_rate = 65  # beats per minute
        
        # हृदय गति को हर्ट्ज में कन्वर्ट करें (चक्र प्रति सेकंड)
        # Convert heart rate to Hertz (cycles per second)
        self.heart_frequency = self.heart_rate / 60.0  # Hz में फ्रीक्वेंसी (Frequency in Hz)
        
        # एक पूर्ण हृदय चक्र की अवधि (सेकंड में)
        # Duration of one complete heart cycle (in seconds)
        self.cardiac_cycle_duration = 1.0 / self.heart_frequency
        
        # पिछला सैंपल वैल्यू - स्मूथ कनेक्टिविटी के लिए
        # Previous sample value - for smooth connectivity
        self.previous_value = 0.0
    
    def generate_ayurvedic_pulse_wave(self, time_array):
        """
        आयुर्वेदिक पल्स वेव जनरेट करें - त्रिदोष सिद्धांत पर आधारित
        
        Generate Ayurvedic Pulse Wave - Based on Tridosha Theory
        
        यह फंक्शन एक आर्टिरियल पल्स वेव बनाता है जिसमें तीन भाग होते हैं:
        This function creates an arterial pulse wave with three components:
        
        1. वत (Vata) - 'Anacrotic limb': वेव का अचानक और तेज़ ऊपर उठना
           Sudden and rapid upward rise of the wave
           
        2. पित्त (Pitta) - 'Dicrotic Notch': नीचे गिरते समय एक स्पष्ट गड्ढा
           A clear dip and slight bounce while falling
           
        3. कफ (Kapha) - 'Diastolic runoff': बेस का चौड़ा होना
           Wide base and slow gradual fall
        
        Parameters:
        -----------
        time_array : numpy array
            समय के मान जिनके लिए पल्स वेव कैलकुलेट करनी है
            Time values for which to calculate pulse wave
            
        Returns:
        --------
        pulse_wave : numpy array
            जनरेटेड आयुर्वेदिक पल्स वेव के एम्पलीट्यूड वैल्यूज
            Amplitude values of generated Ayurvedic pulse wave
        """
        
        # समय सरणी को नॉर्मलाइज करें - कार्डियक साइकिल के अंदर 0 से 1 तक
        # Normalize time array - from 0 to 1 within cardiac cycle
        # modulo ऑपरेटर का उपयोग करके हम लगातार वेव्स बना सकते हैं
        # Using modulo operator allows us to create continuous waves
        normalized_time = np.mod(time_array, self.cardiac_cycle_duration) / self.cardiac_cycle_duration
        
        # ====================
        # 1. वत (VATA) घटक - Anacrotic Limb (तेज़ चढ़ाई)
        #    VATA Component - Rapid Upward Rise
        # ====================
        
        # सिग्मॉइड फंक्शन का उपयोग करके अचानक चढ़ाई बनाएं
        # Create sudden rise using sigmoid function
        # tanh (hyperbolic tangent) एक S-शेप्ड कर्व बनाता है
        # tanh creates an S-shaped curve
        
        # vata_strength जितना अधिक होगा, चढ़ाई उतनी ही तेज़ होगी
        # Higher vata_strength = steeper rise
        vata_component = 0.5 * (1 + np.tanh(normalized_time * 20 * (1 - self.vata_strength)))
        
        # वत वेव को धीरे-धीरे कम करें (decay) ताकि यह प्राकृतिक लगे
        # Gradually decay vata wave to make it look natural
        # exp (exponential) फंक्शन धीरे-धीरे कम होता है
        # exp function gradually decreases
        vata_decay = np.exp(-normalized_time * 3)  # 3 = decay rate
        vata_wave = vata_component * vata_decay
        
        # ====================
        # 2. पित्त (PITTA) घटक - Dicrotic Notch (गड्ढा और उछाल)
        #    PITTA Component - Dip and Bounce
        # ====================
        
        # डिक्रोटिक नॉच वेव के नीचे गिरने के समय बनता है (लगभग 60-70% पर)
        # Dicrotic notch forms during wave descent (around 60-70%)
        
        # गॉसियन फंक्शन का उपयोग करके नॉच बनाएं
        # Create notch using Gaussian function
        
        # नॉच की स्थिति - वेव के 65% पर (सिस्टोलिक के बाद)
        # Notch position - at 65% of wave (after systolic peak)
        notch_position = 0.65
        
        # नॉच की चौड़ाई - पित्त स्ट्रेंथ पर निर्भर
        # Notch width - depends on pitta strength
        notch_width = 0.15 * (1 - self.pitta_strength)  # अधिक पित्त = संकरा नॉच
        
        # गॉसियन बंप बनाएं (नकारात्मक для गड्ढा)
        # Create Gaussian bump (negative for dip)
        pitta_component = -np.exp(-((normalized_time - notch_position) ** 2) / (2 * notch_width ** 2))
        
        # नॉच के बाद हल्का उछाल (Tidal wave)
        # Slight bounce after notch (Tidal wave)
        tidal_position = 0.75  # नॉच के बाद
        tidal_width = 0.1
        tidal_bounce = 0.3 * np.exp(-((normalized_time - tidal_position) ** 2) / (2 * tidal_width ** 2))
        
        # पित्त वेव = नॉच + उछाल
        # Pitta wave = notch + bounce
        pitta_wave = pitta_component + tidal_bounce
        
        # ====================
        # 3. कफ (KAPHA) घटक - Diastolic Runoff (चौड़ा बेस)
        #    KAPHA Component - Wide Base
        # ====================
        
        # कफ वेव धीमी और चौड़ी होती है - पूरे कार्डियक साइकिल में फैली हुई
        # Kapha wave is slow and wide - spread across cardiac cycle
        
        # एक्सपोनेंशियल decay फंक्शन जो धीरे-धीरे कम होता है
        # Exponential decay function that decreases slowly
        
        # kapha_strength जितना अधिक होगा, बेस उतना ही चौड़ा होगा
        # Higher kapha_strength = wider base
        kapha_decay_rate = 2 * (1 - self.kapha_strength)  # कम दर = चौड़ा बेस
        kapha_wave = np.exp(-normalized_time * kapha_decay_rate)
        
        # ====================
        # तीनों दोषों को मिलाएं - त्रिदोष संतुलन
        # Combine all three doshas - Tridosha Balance
        # ====================
        
        # प्रत्येक दोष को उसकी स्ट्रेंथ के अनुसार वजन दें
        # Weight each dosha according to its strength
        combined_wave = (
            self.vata_strength * vata_wave +      # वत का योगदान (Vata contribution)
            self.pitta_strength * pitta_wave +    # पित्त का योगदान (Pitta contribution)
            self.kapha_strength * kapha_wave      # कफ का योगदान (Kapha contribution)
        )
        
        # वेव को नॉर्मलाइज करें (0.2 से 1.0 के बीच) ताकि यह रियलिस्टिक लगे
        # Normalize wave (between 0.2 and 1.0) to make it realistic
        # min-max scaling का उपयोग करें
        # Use min-max scaling
        wave_min = np.min(combined_wave)
        wave_max = np.max(combined_wave)
        
        # नॉर्मलाइजेशन फॉर्मूला: (value - min) / (max - min)
        # Normalization formula: (value - min) / (max - min)
        if wave_max > wave_min:  # डिवाइड बाय जीरो एरर से बचने के लिए
            # To avoid divide by zero error
            normalized_pulse = 0.2 + 0.8 * (combined_wave - wave_min) / (wave_max - wave_min)
        else:
            # अगर सभी वैल्यूज समान हैं (edge case)
            # If all values are same (edge case)
            normalized_pulse = np.ones_like(combined_wave) * 0.6
        
        return normalized_pulse
    
    def generate_batch(self):
        """
        50 सैंपल्स का एक बैच जनरेट करें
        
        Generate a batch of 50 samples
        
        यह मेथड 1000Hz पर 50 सैंपल्स जनरेट करता है जो 50ms के समय अंतराल के बराबर है।
        This method generates 50 samples at 1000Hz which equals 50ms time interval.
        
        Returns:
        --------
        batch_data : numpy array
            50 सैंपल्स का numpy array containing pulse wave amplitudes
            50 samples का numpy array जिसमें पल्स वेव के एम्पलीट्यूड हैं
        """
        
        # समय अंतराल की गणना - 1000Hz पर एक सैंपल = 1ms
        # Calculate time interval - one sample at 1000Hz = 1ms
        dt = 1.0 / self.sampling_rate  # 0.001 सेकंड = 1 मिलीसेकंड
        
        # 50 सैंपल्स के लिए समय सरणी बनाएं
        # Create time array for 50 samples
        # np.arange(start, stop, step) - start से stop तक step अंतराल पर वैल्यूज
        # np.arange - values from start to stop with step interval
        batch_time = np.arange(0, self.batch_size * dt, dt)
        
        # वर्तमान समय में ऑफसेट जोड़ें ताकि लगातार वेव बने
        # Add offset to current time for continuous wave
        actual_time = batch_time + self.current_time
        
        # आयुर्वेदिक पल्स वेव जनरेट करें
        # Generate Ayurvedic pulse wave
        batch_data = self.generate_ayurvedic_pulse_wave(actual_time)
        
        # पिछले वैल्यू के साथ स्मूथ कनेक्टिविटी सुनिश्चित करें
        # Ensure smooth connectivity with previous value
        
        # पहला सैंपल पिछले बैच के आखिरी सैंपल से मेल खाना चाहिए
        # First sample should match last sample of previous batch
        
        # लि니어 इंटरपोलेशन का उपयोग करके स्मूथ ट्रांजिशन बनाएं
        # Create smooth transition using linear interpolation
        
        # पहले 5 सैंपल्स को धीरे-धीरे एडजस्ट करें
        # Gradually adjust first 5 samples
        transition_samples = 5
        for i in range(transition_samples):
            # interpolation_factor: 0 से 1 तक (पहले सैंपल = 0, 5वें सैंपल = 1)
            # interpolation_factor: 0 to 1 (1st sample = 0, 5th sample = 1)
            alpha = i / transition_samples
            
            # लीनियर इंटरपोलेशन: new_value = old + alpha * (new - old)
            # Linear interpolation: new_value = old + alpha * (new - old)
            batch_data[i] = (1 - alpha) * self.previous_value + alpha * batch_data[i]
        
        # अपडेट previous_value untuk अगले बैच के लिए
        # Update previous_value for next batch
        self.previous_value = batch_data[-1]  # आखिरी सैंपल स्टोर करें
        
        # वर्तमान समय को आगे बढ़ाएं - अगले बैच के लिए
        # Advance current time - for next batch
        self.current_time += self.batch_size * dt
        
        # यदि समय कार्डियक साइकिल से अधिक हो जाता है, तो रीसेट करें
        # If time exceeds cardiac cycle, reset
        # यह सुनिश्चित करता है कि संख्याएं बहुत बड़ी न हो जाएं
        # This ensures numbers don't become too large
        if self.current_time > 10 * self.cardiac_cycle_duration:  # 10 साइकिल के बाद रीसेट
            # Reset after 10 cycles
            self.current_time = 0.0
        
        return batch_data
    
    def run(self):
        """
        थ्रेड का मुख्य फंक्शन - लगातार डेटा जनरेट करता है
        
        Main thread function - continuously generates data
        
        यह मेथड तब चलता है जब आप start() कोल करते हैं।
        This method runs when you call start().
        
        यह एक लूप है जो तब तक चलता रहता है जब तक _stop_event सेट नहीं होता।
        This is a loop that runs until _stop_event is set.
        """
        
        # बैच अवधि की गणना - 50 सैंपल्स @ 1000Hz = 50ms
        # Calculate batch duration - 50 samples @ 1000Hz = 50ms
        batch_duration = self.batch_size / self.sampling_rate  # 0.05 सेकंड
        
        # लूप तब तक चलता रहेगा जब तक स्टॉप इवेंट सेट नहीं होता
        # Loop continues until stop event is set
        while not self._stop_event.is_set():
            
            # एक बैच डेटा जनरेट करें (50 सैंपल्स)
            # Generate one batch of data (50 samples)
            batch_data = self.generate_batch()
            
            # डेटा को थ्रेड-सेफ कतार में डालें
            # Put data into thread-safe queue
            
            # queue.put() थ्रेड-सेफ है - मल्टीपल थ्रेड्स से सुरक्षित
            # queue.put() is thread-safe - safe from multiple threads
            try:
                self.data_queue.put(batch_data, block=False)
            except queue.Full:
                # यदि कतार भरी हुई है (edge case)
                # If queue is full (edge case)
                pass
            
            # 50ms वेट करें - रियल-टाइम सैंपलिंग सिमुलेट करें
            # Wait 50ms - simulate real-time sampling
            
            # time.sleep() सेकंड में वेट करता है
            # time.sleep() waits in seconds
            time.sleep(batch_duration)
        
        # जब लूप समाप्त होता है (स्टॉप के बाद)
        # When loop ends (after stop)
        print("Virtual Sensor stopped - वर्चुअल सेंसर रुक गया")
    
    def start(self):
        """
        वर्चुअल सेंसर को शुरू करें - बैकग्राउंड थ्रेड स्टार्ट करें
        
        Start virtual sensor - start background thread
        
        यह मेथड पेरेंट क्लास के start() को ओवरराइड करता है।
        This method overrides parent class's start().
        
        Note:
        -----
        इसमेथड को कॉल करने के बाद, सेंसर बैकग्राउंड में डेटा जनरेट करना शुरू कर देता है।
        After calling this method, sensor starts generating data in background.
        """
        
        # पेरेंट क्लास का start() मेथड कॉल करें
        # Call parent class's start() method
        super().start()
        
        # स्टार्टअप मैसेज प्रिंट करें
        # Print startup message
        print(f"Virtual Sensor started - वर्चुअल सेंसर शुरू हो गया")
        print(f"Sampling Rate: {self.sampling_rate}Hz ({self.sampling_rate/1000:.1f}kHz)")
        print(f"Batch Size: {self.batch_size} samples ({self.batch_size/self.sampling_rate*1000:.1f}ms)")
        print(f"Heart Rate: {self.heart_rate} BPM")
        print(f"Tridosha - Vata: {self.vata_strength:.2f}, Pitta: {self.pitta_strength:.2f}, Kapha: {self.kapha_strength:.2f}")
    
    def stop(self):
        """
        वर्चुअल सेंसर को रोकें - थ्रेड को ग्रेशफुली टर्मिनेट करें
        
        Stop virtual sensor - gracefully terminate thread
        
        यह मेथड _stop_event को सेट करता है जो run() लूप को रोक देता है।
        This method sets _stop_event which stops the run() loop.
        """
        
        # स्टॉप इवेंट सेट करें - यह run() लूप को सिग्नल भेजता है
        # Set stop event - this signals the run() loop
        
        # threading.Event.set() इवेंट फ्लैग को True बना देता है
        # threading.Event.set() makes event flag True
        self._stop_event.set()
        
        # थ्रेड के समाप्त होने का इंतजार करें (join)
        # Wait for thread to finish (join)
        
        # join() ब्लॉकिंग कॉल है - थ्रेड खत्म होने तक वेट करता है
        # join() is blocking call - waits until thread finishes
        self.join(timeout=1.0)  # 1 सेकंड का टाइमआउट
        
        # स्टॉप मैसेज प्रिंट करें
        # Print stop message
        print("Virtual Sensor stopped successfully - वर्चुअल सेंसर सफलतापूर्वक रुक गया")
    
    def get_latest_batch(self, timeout=1.0):
        """
        सबसे हालिया बैच डेटा प्राप्त करें
        
        Get latest batch data
        
        Parameters:
        -----------
        timeout : float, default=1.0
            अधिकतम प्रतीक्षा समय सेकंड में
            Maximum wait time in seconds
            
        Returns:
        --------
        batch : numpy array or None
            50 सैंपल्स का numpy array, या None यदि टाइमआउट
            50 samples numpy array, or None if timeout
        """
        
        # कतार से डेटा प्राप्त करें
        # Get data from queue
        
        # queue.get() ब्लॉकिंग कॉल है - डेटा आने तक वेट करता है
        # queue.get() is blocking call - waits until data arrives
        try:
            batch = self.data_queue.get(timeout=timeout)
            return batch
        except queue.Empty:
            # यदि टाइमआउट हो गया (कोई डेटा नहीं आया)
            # If timeout occurred (no data arrived)
            return None
    
    def get_queue_size(self):
        """
        कतार में वर्तमान बैच की संख्या प्राप्त करें
        
        Get current number of batches in queue
        
        Returns:
        --------
        size : int
            कतार में उपलब्ध बैच की संख्या
            Number of batches available in queue
        """
        
        # queue.qsize() कतार में आइटम की संख्या बताता है
        # queue.qsize() returns number of items in queue
        return self.data_queue.qsize()


# ============================================================================
# उदाहरण उपयोग (Example Usage) - टेस्टिंग के लिए
# ============================================================================

if __name__ == "__main__":
    """
    मेन एक्जीक्यूशन ब्लॉक - केवल तब चलता है जब यह फ़ाइल सीधे चलाई जाती है
    
    Main execution block - only runs when this file is executed directly
    
    यह उदाहरण कोड दिखाता है कि VirtualSensor का उपयोग कैसे करें।
    This example code shows how to use VirtualSensor.
    """
    
    print("=" * 70)
    print("Ayurvedic Nadi Pariksha DSP Sandbox - नाड़ी परीक्षण डीएसपी सैंडबॉक्स")
    print("=" * 70)
    print()
    
    # वर्चुअल सेंसर बनाएं - त्रिदोष के साथ सामान्य पल्स
    # Create virtual sensor - normal pulse with tridosha
    
    # Vata=0.4 (तेज़ चढ़ाई), Pitta=0.3 (स्पष्ट नॉच), Kapha=0.3 (चौड़ा बेस)
    # Vata=0.4 (rapid rise), Pitta=0.3 (clear notch), Kapha=0.3 (wide base)
    sensor = VirtualSensor(
        sampling_rate=1000,           # 1000Hz सैम्पलिंग रेट
        batch_size=50,                # 50 सैंपल्स प्रति बैच
        vata_strength=0.4,            # वात: 40% तीव्रता
        pitta_strength=0.3,           # पित्त: 30% तीव्रता
        kapha_strength=0.3            # कफ: 30% तीव्रता
    )
    
    # सेंसर शुरू करें - बैकग्राउंड थ्रेड स्टार्ट
    # Start sensor - start background thread
    sensor.start()
    
    print("\nडेटा संग्रह शुरू हो रहा है... Starting data collection...\n")
    
    # 5 सेकंड के लिए डेटा एकत्र करें
    # Collect data for 5 seconds
    
    batches_collected = 0  # एकत्रित बैच की संख्या काउंटर
    max_batches = 100      # अधिकतम बैच (5 सेकंड = 100 बैच @ 50ms)
    
    try:
        # लूप 100 बैच तक चलेगा
        # Loop will run for 100 batches
        while batches_collected < max_batches:
            
            # कतार से बैच प्राप्त करें
            # Get batch from queue
            
            # timeout=0.1 का मतलब 100ms तक वेट करें
            # timeout=0.1 means wait up to 100ms
            batch = sensor.get_latest_batch(timeout=0.1)
            
            # यदि डेटा मिला है
            # If data received
            if batch is not None:
                # बैच साइज प्रिंट करें
                # Print batch size
                
                # len(batch) = 50 सैंपल्स होने चाहिए
                # len(batch) should be 50 samples
                print(f"Batch {batches_collected + 1}: Shape={batch.shape}, "
                      f"Min={np.min(batch):.3f}, Max={np.max(batch):.3f}, "
                      f"Queue Size={sensor.get_queue_size()}")
                
                # काउंटर बढ़ाएं
                # Increment counter
                batches_collected += 1
                
                # हर 10 बैच के बाद थोड़ा वेट करें (प्रिंट को पढ़ने के लिए)
                # Wait slightly after every 10 batches (to read prints)
                if batches_collected % 10 == 0:
                    time.sleep(0.1)
            
            # यदि उपयोगकर्ता Ctrl+C दबाता है
            # If user presses Ctrl+C
    except KeyboardInterrupt:
        print("\n\nउपयोगकर्ता द्वारा रोका गया - Stopped by user")
    
    # सेंसर रोकें - थ्रेड ग्रेशफुली टर्मिनेट
    # Stop sensor - gracefully terminate thread
    print("\nसेंसर रोक रहा है... Stopping sensor...")
    sensor.stop()
    
    print("\n" + "=" * 70)
    print("परीक्षण पूर्ण - Test Complete")
    print("=" * 70)
    print(f"कुल बैच संग्रहीत: {batches_collected}")
    print(f"अंतिम कतार आकार: {sensor.get_queue_size()}")
    print()
