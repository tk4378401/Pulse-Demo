"""
nadi_dsp.py - आयुर्वेदिक नाड़ी परीक्षण के लिए DSP इंजन
DSP Engine for Ayurvedic Nadi Pariksha

यह मॉड्यूल 50-सैंपल बैच प्रोसेस करता है और डबल इंटीग्रेशन के माध्यम से
वेवफॉर्म की वेलोसिटी और डिस्प्लेसमेंट कैलकुलेट करता है।
"""

import numpy as np                      # न्यूमेरिकल कंप्यूटेशन के लिए (Numerical computation)
from scipy import signal                # सिग्नल प्रोसेसिंग और फिल्टर डिज़ाइन के लिए (Signal processing & filter design)
from scipy import integrate             # इंटीग्रेशन (समाकलन) के लिए (Integration calculus)


class NadiDSP:
    """
    नाड़ी DSP क्लास - आयुर्वेदिक पल्स वेव का डिजिटल सिग्नल प्रोसेसिंग
    
    Nadi DSP Class - Digital Signal Processing of Ayurvedic Pulse Wave
    
    यह क्लास लगातार आ रहे 50-सैंपल बैच को प्रोसेस करती है और:
    This class continuously processes 50-sample batches and:
    
    1. Raw Filtered Signal -> मूल सिग्नल को साफ़ करता है
       Raw Filtered Signal -> Cleans the original signal
       
    2. Velocity -> पहला इंटीग्रेशन (गति)
       Velocity -> First integration (speed/rate of change)
       
    3. Displacement -> दूसरा इंटीग्रेशन (विस्थापन)
       Displacement -> Second integration (total displacement)
    """
    
    def __init__(self, sampling_rate=1000, highpass_cutoff=0.1):
        """
        NadiDSP का कंस्ट्रक्टर - फिल्टर और स्टेट वेरिएबल्स को इनिशियलाइज करें
        
        NadiDSP Constructor - Initialize filters and state variables
        
        Parameters:
        -----------
        sampling_rate : int, default=1000
            सैम्पलिंग रेट हर्ट्ज में (Sampling rate in Hertz)
            1000Hz = प्रति सेकंड 1000 सैंपल्स (1000 samples per second)
            
        highpass_cutoff : float, default=0.1
            हाईपास फिल्टर की कटऑफ फ्रीक्वेंसी हर्ट्ज में
            Highpass filter cutoff frequency in Hertz
            0.1Hz = बहुत धीमे ड्रिफ्ट को हटाता है (Removes very slow drift)
        """
        
        # सैंपलिंग रेट स्टोर करें - 1000Hz
        # Store sampling rate - 1000Hz
        self.sampling_rate = sampling_rate
        
        # हाईपास कटऑफ फ्रीक्वेंसी स्टोर करें - 0.1Hz
        # Store highpass cutoff frequency - 0.1Hz
        self.highpass_cutoff = highpass_cutoff
        
        # ====================
        # फिल्टर डिज़ाइन - Butterworth Highpass Filter (SOS format)
        # ====================
        
        # Butterworth फिल्टर ऑर्डर - 4th order
        # Butterworth filter order - 4th order
        
        # उच्च ऑर्डर = तेज़ रोल-ऑफ (अधिक सख्त फिल्टरिंग)
        # Higher order = faster roll-off (stricter filtering)
        filter_order = 4
        
        # Nyquist frequency = sampling_rate / 2
        # Nyquist frequency = सैम्पलिंग रेट / 2
        
        # Nyquist से ऊपर की फ्रीक्वेंसीज एलियासिंग का कारण बनती हैं
        # Frequencies above Nyquist cause aliasing
        nyquist = self.sampling_rate / 2.0  # 1000Hz / 2 = 500Hz
        
        # Normalized frequency = cutoff / nyquist
        # Normalized frequency = कटऑफ / नाइक्विस्ट
        
        # scipy को normalized frequency चाहिए (0 से 1 के बीच)
        # scipy needs normalized frequency (between 0 and 1)
        normalized_cutoff = self.highpass_cutoff / nyquist  # 0.1 / 500 = 0.0002
        
        # Butterworth हाईपास फिल्टर डिज़ाइन करें (SOS format में)
        # Design Butterworth highpass filter (in SOS format)
        
        # SOS (Second-Order Sections) format:
        # - numerical stability के लिए बेहतर (better for numerical stability)
        # - cascaded biquad filters का उपयोग करता है (uses cascaded biquad filters)
        # sos = second-order sections matrix
        self.filter_sos = signal.butter(
            N=filter_order,                    # फिल्टर ऑर्डर (Filter order)
            Wn=normalized_cutoff,              # Normalized cutoff frequency
            btype='high',                      # Highpass filter type
            output='sos'                       # Second-order sections output
        )
        
        # ====================
        # State Management - Initial Conditions (zi)
        # ====================
        
        # zi = initial conditions for filter states
        # zi = फिल्टर स्टेट्स के लिए प्रारंभिक स्थितियां
        
        # यह सुनिश्चित करता है कि बैच के बीच कनेक्टिविटी बनी रहे
        # This ensures connectivity between batches remains intact
        
        # बिना zi के, हर बैच की शुरुआत 0 से होगी और ग्राफ़ टूटेगा
        # Without zi, each batch would start at 0 and graph would break
        
        # सोस फिल्टर के लिए zi की संरचना:
        # Structure of zi for sos filter:
        # - shape: (n_sections, 2) जहाँ n_sections = filter sections की संख्या
        # - shape: (n_sections, 2) where n_sections = number of filter sections
        
        # प्रत्येक सेक्शन के पास 2 state variables होते हैं
        # Each section has 2 state variables
        n_sections = self.filter_sos.shape[0]  # SOS array से sections की संख्या प्राप्त करें
        
        # जीरो इनिशियल कंडीशन - सभी स्टेट्स 0 से शुरू
        # Zero initial conditions - all states start at 0
        
        # np.zeros(shape) = शून्य का array बनाता है
        # np.zeros(shape) = creates array of zeros
        # zi shape must be (n_sections, 2) for sosfilt
        self.zi_raw = np.zeros((n_sections, 2))  # Raw filtered signal के लिए
        self.zi_vel = np.zeros((n_sections, 2))  # Velocity signal के लिए
        self.zi_disp = np.zeros((n_sections, 2))  # Displacement signal के लिए
        
        # ====================
        # Integration Continuity - Last Value Tracking
        # ====================
        
        # cumulative_trapezoid पिछले वैल्यू को नहीं जानता
        # cumulative_trapezoid doesn't know previous values
        
        # Continuous वेवफॉर्म के लिए, हमें पिछले बैच का आखिरी मान जोड़ना होगा
        # For continuous waveform, we must add last value of previous batch
        
        # Velocity के लिए पिछला मान - इंटीग्रेशन continuity के लिए
        # Previous value for velocity - for integration continuity
        self.last_velocity = 0.0
        
        # Displacement के लिए पिछला मान - इंटीग्रेशन continuity के लिए
        # Previous value for displacement - for integration continuity
        self.last_displacement = 0.0
    
    def process_batch(self, raw_batch):
        """
        50-सैंपल बैच को प्रोसेस करें - Drift-Free DSP Pipeline
        
        Process 50-sample batch - Drift-Free DSP Pipeline
        
        Parameters:
        -----------
        raw_batch : numpy array
            50 सैंपल्स का कच्चा डेटा (Raw data of 50 samples)
            Shape: (50,)
            
        Returns:
        --------
        results : dict
            डिक्शनरी जिसमें तीन_processed सिग्नल्स हैं:
            Dictionary containing three processed signals:
            
            {
                'raw_filtered': numpy array,   # Step 1: साफ़ किया हुआ सिग्नल (Cleaned signal)
                'velocity': numpy array,        # Step 2: वेलोसिटी (गति की दर)
                'displacement': numpy array     # Step 3: डिस्प्लेसमेंट (कुल विस्थापन)
            }
        """
        
        # ====================
        # STEP 1: Raw Batch -> Highpass Filter -> raw_filtered
        # ====================
        
        # कच्चे डेटा पर हाईपास फिल्टर लागू करें
        # Apply highpass filter to raw data
        
        # scipy.signal.sosfilt का उपयोग करके Second-Order Sections filtering
        # Using scipy.signal.sosfilt for Second-Order Sections filtering
        
        # sosfilt(data, sos, zi=None) returns:
        # - filtered_data: फिल्टर किया हुआ डेटा (Filtered data)
        # - zf: final filter states (अंतिम फिल्टर स्टेट्स)
        
        # zi (initial conditions) का उपयोग करने से:
        # Using zi (initial conditions):
        # - बैच के बीच कनेक्टिविटी बनी रहती है (Connectivity remains between batches)
        # - कोई अचानक जंप या ब्रेक नहीं होता (No sudden jumps or breaks)
        raw_filtered, self.zi_raw = signal.sosfilt(
            self.filter_sos,      # फिल्टर coefficients (Filter coefficients)
            raw_batch,            # इनपुट डेटा (Input data)
            zi=self.zi_raw        # पिछली स्टेट्स (Previous states) - continuity के लिए महत्वपूर्ण
        )
        
        # अब self.zi_raw अपडेट हो गया है - अगले बैच के लिए तैयार
        # Now self.zi_raw is updated - ready for next batch
        
        # ====================
        # STEP 2: raw_filtered -> Integration -> Velocity -> Highpass Filter
        # ====================
        
        # पहला इंटीग्रेशन - Velocity (गति) कैलकुलेट करें
        # First integration - Calculate Velocity (rate of change)
        
        # scipy.integrate.cumulative_trapezoid का उपयोग करें
        # Use scipy.integrate.cumulative_trapezoid
        
        # Trapezoidal rule: क्षेत्रफल को trapezoids में बांटता है
        # Trapezoidal rule: Divides area into trapezoids
        
        # Formula: integral ≈ Σ (y[i] + y[i+1]) / 2 * dx
        # जहाँ dx = समय अंतराल (time interval)
        
        # dx = 1/sampling_rate = 1ms = 0.001 सेकंड
        dx = 1.0 / self.sampling_rate
        
        # cumulative_trapezoid(y, dx=None, initial=0) returns:
        # - array जिसकी लंबाई len(y)-1 होती है (length is len(y)-1)
        # - initial=0 जोड़ने से output length = input length हो जाता है
        # - initial=0 adding makes output length = input length
        
        velocity_raw = integrate.cumulative_trapezoid(
            raw_filtered,         # इनपुट सिग्नल (Input signal)
            dx=dx,                # समय अंतराल (Time interval) - 0.001s
            initial=0             # शुरुआत में 0 जोड़ें (Add 0 at start) - length maintain करने के लिए
        )
        
        # IMPORTANT: Continuity के लिए पिछला मान जोड़ें
        # IMPORTANT: Add previous value for continuity
        
        # cumulative_trapezoid हर बैच को independently इंटीग्रेट करता है
        # cumulative_trapezoid integrates each batch independently
        
        # Continuous वेवफॉर्म के लिए, पिछले बैच का आखिरी मान जोड़ें
        # For continuous waveform, add last value of previous batch
        velocity_continuous = velocity_raw + self.last_velocity
        
        # Velocity पर हाईपास फिल्टर लागू करें - DC drift हटाने के लिए
        # Apply highpass filter to velocity - to remove DC drift
        
        # इंटीग्रेशन के बाद low-frequency drift आ सकता है
        # Low-frequency drift can come after integration
        
        # sosfilt with zi maintains continuity
        velocity_filtered, self.zi_vel = signal.sosfilt(
            self.filter_sos,      # वही फिल्टर (Same filter)
            velocity_continuous,  # Continuous velocity signal
            zi=self.zi_vel        # Velocity के लिए state - continuity के लिए
        )
        
        # अगले बैच के लिए last_velocity अपडेट करें
        # Update last_velocity for next batch
        
        # आखिरी सैंपल स्टोर करें - अगले बैच में जोड़ने के लिए
        # Store last sample - to add in next batch
        self.last_velocity = velocity_continuous[-1]
        
        # ====================
        # STEP 3: velocity_filtered -> Integration -> Displacement -> Highpass Filter
        # ====================
        
        # दूसरा इंटीग्रेशन - Displacement (विस्थापन) कैलकुलेट करें
        # Second integration - Calculate Displacement (total displacement)
        
        # Velocity का इंटीग्रेशन = Displacement (कुल तय की गई दूरी)
        # Integral of velocity = Displacement (total distance traveled)
        
        displacement_raw = integrate.cumulative_trapezoid(
            velocity_filtered,    # Velocity signal (already filtered)
            dx=dx,                # समय अंतराल (Time interval) - 0.001s
            initial=0             # शुरुआत में 0 जोड़ें (Add 0 at start)
        )
        
        # IMPORTANT: Continuity के लिए पिछला मान जोड़ें
        # IMPORTANT: Add previous value for continuity
        
        # Displacement को भी continuous बनाना होगा
        # Displacement also needs to be continuous
        displacement_continuous = displacement_raw + self.last_displacement
        
        # Displacement पर हाईपास फिल्टर लागू करें - DC drift हटाने के लिए
        # Apply highpass filter to displacement - to remove DC drift
        
        # Double integration के बाद drift बहुत अधिक हो सकता है
        # After double integration, drift can be very high
        
        # sosfilt with zi maintains continuity
        displacement_filtered, self.zi_disp = signal.sosfilt(
            self.filter_sos,      # वही फिल्टर (Same filter)
            displacement_continuous,  # Continuous displacement signal
            zi=self.zi_disp       # Displacement के लिए state - continuity के लिए
        )
        
        # अगले बैच के लिए last_displacement अपडेट करें
        # Update last_displacement for next batch
        
        # आखिरी सैंपल स्टोर करें - अगले बैच में जोड़ने के लिए
        # Store last sample - to add in next batch
        self.last_displacement = displacement_continuous[-1]
        
        # ====================
        # Results Dictionary बनाएं
        # ====================
        
        # तीनों_processed सिग्नल्स को डिक्शनरी में पैक करें
        # Pack all three processed signals into dictionary
        
        results = {
            'raw_filtered': raw_filtered,      # Step 1: Cleaned raw signal (DC removed)
            'velocity': velocity_filtered,     # Step 2: First derivative (rate of change)
            'displacement': displacement_filtered  # Step 3: Second derivative (total displacement)
        }
        
        return results
    
    def reset_state(self):
        """
        सभी स्टेट्स को रीसेट करें - नई सिक्वेंस के लिए
        
        Reset all states - for new sequence
        
        यह मेथड फिल्टर स्टेट्स (zi) और इंटीग्रेशन वैल्यूज को जीरो कर देता है।
        This method zeroes filter states (zi) and integration values.
        
        Use Case:
        ---------
        जब आप बिल्कुल नए सिरे से शुरू करना चाहते हैं
        When you want to start completely fresh
        """
        
        # फिल्टर स्टेट्स रीसेट करें - सभी zi को zeros बनाएं
        # Reset filter states - make all zi zeros
        
        n_sections = len(self.zi_raw)  # Number of sections
        
        # np.zeros_like() existing array के समान shape में zeros बनाता है
        # np.zeros_like() creates zeros in same shape as existing array
        self.zi_raw = np.zeros_like(self.zi_raw)
        self.zi_vel = np.zeros_like(self.zi_vel)
        self.zi_disp = np.zeros_like(self.zi_disp)
        
        # इंटीग्रेशन वैल्यूज रीसेट करें
        # Reset integration values
        self.last_velocity = 0.0
        self.last_displacement = 0.0
        
        print("DSP State reset - DSP स्टेट रीसेट हो गया")
    
    def get_filter_info(self):
        """
        फिल्टर की जानकारी प्राप्त करें
        
        Get filter information
        
        Returns:
        --------
        info : dict
            फिल्टर पैरामीटर्स का डिक्शनरी
            Dictionary of filter parameters
        """
        
        # फिल्टर जानकारी इकट्ठा करें
        # Collect filter information
        info = {
            'sampling_rate': self.sampling_rate,           # Hz में
            'cutoff_frequency': self.highpass_cutoff,      # Hz में
            'filter_type': 'Butterworth Highpass',         # फिल्टर प्रकार
            'filter_order': len(self.filter_sos) * 2,      # Filter order (sections * 2)
            'nyquist': self.sampling_rate / 2.0            # Nyquist frequency
        }
        
        return info


# ============================================================================
# उदाहरण उपयोग (Example Usage) - टेस्टिंग के लिए
# ============================================================================

if __name__ == "__main__":
    """
    मेन एक्जीक्यूशन ब्लॉक - केवल तब चलता है जब यह फ़ाइल सीधे चलाई जाती है
    
    Main execution block - only runs when this file is executed directly
    
    यह उदाहरण कोड दिखाता है कि NadiDSP का उपयोग कैसे करें।
    This example code shows how to use NadiDSP.
    """
    
    print("=" * 70)
    print("Ayurvedic Nadi Pariksha DSP Engine - नाड़ी परीक्षण DSP इंजन")
    print("=" * 70)
    print()
    
    # NadiDSP इंजन बनाएं
    # Create NadiDSP engine
    dsp = NadiDSP(
        sampling_rate=1000,           # 1000Hz सैम्पलिंग रेट
        highpass_cutoff=0.1           # 0.1Hz हाईपास कटऑफ
    )
    
    # फिल्टर जानकारी प्रिंट करें
    # Print filter information
    filter_info = dsp.get_filter_info()
    print("Filter Configuration:")
    print(f"  Sampling Rate: {filter_info['sampling_rate']}Hz")
    print(f"  Cutoff Frequency: {filter_info['cutoff_frequency']}Hz")
    print(f"  Filter Type: {filter_info['filter_type']}")
    print(f"  Filter Order: {filter_info['filter_order']}")
    print(f"  Nyquist Frequency: {filter_info['nyquist']}Hz")
    print()
    
    # टेस्ट डेटा बनाने के लिए nadi_generator import करें
    # Import nadi_generator for test data
    
    try:
        from nadi_generator import VirtualSensor
        print("VirtualSensor imported successfully - वर्चुअल सेंसर आयात सफल")
        print()
        
        # वर्चुअल सेंसर बनाएं और शुरू करें
        # Create and start virtual sensor
        sensor = VirtualSensor(
            sampling_rate=1000,
            batch_size=50,
            vata_strength=0.4,
            pitta_strength=0.3,
            kapha_strength=0.3
        )
        
        sensor.start()
        
        print("\nडेटा प्रोसेसिंग शुरू हो रही है... Starting data processing...\n")
        
        # 20 बैच प्रोसेस करें (1 सेकंड का डेटा)
        # Process 20 batches (1 second of data)
        batches_processed = 0
        max_batches = 20
        
        try:
            while batches_processed < max_batches:
                
                # कतार से बैच प्राप्त करें
                # Get batch from queue
                batch = sensor.get_latest_batch(timeout=0.1)
                
                # यदि डेटा मिला है
                # If data received
                if batch is not None:
                    
                    # DSP प्रोसेसिंग लागू करें
                    # Apply DSP processing
                    results = dsp.process_batch(batch)
                    
                    # प्रत्येक सिग्नल की जानकारी प्रिंट करें
                    # Print information for each signal
                    
                    # Shape = (50,) होना चाहिए
                    # Shape should be (50,)
                    print(f"Batch {batches_processed + 1}:")
                    print(f"  Raw Filtered:   Shape={results['raw_filtered'].shape}, "
                          f"Min={np.min(results['raw_filtered']):.4f}, "
                          f"Max={np.max(results['raw_filtered']):.4f}")
                    
                    print(f"  Velocity:       Shape={results['velocity'].shape}, "
                          f"Min={np.min(results['velocity']):.4f}, "
                          f"Max={np.max(results['velocity']):.4f}")
                    
                    print(f"  Displacement:   Shape={results['displacement'].shape}, "
                          f"Min={np.min(results['displacement']):.4f}, "
                          f"Max={np.max(results['displacement']):.4f}")
                    
                    # Queue size चेक करें
                    # Check queue size
                    print(f"  Queue Size: {sensor.get_queue_size()}")
                    print()
                    
                    # काउंटर बढ़ाएं
                    # Increment counter
                    batches_processed += 1
                    
                    # हर 5 बैच के बाद थोड़ा वेट करें
                    # Wait slightly after every 5 batches
                    if batches_processed % 5 == 0:
                        time.sleep(0.1)
                
                # यदि उपयोगकर्ता Ctrl+C दबाता है
                # If user presses Ctrl+C
        except KeyboardInterrupt:
            print("\n\nउपयोगकर्ता द्वारा रोका गया - Stopped by user")
        
        # सेंसर रोकें
        # Stop sensor
        print("\nसेंसर रोक रहा है... Stopping sensor...")
        sensor.stop()
        
    except ImportError as e:
        # यदि nadi_generator मौजूद नहीं है
        # If nadi_generator is not available
        
        print(f"\nWarning: nadi_generator not found - {e}")
        print("Creating synthetic test data instead...")
        print()
        
        # सिंथेटिक टेस्ट डेटा बनाएं
        # Create synthetic test data
        
        # Simple sine wave + some noise
        t = np.linspace(0, 0.05, 50)  # 50ms का डेटा
        test_batch = 0.5 * np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(50)
        
        print("Processing synthetic test batch...")
        results = dsp.process_batch(test_batch)
        
        print(f"Raw Filtered:   Shape={results['raw_filtered'].shape}")
        print(f"Velocity:       Shape={results['velocity'].shape}")
        print(f"Displacement:   Shape={results['displacement'].shape}")
    
    print("\n" + "=" * 70)
    print("DSP Test Complete - DSP परीक्षण पूर्ण")
    print("=" * 70)
    print(f"कुल बैच प्रोसेस किए: {batches_processed if 'batches_processed' in locals() else 1}")
    print()
