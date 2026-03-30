"""
nadi_main.py - आयुर्वेदिक नाड़ी परीक्षण के लिए डेस्कटॉप GUI एप्लिकेशन
Desktop GUI Application for Ayurvedic Nadi Pariksha

यह मॉड्यूल तीनों मॉड्यूल को एक साथ लाता है:
- nadi_generator से VirtualSensor (डेटा जनरेशन)
- nadi_dsp से NadiDSP (सिग्नल प्रोसेसिंग)
- PyQt6 + pyqtgraph (लाइव विजुअलाइजेशन)
"""

import sys                          # सिस्टम पैरामीटर्स और एग्जिट कोड के लिए (System parameters & exit codes)
import numpy as np                  # न्यूमेरिकल कंप्यूटेशन के लिए (Numerical computation)
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel  # PyQt6 UI components
from PyQt6.QtCore import QTimer     # टाइमर के लिए - नियमित अंतराल पर इवेंट ट्रिगर करने के लिए (Timer for periodic events)
from PyQt6.QtGui import QFont       # फ़ॉन्ट स्टाइलिंग के लिए (Font styling)
import pyqtgraph as pg              # रियल-टाइम ग्राफ़ प्लॉटिंग के लिए (Real-time graph plotting)

# हमारे कस्टम मॉड्यूल इंपोर्ट करें
# Import our custom modules
from nadi_generator import VirtualSensor  # वर्चुअल सेंसर - सिंथेटिक पल्स वेव जनरेट करता है
from nadi_dsp import NadiDSP              # DSP इंजन - सिग्नल प्रोसेसिंग और इंटीग्रेशन


class NadiMainWindow(QMainWindow):
    """
    नाड़ी मेन विंडो - मुख्य GUI एप्लिकेशन
    
    Nadi Main Window - Main GUI application
    
    यह क्लास PyQt6 का उपयोग करके डेस्कटॉप GUI बनाती है जिसमें:
    This class creates desktop GUI using PyQt6 with:
    
    1. Start/Stop बटन - सिमुलेशन कंट्रोल
       Start/Stop buttons - Simulation control
       
    2. 3 लाइव ग्राफ़ - रियल-टाइम विजुअलाइजेशन
       3 Live graphs - Real-time visualization
       - Raw Sensor Wave (पीला/Yellow)
       - Velocity Wave (नीला/Blue)
       - Displacement Wave (हरा/Green)
    """
    
    def __init__(self):
        """
        मेन विंडो का कंस्ट्रक्टर - UI और कंपोनेंट्स को इनिशियलाइज करें
        
        Main window constructor - Initialize UI and components
        """
        
        # पेरेंट क्लास (QMainWindow) को इनिशियलाइज करें
        # Initialize parent class (QMainWindow)
        super().__init__()
        
        # ====================
        # विंडो सेटअप - Window Setup
        # ====================
        
        # विंडो का शीर्षक सेट करें
        # Set window title
        self.setWindowTitle("Ayurvedic Nadi Pariksha DSP Sandbox - आयुर्वेदिक नाड़ी परीक्षा")
        
        # विंडो की आकार सेट करें - चौड़ाई x ऊंचाई (pixels में)
        # Set window size - width x height (in pixels)
        self.setGeometry(100, 100, 1400, 900)  # (x, y, width, height)
        
        # ====================
        # कंपोनेंट्स इनिशियलाइजेशन - Components Initialization
        # ====================
        
        # VirtualSensor बनाएं - 1000Hz, 50-सैंपल बैच
        # Create VirtualSensor - 1000Hz, 50-sample batch
        self.sensor = VirtualSensor(
            sampling_rate=1000,           # 1000Hz सैम्पलिंग रेट
            batch_size=50,                # 50 सैंपल्स प्रति बैच
            vata_strength=0.4,            # वात: 40% तीव्रता
            pitta_strength=0.3,           # पित्त: 30% तीव्रता
            kapha_strength=0.3            # कफ: 30% तीव्रता
        )
        
        # NadiDSP इंजन बनाएं - 0.1Hz हाईपास फिल्टर
        # Create NadiDSP engine - 0.1Hz highpass filter
        self.dsp = NadiDSP(
            sampling_rate=1000,           # 1000Hz सैम्पलिंग रेट
            highpass_cutoff=0.1           # 0.1Hz हाईपास कटऑफ
        )
        
        # ====================
        # Data Buffers - ग्राफ़ डेटा स्टोरेज
        # ====================
        
        # अधिकतम सैंपल्स जो ग्राफ़ पर दिखाए जाएंगे - 2000 सैंपल्स = 2 सेकंड
        # Maximum samples to display on graph - 2000 samples = 2 seconds
        self.max_samples = 2000
        
        # डेटा बफर्स बनाएं - खाली numpy arrays
        # Create data buffers - empty numpy arrays
        
        # raw_filtered डेटा के लिए बफर (पीला ग्राफ़)
        # Buffer for raw_filtered data (yellow graph)
        self.raw_buffer = np.array([])
        
        # velocity डेटा के लिए बफर (नीला ग्राफ़)
        # Buffer for velocity data (blue graph)
        self.velocity_buffer = np.array([])
        
        # displacement डेटा के लिए बफर (हरा ग्राफ़)
        # Buffer for displacement data (green graph)
        self.displacement_buffer = np.array([])
        
        # ====================
        # UI सेटअप - User Interface Setup
        # ====================
        
        # मेन विंडो बनाने के लिए UI सेटअप फंक्शन कॉल करें
        # Call UI setup function to create main window
        self.setup_ui()
        
        # ====================
        # Timer सेटअप - QTimer for Data Polling
        # ====================
        
        # QTimer बनाएं - हर 50ms में डेटा पोल करने के लिए
        # Create QTimer - to poll data every 50ms
        
        # QTimer एक टाइमर है जो नियमित अंतराल पर signal भेजता है
        # QTimer is a timer that sends signals at regular intervals
        
        # 50ms = 0.05 सेकंड = 20 बार प्रति सेकंड (20 Hz)
        # 50ms = 0.05 seconds = 20 times per second (20 Hz)
        self.timer = QTimer(self)
        
        # timer.timeout signal को update_data मेथड से कनेक्ट करें
        # Connect timer.timeout signal to update_data method
        
        # जब भी टाइमर ट्रिगर होगा, update_data() कॉल किया जाएगा
        # Whenever timer triggers, update_data() will be called
        self.timer.timeout.connect(self.update_data)
        
        # टाइमर इंटरवल सेट करें - 50 मिलीसेकंड
        # Set timer interval - 50 milliseconds
        self.timer.setInterval(50)  # 50ms
        
        # सिमुलेशन स्थिति फ्लैग - क्या सिमुलेशन चल रहा है?
        # Simulation status flag - is simulation running?
        self.is_running = False
    
    def setup_ui(self):
        """
        यूजर इंटरफेस सेटअप - सभी UI एलिमेंट्स बनाएं
        
        User Interface Setup - Create all UI elements
        
        यह मेथड विंडो के अंदर सभी बटन, लेबल और ग्राफ़ बनाता है।
        This method creates all buttons, labels and graphs inside the window.
        """
        
        # ====================
        # Central Widget और Layout
        # ====================
        
        # QMainWindow के लिए central widget बनाएं
        # Create central widget for QMainWindow
        
        # QWidget एक कंटेनर है जिसमें हम अन्य UI एलिमेंट्स रखते हैं
        # QWidget is a container where we place other UI elements
        central_widget = QWidget()
        
        # central widget को मेन विंडो में सेट करें
        # Set central widget to main window
        self.setCentralWidget(central_widget)
        
        # मुख्य लेआउट बनाएं - vertical布局 (ऊपर से नीचे)
        # Create main layout - vertical layout (top to bottom)
        
        # QVBoxLayout एलिमेंट्स को ऊपर से नीचे व्यवस्थित करता है
        # QVBoxLayout arranges elements from top to bottom
        main_layout = QVBoxLayout()
        
        # central widget पर layout लागू करें
        # Apply layout to central widget
        central_widget.setLayout(main_layout)
        
        # ====================
        # Control Panel - बटन और लेबल
        # ====================
        
        # बटन पैनल के लिए horizontal layout (बाएं से दाएं)
        # Horizontal layout for button panel (left to right)
        
        # QHBoxLayout एलिमेंट्स को बाएं से दाएं व्यवस्थित करता है
        # QHBoxLayout arranges elements from left to right
        control_layout = QHBoxLayout()
        
        # ====================
        # Start Button
        # ====================
        
        # Start Simulation बटन बनाएं
        # Create Start Simulation button
        self.start_button = QPushButton("▶ Start Simulation")
        
        # बटन का आकार सेट करें - चौड़ाई x ऊंचाई (pixels)
        # Set button size - width x height (pixels)
        self.start_button.setFixedSize(200, 50)
        
        # बटन के टेक्स्ट का फ़ॉन्ट सेट करें - bold और बड़ा
        # Set button text font - bold and large
        self.start_button.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        
        # बटन का बैकग्राउंड रंग हरा सेट करें (start के लिए)
        # Set button background color green (for start)
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white;")
        
        # बटन को click event से कनेक्ट करें
        # Connect button to click event
        
        # जब कोई बटन क्लिक करेगा, start_simulation() कॉल होगा
        # When someone clicks button, start_simulation() will be called
        self.start_button.clicked.connect(self.start_simulation)
        
        # बटन को control layout में जोड़ें
        # Add button to control layout
        control_layout.addWidget(self.start_button)
        
        # ====================
        # Stop Button
        # ====================
        
        # Stop Simulation बटन बनाएं
        # Create Stop Simulation button
        self.stop_button = QPushButton("⏹ Stop Simulation")
        
        # बटन का आकार सेट करें - चौड़ाई x ऊंचाई (pixels)
        # Set button size - width x height (pixels)
        self.stop_button.setFixedSize(200, 50)
        
        # बटन के टेक्स्ट का फ़ॉन्ट सेट करें - bold और बड़ा
        # Set button text font - bold and large
        self.stop_button.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        
        # बटन का बैकग्राउंड रंग लाल सेट करें (stop के लिए)
        # Set button background color red (for stop)
        self.stop_button.setStyleSheet("background-color: #f44336; color: white;")
        
        # बटन को disable कर दें - शुरू में बंद रहेगा
        # Disable button initially - will remain off at start
        self.stop_button.setEnabled(False)
        
        # बटन को click event से कनेक्ट करें
        # Connect button to click event
        
        # जब कोई बटन क्लिक करेगा, stop_simulation() कॉल होगा
        # When someone clicks button, stop_simulation() will be called
        self.stop_button.clicked.connect(self.stop_simulation)
        
        # बटन को control layout में जोड़ें
        # Add button to control layout
        control_layout.addWidget(self.stop_button)
        
        # ====================
        # Status Label
        # ====================
        
        # Status दिखाने के लिए लेबल बनाएं
        # Create label to show status
        self.status_label = QLabel("Status: Ready to Start")
        
        # लेबल का फ़ॉन्ट सेट करें - थोड़ा बड़ा
        # Set label font - slightly large
        self.status_label.setFont(QFont("Arial", 12))
        
        # लेबल को control layout में जोड़ें
        # Add label to control layout
        control_layout.addWidget(self.status_label)
        
        # control layout को main layout में जोड़ें
        # Add control layout to main layout
        main_layout.addLayout(control_layout)
        
        # ====================
        # Graph Widgets - 3 pyqtgraph प्लॉट
        # ====================
        
        # ====================
        # Graph 1: Raw Sensor Wave (पीला/Yellow)
        # ====================
        
        # पहले ग्राफ़ के लिए लेबल बनाएं
        # Create label for first graph
        raw_label = QLabel("Raw Sensor Wave (Filtered) - पीला ग्राफ़")
        raw_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        main_layout.addWidget(raw_label)
        
        # pyqtgraph PlotWidget बनाएं - raw wave के लिए
        # Create pyqtgraph PlotWidget for raw wave
        
        # PlotWidget एक interactive plot एरिया है
        # PlotWidget is an interactive plot area
        self.raw_plot = pg.PlotWidget()
        
        # ग्राफ़ का बैकग्राउंड रंग सेट करें - dark theme
        # Set graph background color - dark theme
        self.raw_plot.setBackground('w')  # white background
        
        # ग्राफ़ का शीर्षक सेट करें
        # Set graph title
        self.raw_plot.setTitle("Raw Filtered Signal", color='k', size='12pt')
        
        # X-axis लेबल सेट करें - समय (milliseconds)
        # Set X-axis label - time (milliseconds)
        self.raw_plot.setLabel('bottom', 'Time', units='ms')
        
        # Y-axis लेबल सेट करें - एम्पलीट्यूड (arbitrary units)
        # Set Y-axis label - amplitude (arbitrary units)
        self.raw_plot.setLabel('left', 'Amplitude', units='AU')
        
        # ग्राफ़ पर grid lines चालू करें
        # Turn on grid lines on graph
        self.raw_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # ग्राफ़ की Y-axis range सेट करें - fixed scale
        # Set graph Y-axis range - fixed scale
        self.raw_plot.setYRange(-0.5, 1.5)
        
        # ग्राफ़ को main layout में जोड़ें
        # Add graph to main layout
        main_layout.addWidget(self.raw_plot)
        
        # Plot curve बनाएं - पीले रंग में
        # Create plot curve - in yellow color
        
        # plot() खाली डेटा के साथ शुरू होता है
        # plot() starts with empty data
        self.raw_curve = self.raw_plot.plot(pen=pg.mkPen('y', width=2))  # yellow pen, width=2
        
        # ====================
        # Graph 2: Velocity Wave (नीला/Blue)
        # ====================
        
        # दूसरे ग्राफ़ के लिए लेबल बनाएं
        # Create label for second graph
        velocity_label = QLabel("Velocity Wave (First Integration) - नीला ग्राफ़")
        velocity_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        main_layout.addWidget(velocity_label)
        
        # pyqtgraph PlotWidget बनाएं - velocity wave के लिए
        # Create pyqtgraph PlotWidget for velocity wave
        self.velocity_plot = pg.PlotWidget()
        
        # ग्राफ़ का बैकग्राउंड रंग सेट करें
        # Set graph background color
        self.velocity_plot.setBackground('w')
        
        # ग्राफ़ का शीर्षक सेट करें
        # Set graph title
        self.velocity_plot.setTitle("Velocity Signal", color='k', size='12pt')
        
        # X-axis लेबल सेट करें
        # Set X-axis label
        self.velocity_plot.setLabel('bottom', 'Time', units='ms')
        
        # Y-axis लेबल सेट करें
        # Set Y-axis label
        self.velocity_plot.setLabel('left', 'Velocity', units='AU/ms')
        
        # ग्राफ़ पर grid lines चालू करें
        # Turn on grid lines on graph
        self.velocity_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # ग्राफ़ की Y-axis range सेट करें
        # Set graph Y-axis range
        self.velocity_plot.setYRange(-0.1, 0.1)
        
        # ग्राफ़ को main layout में जोड़ें
        # Add graph to main layout
        main_layout.addWidget(self.velocity_plot)
        
        # Plot curve बनाएं - नीले रंग में
        # Create plot curve - in blue color
        self.velocity_curve = self.velocity_plot.plot(pen=pg.mkPen('b', width=2))  # blue pen, width=2
        
        # ====================
        # Graph 3: Displacement Wave (हरा/Green)
        # ====================
        
        # तीसरे ग्राफ़ के लिए लेबल बनाएं
        # Create label for third graph
        displacement_label = QLabel("Displacement Wave / Vata-Pitta-Kapha Morphology - हरा ग्राफ़")
        displacement_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        main_layout.addWidget(displacement_label)
        
        # pyqtgraph PlotWidget बनाएं - displacement wave के लिए
        # Create pyqtgraph PlotWidget for displacement wave
        self.displacement_plot = pg.PlotWidget()
        
        # ग्राफ़ का बैकग्राउंड रंग सेट करें
        # Set graph background color
        self.displacement_plot.setBackground('w')
        
        # ग्राफ़ का शीर्षक सेट करें
        # Set graph title
        self.displacement_plot.setTitle("Displacement Signal (Tridosha Morphology)", color='k', size='12pt')
        
        # X-axis लेबल सेट करें
        # Set X-axis label
        self.displacement_plot.setLabel('bottom', 'Time', units='ms')
        
        # Y-axis लेबल सेट करें
        # Set Y-axis label
        self.displacement_plot.setLabel('left', 'Displacement', units='AU·ms')
        
        # ग्राफ़ पर grid lines चालू करें
        # Turn on grid lines on graph
        self.displacement_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # ग्राफ़ की Y-axis range सेट करें
        # Set graph Y-axis range
        self.displacement_plot.setYRange(-0.05, 0.05)
        
        # ग्राफ़ को main layout में जोड़ें
        # Add graph to main layout
        main_layout.addWidget(self.displacement_plot)
        
        # Plot curve बनाएं - हरे रंग में
        # Create plot curve - in green color
        self.displacement_curve = self.displacement_plot.plot(pen=pg.mkPen('g', width=2))  # green pen, width=2
    
    def update_data(self):
        """
        QTimer callback - हर 50ms में डेटा पोल और अपडेट करें
        
        QTimer callback - Poll and update data every 50ms
        
        यह मेथड स्वचालित रूप से हर 50ms में कॉल होता है जब टाइमर चल रहा होता है।
        This method is automatically called every 50ms when timer is running.
        
        कार्य (Tasks):
        1. VirtualSensor की queue से 50-सैंपल बैच निकालें
        2. NadiDSP को भेजें प्रोसेसिंग के लिए
        3. परिणामों को ग्राफ़ पर अपडेट करें
        """
        
        # ====================
        # Step 1: Queue से डेटा निकालें
        # ====================
        
        # Queue से बैच प्राप्त करें - non-blocking call
        # Get batch from queue - non-blocking call
        
        # get_nowait() तुरंत return करता है, भले ही queue खाली हो
        # get_nowait() returns immediately, even if queue is empty
        try:
            # queue.get_nowait() - यदि queue खाली है तो queue.Empty exception उठाता है
            # queue.get_nowait() - raises queue.Empty exception if queue is empty
            batch = self.sensor.data_queue.get_nowait()
            
        except Exception:
            # यदि queue खाली है (कोई डेटा नहीं आया)
            # If queue is empty (no data arrived)
            
            # चुपचाप return करें - कोई एरर नहीं
            # Return quietly - no error
            return
        
        # ====================
        # Step 2: DSP प्रोसेसिंग
        # ====================
        
        # 50-सैंपल बैच को DSP इंजन में प्रोसेस करें
        # Process 50-sample batch through DSP engine
        
        # process_batch() डिक्शनरी return करता है:
        # process_batch() returns dictionary:
        # {'raw_filtered': array, 'velocity': array, 'displacement': array}
        results = self.dsp.process_batch(batch)
        
        # ====================
        # Step 3: Data Buffers अपडेट करें
        # ====================
        
        # नए डेटा को buffers में जोड़ें - append operation
        # Add new data to buffers - append operation
        
        # np.concatenate() दो arrays को जोड़ता है
        # np.concatenate() joins two arrays
        
        # Raw filtered data अपडेट करें
        # Update raw filtered data
        self.raw_buffer = np.concatenate([self.raw_buffer, results['raw_filtered']])
        
        # Velocity data अपडेट करें
        # Update velocity data
        self.velocity_buffer = np.concatenate([self.velocity_buffer, results['velocity']])
        
        # Displacement data अपडेट करें
        # Update displacement data
        self.displacement_buffer = np.concatenate([self.displacement_buffer, results['displacement']])
        
        # ====================
        # Step 4: Buffer Size मैनेजमेंट - 2000 सैंपल्स लिमिट
        # ====================
        
        # जांचें कि buffer अधिकतम लिमिट से तो नहीं गया
        # Check if buffer exceeded maximum limit
        
        # len(array) array की लंबाई बताता है
        # len(array) returns array length
        if len(self.raw_buffer) > self.max_samples:
            
            # अधिक डेटा काट दें - पिछले 2000 सैंपल्स रखें
            # Trim excess data - keep last 2000 samples
            
            # array[-2000:] आखिरी 2000 elements देता है
            # array[-2000:] gives last 2000 elements
            self.raw_buffer = self.raw_buffer[-self.max_samples:]
            self.velocity_buffer = self.velocity_buffer[-self.max_samples:]
            self.displacement_buffer = self.displacement_buffer[-self.max_samples:]
        
        # ====================
        # Step 5: Graph Curves अपडेट करें
        # ====================
        
        # pyqtgraph curves को नया डेटा दें
        # Give new data to pyqtgraph curves
        
        # setData() curve का डेटा अपडेट करता है
        # setData() updates curve data
        
        # Raw curve अपडेट करें - पीला ग्राफ़
        # Update raw curve - yellow graph
        self.raw_curve.setData(self.raw_buffer)
        
        # Velocity curve अपडेट करें - नीला ग्राफ़
        # Update velocity curve - blue graph
        self.velocity_curve.setData(self.velocity_buffer)
        
        # Displacement curve अपडेट करें - हरा ग्राफ़
        # Update displacement curve - green graph
        self.displacement_curve.setData(self.displacement_buffer)
    
    def start_simulation(self):
        """
        Start बटन क्लिक हैंडलर - सिमुलेशन शुरू करें
        
        Start button click handler - Start simulation
        
        यह मेथड Start बटन दबाने पर कॉल होता है।
        This method is called when Start button is pressed.
        """
        
        # जांचें कि सिमुलेशन पहले से तो नहीं चल रहा
        # Check if simulation is already running
        
        if self.is_running:
            # यदि पहले से चल रहा है, तो कुछ मत करो
            # If already running, do nothing
            return
        
        # ====================
        # Step 1: VirtualSensor शुरू करें
        # ====================
        
        # वर्चुअल सेंसर को start() कॉल करें - बैकग्राउंड थ्रेड शुरू
        # Call start() on virtual sensor - start background thread
        self.sensor.start()
        
        # ====================
        # Step 2: QTimer शुरू करें
        # ====================
        
        # टाइमर को start() कॉल करें - 50ms इंटरवल पर ट्रिगर होगा
        # Call start() on timer - will trigger at 50ms intervals
        self.timer.start()
        
        # ====================
        # Step 3: UI अपडेट करें
        # ====================
        
        # Status फ्लैग सेट करें - सिमुलेशन चल रहा है
        # Set status flag - simulation is running
        self.is_running = True
        
        # Start बटन को disable करें - दोबारा क्लिक न हो
        # Disable start button - prevent double-click
        self.start_button.setEnabled(False)
        
        # Stop बटन को enable करें - अब दबाया जा सकता है
        # Enable stop button - can be pressed now
        self.stop_button.setEnabled(True)
        
        # Status लेबल अपडेट करें - हरा रंग
        # Update status label - green color
        self.status_label.setText("Status: ● Running")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        
        # DSP state रीसेट करें - नई सिक्वेंस के लिए
        # Reset DSP state - for new sequence
        self.dsp.reset_state()
        
        # Data buffers साफ़ करें - खाली arrays
        # Clear data buffers - empty arrays
        self.raw_buffer = np.array([])
        self.velocity_buffer = np.array([])
        self.displacement_buffer = np.array([])
        
        # Console पर मैसेज प्रिंट करें
        # Print message on console
        print("Simulation started - सिमुलेशन शुरू हो गया")
    
    def stop_simulation(self):
        """
        Stop बटन क्लिक हैंडलर - सिमुलेशन रोकें
        
        Stop button click handler - Stop simulation
        
        यह मेथड Stop बटन दबाने पर कॉल होता है।
        This method is called when Stop button is pressed.
        """
        
        # जांचें कि सिमुलेशन चल तो रहा है
        # Check if simulation is actually running
        
        if not self.is_running:
            # यदि नहीं चल रहा, तो कुछ मत करो
            # If not running, do nothing
            return
        
        # ====================
        # Step 1: QTimer रोकें
        # ====================
        
        # टाइमर को stop() कॉल करें - अब update_data() नहीं कॉल होगा
        # Call stop() on timer - update_data() won't be called now
        self.timer.stop()
        
        # ====================
        # Step 2: VirtualSensor रोकें
        # ====================
        
        # वर्चुअल सेंसर को stop() कॉल करें - थ्रेड ग्रेशफुली टर्मिनेट
        # Call stop() on virtual sensor - gracefully terminate thread
        self.sensor.stop()
        
        # ====================
        # Step 3: UI अपडेट करें
        # ====================
        
        # Status फ्लैग रीसेट करें - सिमुलेशन बंद हो गया
        # Reset status flag - simulation stopped
        self.is_running = False
        
        # Start बटन को enable करें - फिर से शुरू किया जा सकता है
        # Enable start button - can start again
        self.start_button.setEnabled(True)
        
        # Stop बटन को disable करें - अब नहीं दबाया जा सकता
        # Disable stop button - cannot be pressed now
        self.stop_button.setEnabled(False)
        
        # Status लेबल अपडेट करें - लाल रंग
        # Update status label - red color
        self.status_label.setText("Status: ■ Stopped")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        # Console पर मैसेज प्रिंट करें
        # Print message on console
        print("Simulation stopped - सिमुलेशन रुक गया")


# ============================================================================
# मेन एक्जीक्यूशन - Main Execution
# ============================================================================

if __name__ == "__main__":
    """
    मेन एक्जीक्यूशन ब्लॉक - एप्लिकेशन को चलाता है
    
    Main execution block - Runs the application
    
    यह कोड तब चलता है जब यह फ़ाइल सीधे चलाई जाती है।
    This code runs when this file is executed directly.
    """
    
    # ====================
    # PyQt6 एप्लिकेशन इनिशियलाइजेशन
    # ====================
    
    # QApplication बनाएं - सभी PyQt6 एप्लिकेशन के लिए आवश्यक
    # Create QApplication - required for all PyQt6 applications
    
    # sys.argv command-line arguments पास करता है
    # sys.argv passes command-line arguments
    app = QApplication(sys.argv)
    
    # ====================
    # मेन विंडो बनाएं और दिखाएं
    # ====================
    
    # NadiMainWindow का इंस्टेंस बनाएं
    # Create instance of NadiMainWindow
    window = NadiMainWindow()
    
    # विंडो को स्क्रीन पर दिखाएं
    # Show window on screen
    window.show()
    
    # ====================
    # एप्लिकेशन लूप शुरू करें
    # ====================
    
    # exec() एप्लिकेशन मेन लूप शुरू करता है
    # exec() starts application main loop
    
    # यह लूप तब तक चलता रहता है जब तक उपयोगकर्ता विंडो बंद नहीं करता
    # This loop continues until user closes window
    sys.exit(app.exec())
