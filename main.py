import pickle, json, time, re, uuid
from difflib import SequenceMatcher
from threading import Thread, main_thread, Event
from queue import Queue
from itertools import permutations

import pyaudio
import wave

import kivy
from kivy.app import App
from kivy.uix.label import Label # Imports Label element
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout # Imports layout call function which pulls from .kv file with the same name as the class that calls it
from kivy.uix.widget import Widget
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.accordion import Accordion, AccordionItem
from kivy.uix.treeview import TreeView, TreeViewLabel
from kivy.uix.popup import Popup
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.core.clipboard import Clipboard
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem


from audio_proc import transcribeAudio

class RootLayout(TabbedPanel): # Constructs a UI element based on the kivy BoxLayout class 
    def __init__(self, **kwargs):
        super(RootLayout, self).__init__(**kwargs) # Calls the superconstructor 
    
    def removeTab(self, tab_id):
        app: NLPDocApp = App.get_running_app()
        tab = None
        for tab in self.tab_list:
            if tab.tab_id == tab_id:
                tab = tab
                break
        if tab:
            app.tabs.remove(tab_id)
            # self.switch_to(self.default_tab) # Ignore if you want to temporarily keep inputs
            self.remove_widget(tab)

class CBoxLayout(BoxLayout):
    def __init__(self, tab_id = 1, **kwargs):
        super().__init__(**kwargs)
        self.tab_id = tab_id
        
 
    def copyLabel(self):
        Clipboard.copy(self.ids.output_text.text)

    def dupeTab(self):
        app: NLPDocApp = App.get_running_app()
        new_tab_id = app.getNewTabId() # This variable will be assigned to both the panel container and the child box to bind the two 
        new_tab = CTabbedPanelItem(text=F"Tab {new_tab_id}", tab_id=new_tab_id)
        new_content = CBoxLayout(tab_id=new_tab_id)
        
        # Copy contents
        new_content.ids.output_text.text = self.ids.output_text.text
        
        new_tab.add_widget(new_content)
        app.root.add_widget(new_tab)
        
    def removeTab(self):
        app = App.get_running_app()
        root_layout: RootLayout = app.root
        root_layout.removeTab(self.tab_id)
        
    def initRecording(self):
        if self.ids.rec.state == "normal": # Only execute if untoggled
            Thread(target=self.captureAudio).start()
    
    def captureAudio(self, sample_rate=44100, channels=1):
        file_path = "data/temp_out.wav"
        print("Recording audio...")
        
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=1024)

        frames = []
        start = time.time()
        
        while self.ids.rec.state == "down": 
            data = stream.read(1024)
            frames.append(data)
            
            elapsed = time.time() - start
            
            secs = elapsed % 60
            mins = elapsed // 60
            hours = mins // 60 
            
            self.ids.timetracker.text = F"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}"
            

        print("Recording completed.")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        print(F"Writing file...")
        # Save audio to an MP3 file
        audio_file = wave.open(file_path, "wb")
        audio_file.setnchannels(1)
        audio_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(b"".join(frames)) # Join frames together
        audio_file.close()
        
        Clock.schedule_once(lambda dt: self.transcribeAudio(dt), 0) # Schedule action for when graphics can update, dt is automatically passed in as the time b/n scheduling and calling of function
    
    def transcribeAudio(self, dt, *args):
        file_path = "temp_out.wav"
        print(F"Transcribing audio")
        transcription = transcribeAudio(file_path)
        if isinstance(transcription, str):
            print(transcription)
            self.ids.output_text.text = transcription          
        else:
            print("Wrong output format")
        

        
class CTabbedPanelItem(TabbedPanelItem):
    def __init__(self, tab_id = 1, **kwargs):
        super().__init__(**kwargs)
        self.tab_id = tab_id
    pass

class CPopup(Popup):
    pass

class CButton(Button):
    pass

class CScrollView(ScrollView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = "testid"
    pass


class NLPDocApp(App): 
    """
    This class inherits the App class from kivy
    The name of this class will also determine the name of the .kv files (for layout/design)
    .kv file name is not case-sensitive to the class name but should be all lowercase to avoid issues
    .kv file name can exclude "App" portion of App class identifier
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tabs = [1] # Initiate tab containing 1 tab id
        self.finished_check = Event() # Cross-thread event to indicate whether or not drug checking has finished
        self.reports_queue = Queue() # Queue for nodes to be added 
        self.recording = False
    


    
    
    # def build(self): # Returns the UI
    #     root = RootLayout()
    #     return root # Return whatever root element you are using for the UI
        


app_instance = NLPDocApp() # Creates instance of the app
app_instance.run() # Runs the instance of the app 
