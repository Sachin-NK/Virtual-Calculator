#import libraries
import os
import cv2
import PIL
import numpy as np
import google.generativeai as genai
import streamlit as st
import mediapipe as mp
from streamlit_extras.add_vertical_space import add_vertical_space
from mediapipe.python.solutions import hands, drawing_utils
from dotenv import load_dotenv
import threading
from warnings import filterwarnings

filterwarnings(action='ignore')

class Calculator:

    # Display instructions for interacting with the virtual calculator
    def display_instructions(self):
      instructions = """
         ### Instructions
        - **Drawing Mode**: Index finger up
        - **Disable Connections**: Index and middle fingers up
        - **Erase Last Line**: Pinky finger up
        - **Reset Canvas**: Open palm
        - **Proceed**: Thumb up
        """
      st.sidebar.markdown(instructions)

    # Configure the Streamlit app
    def streamlit_config(self):
        st.set_page_config(page_title='SNK Virtual Calculator', layout="wide")
        page_background_color = """
        <style>
        [data-testid="stHeader"] {background: rgb(0,0,0);}
        .block-container {padding-top: 3rem;}
        </style>
        """
        st.markdown(page_background_color, unsafe_allow_html=True)
        st.markdown('<h1 style="text-align: center;">Virtual Calculator</h1>', unsafe_allow_html=True)
        self.display_instructions() 
        self.customization_options()
        add_vertical_space(1)

    def __init__(self):
        # Initialize environment variables from .env file
        load_dotenv()

        # Configure the Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 950) # Set webcam frame width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 550) # Set webcam frame height
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 130) # Set webcam brightness

        # Initialize Drawing Canvas
        self.imgCanvas = np.zeros((550, 950, 3), dtype=np.uint8)

        # Initialize MediaPipe Hands for hand detection
        self.mphands = hands.Hands(max_num_hands=1, min_detection_confidence=0.75)

        # Initialize drawing origin and timing variables
        self.p1, self.p2 = 0, 0
        self.p_time = 0

        # Create Fingers Open/Close Position List
        self.fingers = []

        self.imgRGB = None  # Initialize RGB image processing

    #Adds comization options to line color, thickness, and background color.
    def customization_options(self): 
        st.sidebar.markdown("### Customization") 
        self.line_color = st.sidebar.color_picker("Pick a line color", "#ff00ff") 
        self.line_thickness = st.sidebar.slider("Select line thickness", 1, 10, 5) 
        self.bg_color = st.sidebar.color_picker("Pick a background color", "#000000")

    # Capture a frame from the webcam
    def process_frame(self):
        success, img = self.cap.read()
        if not success:
            return False
        self.process_image(img)
        return True
    
    def process_image(self, img):
        img = cv2.resize(img, (950, 550))
        self.img = cv2.flip(img, 1)  # Mirror the image
        self.imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)# Convert BGR to RGB

    def process_hands(self):
        result = self.mphands.process(self.imgRGB)
        self.landmark_list = []
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                drawing_utils.draw_landmarks(self.img, hand_lms, hands.HAND_CONNECTIONS)
                self.landmark_list = self.extract_landmarks(hand_lms)

    def extract_landmarks(self, hand_lms):
        landmark_list = []
        for id, lm in enumerate(hand_lms.landmark):
            h, w, _ = self.img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmark_list.append([id, cx, cy])
        return landmark_list

    def identify_fingers(self):
        self.fingers = []
        if self.landmark_list:
            self.fingers = [
                1 if self.landmark_list[id][2] < self.landmark_list[id-2][2] else 0
                for id in [4, 8, 12, 16, 20]
            ]
            self.fingers[0] = 1 if self.landmark_list[4][1] < self.landmark_list[2][1] else 0

    # Perform actions based on finger positions
    def handle_drawing_mode(self):
        if sum(self.fingers) == 1 and self.fingers[1] == 1: # Index finger up
            self.draw_line()
        elif sum(self.fingers) == 2 and self.fingers[1] == self.fingers[2] == 1: # Index and middle fingers up
            self.reset_drawing_origin()
        elif sum(self.fingers) == 1 and self.fingers[4] == 1: # Pinky finger up
            self.erase_last_line()
        elif sum(self.fingers) == 5: # All fingers up
            self.reset_canvas()
        elif sum(self.fingers) == 1 and self.fingers[0] == 1: # Thumb up
            self.proceed_with_submission()

    # Convert a hex color string to a RGB tuple
    def hex_to_rgb(self, hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def draw_line(self):
        cx, cy = self.landmark_list[8][1], self.landmark_list[8][2]
        if self.p1 == 0 and self.p2 == 0:
            self.p1, self.p2 = cx, cy

        # Convert hex color to BGR
        line_color_bgr = self.hex_to_rgb(self.line_color)
        
         # Draw the line on the canvas
        cv2.line(self.imgCanvas, (self.p1, self.p2), (cx, cy), line_color_bgr, self.line_thickness)
        
        # Update previous position
        self.p1, self.p2 = cx, cy

    def reset_drawing_origin(self):
        self.p1, self.p2 = 0, 0

    def erase_last_line(self):
        pass

    def reset_canvas(self):
        self.imgCanvas = np.zeros((550, 950, 3), dtype=np.uint8)

    def proceed_with_submission(self):
        print("Proceeding with the drawing.")

    def blend_canvas_with_feed(self):
        img = cv2.addWeighted(self.img, 0.7, self.imgCanvas, 1, 0)
        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        self.img = cv2.bitwise_or(img, self.imgCanvas)

    def analyze_image_with_genai(self):
        imgCanvas = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2RGB)
        imgCanvas = PIL.Image.fromarray(imgCanvas)
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        prompt = (
            "Analyze the image and provide the following:\n"
            "* The mathematical equation represented in the image.\n"
            "* The solution to the equation.\n"
            "* A short and concise explanation of the steps taken to arrive at the solution."
        )
        response = model.generate_content([prompt, imgCanvas])
        return response.text

    def main(self):
        col1, _, col3 = st.columns([0.8, 0.02, 0.18])
        with col1:
            stframe = st.empty()
        with col3:
            st.markdown('<h5 style="text-position:center;color:green;">OUTPUT:</h5>', unsafe_allow_html=True)
            result_placeholder = st.empty()

        while True:
            # Display an error if the webcam is not accessible
            if not self.cap.isOpened():
                add_vertical_space(5)
                st.markdown('<h4 style="text-position:center; color:red;">Error: web cam can not be open. '
                            'Please ensure your webcam is connected and try again</h4>', unsafe_allow_html=True)
                break

            if not self.process_frame():
                continue

            self.process_hands()
            self.identify_fingers()
            self.handle_drawing_mode()
            self.blend_canvas_with_feed()

            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            stframe.image(self.img, channels="RGB")

            if sum(self.fingers) == 2 and self.fingers[1] == self.fingers[2] == 1:
                result = self.analyze_image_with_genai()
                result_placeholder.write(f"Result: {result}")

        self.cap.release()
        cv2.destroyAllWindows()

try:
    calc = Calculator() 
    calc.streamlit_config()
    calc.main()             
except Exception as e:
    add_vertical_space(5)
    st.markdown(f'<h5 style="text-position:center;color:orange;">{e}</h5>', unsafe_allow_html=True)
