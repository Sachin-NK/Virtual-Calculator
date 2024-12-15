# Virtual-Calculator

## Overview

This project is an advanced virtual calculator that leverages hand gesture recognition using MediaPipe and a user-friendly interface built with Streamlit. Additionally, it integrates with Google Generative AI to provide content analysis for user inputs, enhancing its functionality beyond basic calculations.

### Features

#### Gesture-Based Interaction
 - Draw: Raise the index finger to start drawing.  
 - Reset Origin: Raise both index and middle fingers to reset the drawing point.  
 - Clear Canvas: Show an open palm to clear everything.  
 - Submit Drawing: Give a thumbs-up to analyze the equation.  

#### Customize User Experience
 * Pick the favorite line color.  
 * Adjust the line thickness for precision.  
 * Change the background color to suit user style.  

#### Powered by AI  
  Once user submit his drawing, the app uses Google Generative AI to:   
   + Identify the mathematical equation user drew.  
   + Solve the equation.  
   + Explain the solution step by step.  

#### Real-Time Feedback
  - User's drawing is overlaid directly on the live webcam feed, making the experience seamless and real-time.  


### Technologies Used
* MediaPipe: For real-time hand detection and gesture recognition.
* Streamlit: To build the web interface.
* OpenCV: For processing webcam feeds and drawing on the canvas.
* Google Generative AI: For analyzing and generating content.
