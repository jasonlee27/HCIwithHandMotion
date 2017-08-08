# HCIwithHandMotion
Human Computer Interface with Hand Motions

# Human-Computer Recognition using Gestures
## abstract
Human-Computer Interaction (HCI) exists ubiquitously in our daily lives. It is usually achieved by using a physical controller such as a mouse, keyboard or touch screen. It hinders Natural User Interface (NUI) as there is a strong barrier between the user and computer. This report mainly discusses the design and implementation of hand gesture recognition and its four different applications which are Pacman game using optical flow vectors, virtual Drawing, Computer control and Gesture classification. We made a dedicated MFC Hand gesture recognition application on Visual Studio 2013 which means that it can run on any Windows platform with required libraries and input images. The main aim of this application is to make human computer interaction more efficient and easy in terms of usage since gestures are an important aspect of human-interaction, both interpersonally and in the context of man-machine interfaces.

## Introduction
The discipline of human-computer interaction is studied to determine how to make computer technology more interactive for efficient human use. In the past years, computers used to very expensive and were almost used by technical people, but now, they are not so expensive and the majority of people using them are non-technical. Thus it is very important to make human-computer interaction more user-friendly. This is the motivation behind our project.

## Application
1. Pacman game: The pacman game is demonstrated. the pacman moves with subject movement such as hand and body using the iterative Lucas-Kanade method with pyramid. The pacman is to eat apples moving within the camera frame and subjects are competing with others with time comsumption for eating all apples. It is noted that control of pacman tracking is convenient and fast to play pacman game, so subjects enjoy the game.

2. Virtual drawing: We demonstrated virtual drawing, which is a platform for drawing with a certain color on the whiteboard and frame images. In frame image, it has buttons for selecting color(red, green and blue), erasing drawings and clearing all drawings. In addition, one finger is for cliking down selection and drawing and two fingers is for clicking up. It is noted that it is convinient and fast to draw and it is also easy to draw and erase.

3. Virtual computer control: We demonstrated computer control system with hand gesture. Similar with the virtual drawing, one finger is for clicking up and moving the mouse cursor, and two fingers is for clicking down. In addition, we can zoom-in and out with moving up and down with open hand, respectively. we search google maps, which is a web mapping sevice developed by Google with the function. It is noted that we obtained high accuracy of searching places that we are interested in on the Google Maps.

4. Hand gesture classification: We demonstrated Hand gesture classification. we assign 8 hand gestures, which are one to five finger up, open and close hand, and okay sign. It is noted that we obtained high accuracy of classifying gestures. In addition, we split arm area and hand area so that it function more robust.
