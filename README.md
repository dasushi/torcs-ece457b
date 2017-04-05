# torcs-ece457b

CONTENTS
----------------

  * Intro
  * Requirements
  * Installing
  * Configuring
  * Troubleshooting

INTRO
--------------
TORCS Actor-Critic Deep Deterministic Gradient Policy driver for ece457b

Check it out! https://www.youtube.com/edit?o=U&video\_id=zoH7e3BybCE

This is a driver that uses Google's Deep Deterministic Gradient Policy through Tensorflow and Keras to drive in TORCS via gym\_torcs and openai-gym, with vTorcsRL

REQUIREMENTS
----------------
  
  - OpenAI-Gym
  - Tensorflow
  - Keras 2
  - Python 2.7
  - gym\_torcs
  - h5dpy, numpy, scipy, etc
  - xautomation
  - vtorcs-TL-color

INSTALLING
---------------


https://github.com/ugo-nama-kun/gym\_torcs
 
Install gym\_torcs somewhere and setup vTorcsRL, copy driver files from here to there

cp \*.py ../gym\_torcs

then run raceSteves.py, it will save to weights files in /gym\_torcs/steveActor/Critic.h5/json. Edit the "trainFlag" in raceSteves.py to disable training
