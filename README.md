# Poker.AI (Reinforcement Based Poker Robot)
This project was done in collaboration with By Paul Pan, Sruthi Papanasa, Advaith Ravishankar, Vincent Tu with the Data Science Student Soceity at UCSD

Work Splits:
1. Game Environment: Paul Pan and Sruthi Papansa
2. RL Desogn and Training Model: Paul Pan, Advaith Ravishankar, Vincent Tu
3. GUI Desgin and Creation: Advaith Ravishankar

# Running the Game with the GUI

This repository contains a Poker environment which has a trained RL robot as the robots made usng stable.baseline3. To use the implementation, install the necessary libraries stated in ```requirements.txt``` using ```pip install -r requirements.txt```.

To play the game with the functional GUI, run ```pyhton poker_env.py``` in the terminal. The following window will open up which is made in Tkinter:

<p align ="center">
  <img src="./statics/homescreen.png">
</p>

Play the game as you like. You will be Playing against Bots created by our team with additional suggestions created by our bot.

## RL Algorithm

We created a model based on stable.baseline3's 
