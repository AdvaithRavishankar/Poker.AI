# Poker.AI (Reinforcement Based Poker Robot)
This project was done in collaboration with By Paul Pan, Sruthi Papanasa, Advaith Ravishankar, Vincent Tu with the Data Science Student Soceity at UCSD

Work Splits:
1. Game Environment: Paul Pan and Sruthi Papansa
2. RL Design and Training Model: Paul Pan, Advaith Ravishankar, Vincent Tu
3. GUI Desgin and Creation: Advaith Ravishankar

# Running the Game with the GUI

This repository contains a Poker environment which has a trained RL robot as the robots made usng stable.baseline3. To use the implementation, install the necessary libraries stated in ```requirements.txt``` using ```pip install -r requirements.txt```.

To play the game with the functional GUI, run ```pyhton poker_env.py``` in the terminal. The following window will open up which is made in Tkinter:

<p align ="center">
  <img src="./statics/homescreen.png">
</p>

Play the game as you like. You will be Playing against Bots created by our team with additional suggestions created by our bot.

## RL Algorithm

We created a model based on stable.baseline3's [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) training which requires an observation space to track to develop the decision making process.

<p align ="center">
  <img src="./statics/ppo.png">
</p>

Observation Space:
1. "actions": All Player Actions
2. "active":  Players which are actvie
3. "chips":   chip total for all players
4. "community_cards": 5 Cards on the table
5. "player_card":  Player card combinations
6. "max_raise":   Total amount player can raise
7. "min_raise":   Minmimum amount player can raise
8. "pot": current size of game bettings
9. "player_stacks": pot_commits for every player in the whole game
10. "stage_bettings": pot_commits for every player in the current stage
11. "hand_score": Player hand's score

After training for ober 8 million games, the performance measured up to:

<p align ="center">
  <img src="./statics/">
</p>
