# Learning to Control - Developing Intelligent Systems
by _Minho Menezes_  

### Behold!

This repository contains all the material and source-codes used in the talk **"Learning to Control - Developing Intelligent Systems"**. 

This talk surveys the basic models and techniques used in Optimal Control and Reinforcement Learning to develop applications that controls continuous dynamical systems by means of numerical optimization. This is the science behind the recent breakthrough in Self-Driving Cars, Robotics, Artificial Intelligence in Video-Games, and so on.

The presentation starts by contextualizing the subject in the perspectives of Control Theory and Machine Learning, stating it as a common problem and unifying the notation. Next, two dynamical systems are presented as the benchmark for the methods to be presented. The main subject of designing an intelligent controller is then divided into two approaches: the _Model-Based Approach_ and the _Model-Free Approach_. The first deals with the setting of having general knowledge about how the system behaves, so we can accurately predict its future state by assuming a set of inputs signals. The agent can infer its actions before even applying then, making it a problem of _planning_. The second approach deals with the setting of not knowing (or not caring to know) the system dynamics, so we can only observe the past taken actions, states and reward at each step (making what is known as a _trajectory_). The agent can only infer its next action in real-time, based on past data, in what is known as _online learning_, the core of reinforcement learning.

Finally, we explore both the approaches for the proposed systems and discuss how one can benefit from another in the goal of designing intelligent systems that can act in highly uncertain environments with high levels of autonomy.
 
In case of any doubts, suggestions, or if you want to send me dog pictures, please contact: [minhotmog@gmail.com](#mail-to:minhotmog@gmail.com)

Ps.: This talk is heavily inspired by the ICML 2018 Tutorial from Benjamin Recht. You should [check it out](https://people.eecs.berkeley.edu/~brecht/l2c-icml2018/)!

Bons estudos! :)