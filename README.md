# Project Overview

In this second part of the project I aim to merge the learned four policies each aimed at a different task into one big policy.<br>
First I generated a state action pair dataset for each environment, then I used a Multi-task Learning (MTL) Convolutional Neural Network (CNN).<br>

## Project Diagram
<p align="center">
    <img src="images/Generalization in RL.png" style="width: auto; max-width: 100%; height: auto;">
  </a>
    <br>
  <em>Whole diagram
</p>

## Dataset Generation
Using the previously trained agents via Ray RLlib, I generated the state action pairs for each environment, each of which is not less than 800,000 instances.<br>
I stored each of those in a .JSON file, but unfortunately due to the files' sizes I'm not able to share them through my repository.<br>

I create 16 different combinations that corresponds to the whole enivronment's dataset, for more MTL adaptability, also, since the hardware resources are very limited I was able to only use a small portion of 
the dataset (200,000 x 16).<br>

### Data processing
Each environments state-space dimension is different therefore, I padded the state-spaces of the short ones to match it unify them all, The longest state-space was of length 6.<br> 
For more accuracte modeling, I used the last 13 states in order to predict the action, resulting now in a (6 x 13) for each environment. 
then these are stacked to form a 3D input which will be processed via a 2D CNN.<br>

### Cross-Validation
due to the sixze of the dataset I opt for 95% train 5% test set.<br>

## MTL CNN Architecture
Since CNNs are proving to be great for sequential based models modeling, I made use of it. 
and to learn the actions which will be correspondant to each environmnet, to overcome this I used a Multi-task learning Net instead of 
a conventional Fully connect layers with a single head output.<br>
