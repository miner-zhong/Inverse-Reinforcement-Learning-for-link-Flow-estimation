# Inverse-Reinforcement-Learning-for-link-Flow-estimation-IRL-F-
# Background
This repository includes source codes for IRL-F and CRL-F, which are proposed to estimate link flows on a road network by combining traffic volume data and vehicle trajectory data. Please refer to [PAPER]

# Problem Definition
Estimate link flows for a given time interval from limited traffic volume data and sparse trajectory data with minimal assumptions and requirements on the available data. Specifically, the following challenges are addressed:
•	Traffic volume data capture the traffic flow of the whole population at observation points. However, the spatial coverage of the available volume data is limited because only a small subset of links has detectors installed.
•	Trajectory data capture the vehicles’ route preferences and movement patterns across the network. However, the available trajectory data are sparse in that they only represent a sample of the whole population. The percentage of vehicle trajectories being captured in the observed trajectory set is unknown. The distribution of such sampling rates across the network is not uniform. Also, some paths and links on the network may not be covered by the observed trajectories.

![](images/figure1.png)

# Solutions
The following two methods have been proposed to solve this problem. 
•	Inverse Reinforcement Learning for link Flow estimation (IRL-F) 
•	Constrained Reinforcement Learning for link Flow estimation (CRL-F).
These two methods can be evaluated in the Nguyen-Dupuis network. There are 13 nodes, 38 links and 18 OD pairs. OD flow data are available in Castillo et al. (2008). Traffic assignment problem has been solved to obtain path flows and link flows, based on which the observed traffic volume data and trajectory data are generated.

![](images/figure2.png)
Input data for IRL-F and CRL-F includes:
•	Network property files, which indicate the number of states, the number of actions, and the dimension of the first state feature vector.
•	Transition information, which indicates the actions that the agent needs to take to transition for one state to another state. 
•	State feature files, which include the feature vector of each state in the road network MDP. 
•	Traffic volume data input, which includes the observed traffic volume data.
•	Trajectory data input, which includes the observed trajectory data, and the scaled-up trajectory data that can later be aggregated to provide an estimated population flow number. 

# How to apply IRL-F
Set input data paths and parameters in arg.py. 
Run IRL-F in main.py
Output of IRL-F is a csv file including the final estimated state visitation frequencies. These frequencies can be used to estimate unobserved link flows following simple procedures.

# How to apply CRL-F
Set input data paths and parameters in arg.py. 
Run CRL-F in main.py
Output of CRL-F is a csv file including the final estimated state visitation frequencies. These frequencies can be used to estimate unobserved link flows following simple procedures.
