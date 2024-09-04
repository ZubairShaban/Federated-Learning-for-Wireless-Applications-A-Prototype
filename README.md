# Federated-Learning-for-Wireless-Applications-A-Prototype
Repository are composed of 3 python files:
1) client1.py
   a. Code contains client1 model training and parameter generation(Weights and Bias) on a subset of MNIST dataset. Passes this information to the USRP's.
   b. Read averaged parameters(client1 and client2) received using USRP and updates the local model.
   c. Lost data packets will be replaced by 0's.
2) client2.py
   a. Same as client1.
3) server.py
   Model aggregator for client1 and client2. Transmits aggregated model to both the clients.
Syncronization between clients and server are being maintained using Zigbee devices.
Correct the file path for python codes in Labview code and file path for python codes.

Open LabView, and connect the USRP's mounted with antennas and Xbee devices properly and run btp_after_zigbee_serial.vi

