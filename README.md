# Deep-Learning-Assignment

**Name:** Gayathri G  
**USN:** 1CD22AI020  
**Subject:** Deep Learning and Reinforcement Learning  
**Subject Code:** BAI701  

---

## 📌 Assignment Overview

This repository contains **optimized and improved implementations** of various **Deep Learning and Reinforcement Learning models**.  
The original programs were modified to improve **performance**, **learning stability**, and **practical usability**, especially for **limited hardware environments**.

All modifications focus on:
- Reducing computational complexity  
- Improving learning behavior and convergence  
- Producing clearer and more meaningful results  

---

## 🎯 Objectives of the Modifications

- Optimize model architectures for faster execution  
- Improve learning stability  
- Enhance output quality (text generation, gameplay, forecasting)  

---

## 📂 File-wise Description of Changes

### 1️⃣ AlexNet.py

**Original Version:**  
A heavy AlexNet architecture with large convolution filters and very large fully connected layers, mainly designed for large-scale datasets.

**Modification Purpose:**  
To reduce model size, improve execution speed, and make the network suitable for limited hardware environments.

**Effectiveness:**  
- Faster training and inference  
- Reduced memory consumption  
- More stable training with a custom learning rate  

✅ **Result:** Optimized and train-ready CNN model.

---

### 2️⃣ TicTacToe.py

**Original Version:**  
A basic reinforcement learning Tic-Tac-Toe implementation with weak rewards, high randomness, and a minor initialization issue.

**Modification Purpose:**  
To improve learning speed, strategic decision-making, and overall training stability.

**Effectiveness:**  
- Faster convergence  
- Smarter and more consistent gameplay  
- Stable execution without errors  

✅ **Result:** Improved reinforcement learning gameplay agent.

---

### 3️⃣ RNN.py

**Original Version:**  
A simple RNN model using short sequences, ReLU activation, and deterministic text generation, leading to repetitive output.

**Modification Purpose:**  
To enhance sequence learning, training stability, and creativity in generated text.

**Effectiveness:**  
- Less repetitive and more natural text  
- Improved stability using `tanh` activation  
- More creative output with temperature sampling  

✅ **Result:** Enhanced and higher-quality text generation model.

---

### 4️⃣ DeepReinforcementLearning.py

**Original Version:**  
A basic Q-learning model with sparse rewards and unstable learning behavior.

**Modification Purpose:**  
To improve convergence stability, learning efficiency, and path optimality.

**Effectiveness:**  
- Faster goal achievement  
- Smoother and more interpretable learning curve  
- Consistent optimal paths  

✅ **Result:** Stable and efficient Q-learning implementation.

---

### 5️⃣ LSTM.py

**Original Version:**  
A single-step LSTM model predicting only the next time-step value.

**Modification Purpose:**  
To enable multi-step future prediction for realistic time-series forecasting.

**Effectiveness:**  
- Multi-step forecasting capability  
- More useful and practical predictions  
- Improved training stability  

✅ **Result:** Advanced and real-world applicable forecasting model.

---

## 🧪 Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Deep Learning  
- Reinforcement Learning  

---



