# Thompson-Sampling-Policy-for-Dynamic-Participating-Client-Scenario-in-Federated-Learning
Implementing this Paper.

🌐 Federated Learning (FL)

✅ What it is:

A type of machine learning where multiple decentralized devices (clients) train a model collaboratively without sharing raw data.

✅ Key Concepts:

Data is kept local on edge devices

Only model updates are sent to a central server

Ensures data privacy and security

✅ Example:

Smartphones collaboratively train a predictive keyboard without sending text data to the cloud (used in Gboard by Google).

Hospitals collaboratively train a medical model without sharing patient data.



______________________________________________________________________________________________________________________________________________

# 🤖 Why Thompson Sampling?

Problem: Malicious or unproductive clients degrade model performance if selected blindly.

Solution: Use Thompson Sampling, a Bayesian multi-armed bandit approach that:

● Scores each client based on success/failure (drift)

● Samples from Beta distribution to balance exploration vs exploitation

● Reduces participation of unreliable clients

