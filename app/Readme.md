[FedTS Simulation Demo](simulation.png)

# 🧠 Federated Learning with Fed-TS (Flask Dashboard)

A beginner-friendly interactive simulation of Federated Learning using **Thompson Sampling (Fed-TS)** to dynamically select trustworthy clients and block malicious ones.

---

## 📌 Project Title

**Dynamic Federated Learning using Fed-TS with Client Trust Evaluation via Flask Dashboard**

---

## 🌍 Tech Stack

- **Programming Language**: Python 3
- **Libraries Used**:
  - [Flask](https://flask.palletsprojects.com/) → for the web dashboard
  - [PyTorch](https://pytorch.org/) → to define and train the neural network
  - [TorchVision](https://pytorch.org/vision/stable/index.html) → for loading the MNIST dataset
  - [scikit-learn](https://scikit-learn.org/) → for KMeans clustering
  - [NumPy](https://numpy.org/) → for math and array operations
  - [Matplotlib](https://matplotlib.org/) (optional) → for plotting results

---

## 🔍 Problem Statement

In traditional **Federated Learning (FL)**:
- Clients are selected randomly
- Some clients may contain noisy, mislabeled, or malicious data

❗ This can degrade the global model’s performance.

### ✅ Solution: Fed-TS (Thompson Sampling)
We introduce a smarter way to select clients by:
- Tracking **client reliability** using Bayesian sampling
- Preferring clients with **low drift** (minimal disruption to global model)
- **Automatically blacklisting** clients after repeated suspicious behavior

---

## 🧠 Project Structure (Code Walkthrough)

### 1. **Setup**
We initialize a simple Flask app:
```python
app = Flask(__name__)
```
### 2. **Configuration**
Define how many clients and which ones are attackers:
```python
NUM_CLIENTS = 10
LABEL_FLIP_ATTACK = [5, 9]
```
### 3. **Dataset Preparation**
->Use MNIST (handwritten digits 0–9)

->Each client receives 600 samples (two digit classes)

->Attacking clients flip labels intentionally

### 4. **Model Definition**
A basic 2-layer neural network:
```python
class SimpleNN(nn.Module):
```

### 5. **Drift Measurement**
We calculate how much a client changes the global model:
```python
get_path_drift(model_before, model_after)
```

### 6. **Fed-TS Client Selection**
We use Thompson Sampling to:

Score clients (success/failure)

Prefer good performers

Block those with frequent failures

### 7. **Web Dashboard Routes**
/ → Home page (HTML)

/train_one_round → Run 1 round of training

/history → View training log & blocked clients

🚀 ***Final Output***
✅ A working interactive Flask dashboard

🔁 Simulate FL training one round at a time

⚠️ Identifies and blocks bad/malicious clients

📊 Tracks accuracy and performance dynamically

✨ ***Why This Project is Cool (For Beginners)***
✅ Simulates real-world federated learning logic

🔍 Implements trust-based client selection (Fed-TS)

🖥️ Visualizes training and blocking in a dashboard

🔐 Promotes privacy-preserving, secure AI training

📈 ***Future Improvements***
📉 Add live plots (accuracy trend per round)

💾 Save model checkpoints

🎨 Switch from Flask to Streamlit for advanced UI

🧠 Use more complex datasets (e.g., CIFAR-10)

🧪 **How to Run**
Clone this repo:
```python
bash
git clone https://github.com/yourusername/fedts-fl-dashboard.git
cd fedts-fl-dashboard
```

Install dependencies:
```python
bash
pip install flask torch torchvision scikit-learn matplotlib
```

Run the app:
```python
bash
python fedts_dashboard_fl.py
```
Open your browser:

arduino
http://localhost:5000

[Simulation Screenshot](simulation.png)
