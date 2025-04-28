# Final_submission
This project aims to control a Tello drone using real-time EEG data streamed from an Emotiv EPOC+ headset. The EEG data, along with gyroscope data, is used to train a Reinforcement Learning (RL) agent that learns to map brain signals to tello drone control commands.


# EEG-Driven Drone Control and Visualization System

## Introduction

This project is a real-time brain-computer interface (BCI) system that connects an Emotiv EEG headset to a drone control agent using LSTM-based sequence modeling and reinforcement learning (RL). It streams EEG signals, extracts rich features, predicts user intent, and maps the predictions to drone control actions while visualizing brain activity in real-time.

### Objectives

- Stream raw EEG + gyro data in real-time.
- Preprocess EEG signals using advanced filtering and denoising.
- Extract physiologically meaningful features.
- Train an LSTM model on sequences of EEG feature vectors.
- Control a drone using reinforcement learning decisions based on brain states.

## Features

### Real-Time EEG Data Acquisition

Connects to Emotiv headsets, decrypts EEG packets, and streams real-time brain activity and gyroscopic data.

### Signal Preprocessing Pipeline

- Bandpass and Notch filtering to clean EEG signals.
- Common Average Referencing (CAR) and Adaptive Noise Cancellation (ANC).
- Artifact removal via Independent Component Analysis (ICA) and Discrete Wavelet Transform (DWT) denoising.

### Feature Extraction

- Frequency Band Power (Delta, Theta, Alpha, Beta, Gamma).
- Hjorth Mobility and Complexity Parameters.
- Spectral Entropy.
- Higuchi Fractal Dimensions.
- First-Order and Second-Order Temporal Derivatives.

### Deep Learning

An LSTM-based model processes sequences of extracted EEG features. Predicts both discrete action classes and continuous control values.

### Reinforcement Learning

A LinUCB Contextual Bandit agent explores and exploits EEG-driven control policies. Continuous online learning through reward feedback and minimal sample requirements.

### Drone Control Environment

Interfaces with DJI Tello drones or simulated drone behavior. Human-in-the-loop feedback after each action to reinforce learning.

### Data Management and Logging

Continuous data saving of raw and processed EEG and gyro measurements into Excel files. Logging of model parameters, rewards, and system metrics at runtime. Full shutdown recovery via custom signal handlers.

### Real-time Visualization

2D and 3D plots of EEG feature activity and head motion.

### Clean Thread Management and Shutdown Handling

## System Thread Architecture

The system follows a carefully organized multithreaded architecture ensuring smooth real-time operation.

| Thread | Purpose | Data Passed |
|:---|:---|:---|
| streaming_thread | Streams real-time EEG+gyro packets from Emotiv device | Raw data packets → data_queue, visualization_queue |
| preprocessing_thread | Buffers EEG data, extracts features, makes LSTM + RL predictions | Features extracted from packets |
| save_data_continuously | Saves raw and processed data batches to Excel | Local data store |
| (Main Visualization Thread) | Renders EEG visualizations in real-time | Visualization queues and events |

### Flow of Data

- `stream_data.py` (EmotivStreamer) reads EEG + gyro → sends into primary_buffer.
- Primary buffer fills → 1s window of 14 EEG channels (256 samples each) → moved to secondary_buffer.
- Secondary buffer → undergoes preprocessing → features extracted (static + dynamic features).
- Feature vector (shape: 10, 10702) → accumulated for 10 seconds into a feature window.
- 10-seconds feature sequence → fed into LSTM model.
- LSTM output → passed into RL agent → selects drone control action.
- Each piece only moves to the next thread if fully prepared.

## Installation & Setup

### Requirements

- Python 3.8+
- numpy
- pandas
- openpyxl
- scikit-learn
- tensorflow
- matplotlib
- hidapi
- pycryptodome
- scipy
- pywavelets
- psutil
- gym

```bash
pip install -r requirements.txt
```

### Hardware

- Emotiv EEG headset connected via USB/Bluetooth
- (Optional) Tello drone for action mapping demonstration

## Usage

### Running the Project

```bash
python main.py --connect-drone
```

or if drone not needed for testing:

```bash
python main.py
```

### How it Works

Launch `main.py`, which:

- Connects to the EEG device
- Starts all necessary threads
- Begins streaming, saving, preprocessing, and visualization

During execution:

- EEG and gyroscope data are streamed and preprocessed.
- Feature vectors are extracted per second and grouped into sequences.
- The LSTM predicts control probabilities based on the feature sequences.
- The bandit agent selects an action based on LSTM output.
- Human feedback is collected.
- Data is continuously saved for later analysis.

Upon interruption:

- Saves current model parameters.
- Exports extracted features and raw EEG data to Excel.
- Shuts down all threads gracefully.

## Feature Extraction and Preprocessing

### How Preprocessing Thread Creates a Feature Vector

1. Raw Data Streaming
2. Buffer Management
3. Preprocessing Steps
4. Feature Extraction

### Explanation of Each Feature (Physiological Meaning)

| Feature | What it Represents | Why Important |
|:---|:---|:---|
| Band Power | Relative energy in known brain rhythms | Brain states reflected by changes |
| Hjorth Mobility | Signal smoothness vs roughness | High mobility = more brain activity |
| Hjorth Complexity | Change rate of mobility | Detects quick shifts in mental states |
| Spectral Entropy | Spread of power across frequencies | Higher entropy = complex brain activity |
| Fractal Dimension | Signal self-similarity over time | Higher FD = intricate neural processes |
| Filtered EEG | Denoised brain signal | Base signal context |
| First Order Derivatives | Change rate of EEG | Captures transitions |
| Second Order Derivatives | Change acceleration of EEG | Identifies sudden transitions |

## stream_data.py: The CPU (Brain and Heart) of the System

Manages:

- Timing
- Preprocessing
- Feature windowing
- Real-time coordination

## Feature Vector Details

| Feature | Size | Description |
|:---|:---|:---|
| Band Power | 70 | Power in Delta, Theta, Alpha, Beta, Gamma |
| Hjorth Parameters | 28 | Mobility and Complexity |
| Spectral Entropy | 14 | Spectral complexity |
| Fractal Dimension | 14 | Higuchi fractal dimension |
| EEG Filtered Signal | 3584 | Denoised EEG signal |
| First Order Derivatives | 3584 | Changes in EEG signal |
| Second Order Derivatives | 3584 | Acceleration in EEG signal |

## Why an LSTM? (Motivation)

- EEG signals evolve over time.
- LSTM captures temporal dependencies.
- Smooths noise.
- Enables flexible and context-aware drone control.

## How the LSTM Model Is Designed

| Attribute | Design Choice | Purpose |
|:---|:---|:---|
| Input | (10, 10702) feature sequence | 10s of brain state |
| Architecture | 2-layer LSTM with 128 hidden units | Models time dependencies |
| Output | Discrete + continuous values | Action classification and fine control |
| Training | Binary Cross Entropy + MSE | Joint classification and regression |
| Activation Functions | Sigmoid, Tanh | Proper output bounding |

## How Data Flows Through the LSTM

- Input: 10-second feature window
- LSTM layers model evolution
- Output: Final hidden state
- Predict discrete action + continuous parameters

## Why This LSTM Setup Helps in BCI

- Captures intention evolution
- Robust to noise
- Enables nuanced control
- Efficient for real-time systems

## LSTM Learning and Prediction

- 10s sequence → LSTM → embedding
- Embedding → Bandit agent → action
- Raw EEG → Structured feature → Pattern understanding → Drone decision

## Reinforcement Learning Agent: Contextual Bandit for Decision-Making

### Why Use a Bandit Agent

- Fast online learning
- Works with low data
- Context-aware decisions

## How the Bandit Agent Works

- Input: LSTM output (5D)
- Model: LinUCB with ε-greedy
- Selection: UCB score maximization
- Learning: Human feedback update

## System Flow (Quick Overview)

| Component | Role |
|:---|:---|
| EmotivStreamer | Connect to EEG headset |
| Preprocessing Thread | Form features |
| Feature Extraction | Extract meaningful vectors |
| LSTMHandler | Predict intention |
| Bandit Agent | Choose best action |
| Drone Controller | Execute movements |
| Signal Handler | Safe shutdown and saves |

## Reward Shaping

- Positive: Action approval, successful maneuver
- Negative: Repetition, rejection, timeout

## File Structure

| File | Purpose |
|:---|:---|
| main.py | Main orchestrator |
| stream_data.py | EEG data handling |
| streaming_thread.py | Streaming EEG+gyro packets |
| preprocessing_thread.py | Feature extraction and prediction |
| data_saver.py | Save EEG and features |
| main_thread.py | Visualizations |
| feature_extraction.py | Feature engineering |
| feature_config.py | Constants storage |
| lstm_handler.py | LSTM model loading and running |
| lstm_model.py | Defines LSTM architecture |
| bandit_agent.py | Bandit agent logic |
| learning_rlagent.py | Drone Gym environment |
| model_utils.py | Model save/load utils |
| signal_handler.py | Shutdown management |
| shared_events.py | Thread-safe event management |

## Contribution

We welcome contributions!

### Commit Style

- Clear, descriptive messages
- Prefixes: [Fix], [Feature], [Docs]

### Pull Request Guidelines

- Fork → Branch → Code → Test → PR
- Describe changes clearly

## Author & License

- Author: Kushal Pagolu
- License: [MIT / Apache 2.0 / Specify]
