

# 🧠 ST-Mamba Architecture Explained

### *Spatio-Temporal Selective State Space Models for Traffic Prediction*

## 📖 Overview

This document explains the **ST-Mamba** architecture used in **Project Newton**. This model is designed to handle complex data that has both **Spatial Structure** (Road Networks) and **Temporal Evolution** (Traffic Flow over time).

Unlike standard Mamba which only reads sequences (Time), ST-Mamba integrates Graph Neural Networks (GNNs) to understand the map.

---

## 🏗️ The Data Structure: The 4D Tensor

The input to the model is not a simple table or image. It is a **4-Dimensional Tensor** representing a batch of simulations.

### The Hierarchy

> **Shape:** `[Batch, Time, Nodes, Features]`

| Dimension | Name | Analogy | Description |
| --- | --- | --- | --- |
| **Dim 0** | **Batch** | The Drawers | Parallel simulations running at once (e.g., 32 separate cities). |
| **Dim 1** | **Time** | The Folders | The sequence history (e.g., 12 seconds of traffic snapshots). |
| **Dim 2** | **Nodes** | The Papers | The locations on the map (e.g., 100 intersections). |
| **Dim 3** | **Features** | The Content | The data at each location (e.g., Speed, Car Count, Rain). |

### 🧠 Mental Model: "The File Cabinet"

To visualize the data, imagine a **File Cabinet**:

1. **Batch:** The separate **Drawers** (Simulation A, Simulation B...).
2. **Time:** The ordered **Folders** inside a drawer (Time 01, Time 02...).
3. **Nodes:** The individual **Pages** inside a folder (Page for Times Square, Page for Central Park...).
4. **Features:** The **Text** written on each page (Speed: 45mph).

---
 ## The Architecture Pipeline

The model uses a "Divide and Conquer" strategy. We cannot feed raw 4D data into Mamba, so we split the job into two specialists.
Code snippet

graph LR
   ##  A["Input Tensor: 4D"] --> B["Spatial Specialist (GAT)"]
   ## B --> C["Dimension Flattening"]
   ##C --> D["Temporal Specialist (Mamba)"]
   ##D --> E["Prediction"]


### Phase 1: Spatial Specialist (GAT)

* **Input:** `[Nodes, Features]` (One snapshot in time).
* **The Job:** Connects the dots on the map.
* **Mechanism:** **Message Passing**. Node A sends its features to Node B. Node B realizes "My neighbor is congested, so I will be soon."
* **Output:** A "Spatially Aware" embedding for every node.

### Phase 2: Dimension Flattening (The Bridge)

* **The Problem:** Mamba expects a 1D sequence, not a graph.
* **The Fix:** We flatten or pool the `Nodes` dimension.
* **Result:** A sequence of "City-State Vectors" representing the entire map at each timestamp.

### Phase 3: Temporal Specialist (Mamba)

* **Input:** `[Time, Hidden_Dim]` (Sequence of map states).
* **The Job:** Connects the past to the future.
* **Mechanism:** **Selective State Space Model (SSM)**. It scans the sequence `T=1 -> T=2 -> T=3` and updates a hidden memory state.
* **Output:** Predicts the traffic state at `T+1`.

---

## 🚀 Why ST-Mamba? (Vs Transformers)

| Feature | ST-Transformer (Attention) | ST-Mamba (SSM) |
| --- | --- | --- |
| **Complexity** |  (Quadratic) | ** (Linear)** |
| **Memory** | Explodes with long history | **Constant Memory** |
| **Inference** | Slow (Re-scans history) | **Fast (Recurrent State)** |
| **Best For** | Short sequences, small graphs | **Long history, massive graphs** |

## 📝 Implementation Note

In `NewtonGraphMamba`, the layers are typically arranged as:

1. **Encoder:** Graph Encoder (GAT/GCN) to process spatial features.
2. **Middle:** `nn.Linear` to project spatial embeddings to Mamba dimension.
3. **Decoder:** Mamba Blocks to process the temporal sequence.
