# LiDAR Point Cloud Classification Project

## Overview

This project focuses on developing a LiDAR point cloud processing and classification pipeline using modern computer vision and AI techniques.

The primary goal is to:
- Load large-scale LiDAR point cloud datasets
- Convert 3D point cloud data into 2D projections
- Apply AI/computer vision models for object classification
- Map the classified results back into 3D space

The project is intended as a research-oriented prototype and proof-of-concept for intelligent LiDAR scene understanding in civil engineering and infrastructure applications.

---

# Research Motivation

LiDAR (Light Detection and Ranging) generates extremely dense 3D spatial datasets representing real-world environments.

Typical LiDAR scans may contain:
- Roads
- Buildings
- Pavements
- Terrain
- Vegetation
- Infrastructure objects

However, raw point cloud data is difficult to process directly due to:
- Massive data size
- Unstructured geometry
- Complex spatial relationships

The project explores a simplified but scalable approach:

text 3D Point Cloud       ↓ 2D Projection       ↓ AI-Based Image Classification       ↓ Projection Back Into 3D 

Instead of directly classifying 3D geometry, the project leverages mature 2D computer vision models and transfers the results back into 3D space.

---

# Core Project Idea

The pipeline follows these steps:

1. Load LiDAR point cloud data
2. Visualize and inspect the point cloud
3. Generate 2D orthographic projections
4. Run AI/computer vision models on the 2D image
5. Obtain segmentation masks/class labels
6. Reproject the classified regions back into 3D
7. Label the original point cloud points

The goal is to create a minimum working example before expanding toward a larger platform.



## Focus on a Minimum Working Example

Initial deliverables should focus on:
- Loading point clouds
- Generating 2D projections
- Detecting simple objects such as roads/buildings
- Mapping classifications back into 3D

A working proof-of-concept is more important than a polished product.

---

## Use Existing Libraries

The project intentionally leverages:
- Open3D
- PDAL
- CloudCompare
- Potree
- OpenCV
- YOLO
- Local multimodal LLMs

instead of reinventing existing tools.

---

# Current Development Goals

Current development focuses on:

- Understanding point cloud structures
- Loading .las LiDAR datasets
- Efficient handling of large-scale point clouds
- Visualization using Open3D
- Chunked processing for memory safety

---

# Dataset Information

Current dataset:

text UPark_Merged_PS_NAD83_G18_USFT_las.las 

Dataset statistics:

- Size: ~16 GB
- Total points: ~411 million points

This is a research-scale LiDAR dataset and requires chunk-based processing instead of full-memory loading.

---

# Technologies Used

## Python Libraries

### Open3D
Used for:
- Point cloud representation
- Visualization
- Geometry operations

### laspy
Used for:
- Reading .las LiDAR files
- Chunk-based LiDAR access

### NumPy
Used for:
- Numerical processing
- Point cloud array manipulation

---

# Current Visualization Pipeline

The current implementation:
- Opens the LAS file
- Loads a chunk of points
- Converts coordinates into Open3D format
- Displays the point cloud


---

# Installation

## Create Virtual Environment

bash python3.11 -m venv .venv source .venv/bin/activate 

---

## Install Dependencies

bash pip install open3d laspy numpy 

---

# Running the Project

bash python app.py 

---

# Current Learning Objectives

The current focus is understanding:

- Point cloud geometry
- LiDAR coordinate systems
- Spatial density
- Orthographic projections
- Visualization pipelines
- Large-scale dataset handling

---

# Future Roadmap

## Phase 1 — Point Cloud Understanding
- Load and inspect LiDAR datasets
- Visualize geometry
- Learn point cloud structures

## Phase 2 — Projection Generation
- Generate 2D orthographic views
- Create camera-based projections

## Phase 3 — AI Classification
- Apply object detection/segmentation
- Detect roads, buildings, pavements

## Phase 4 — 3D Reprojection
- Map classified pixels back into 3D
- Label original point cloud points

## Phase 5 — Interface and Automation
- Build interactive tools
- Possibly create a lightweight web interface

---

# Potential AI Models

Possible models/tools under consideration:

- YOLO
- OpenCV segmentation
- Llama multimodal models
- Local open-source multimodal LLMs

The current preference is toward:
- locally executable models
- minimal API dependency
- lightweight experimental workflows

---

# Important Notes

## Large Dataset Warning

This dataset contains hundreds of millions of points.

Loading the entire dataset simultaneously may:
- exhaust RAM
- freeze the system
- cause severe slowdown

Chunked processing and downsampling are strongly recommended.

---

# Long-Term Research Direction

This project has applications in:
- Civil engineering
- Infrastructure analysis
- Autonomous systems
- Robotics
- GIS
- Smart cities
- Pavement analysis
- Road extraction
- Scene understanding

The long-term objective is to develop scalable AI-assisted LiDAR analysis workflows.

---

# Author Notes

This repository is currently in the exploratory/research phase.

The emphasis is on:
- understanding LiDAR workflows
- validating projection-based classification approaches
- building a robust conceptual foundation before optimization

---