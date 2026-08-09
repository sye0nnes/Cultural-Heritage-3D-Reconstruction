# Cultural Heritage 3D Reconstruction

> **Work in progress** - A team capstone project for reconstructing Korean cultural heritage sites from real-world imagery.

## Overview

This project investigates a practical 3D reconstruction pipeline for documenting cultural heritage architecture. Images of the **Seooreung Jeongjagak heritage structure** are collected from complementary viewpoints using a drone and a quadruped robot, then processed with Gaussian Splatting-based reconstruction methods.

The broader goal is to evaluate how aerial and ground-level observations can be combined to produce a detailed, navigable 3D representation of a real cultural heritage site.

## System Pipeline

```text
Drone imagery + Quadruped-robot imagery
                    |
                    v
        Data organization and preprocessing
                    |
                    v
        Camera pose estimation / scene setup
                    |
                    v
        Gaussian Splatting training
                    |
                    v
        Novel-view rendering
                    |
                    v
        Reconstruction quality evaluation
```

## Project Components

### Data Acquisition

- Aerial imagery collected from a drone
- Ground-level imagery collected using a quadruped robot
- Multi-view observations designed to cover architectural details from complementary perspectives

### 3D Reconstruction

- Dataset preparation and image organization
- Gaussian Splatting-based scene reconstruction
- Training and rendering pipeline configuration
- Novel-view rendering for qualitative inspection

### Evaluation

The project considers both visual quality and reconstruction reliability. Evaluation work includes:

- Qualitative comparison of rendered views
- Inspection of missing or poorly reconstructed regions
- Analysis of viewpoint coverage
- Reconstruction quality assessment using available image-based metrics

## My Contributions

My work focuses on the reconstruction side of the project:

- Organizing and preparing captured image data for reconstruction
- Building and running the Gaussian Splatting training and rendering pipeline
- Inspecting reconstructed scenes and rendered novel views
- Supporting reconstruction quality evaluation and experiment documentation
- Coordinating the connection between data acquisition and 3D reconstruction stages

## Research Motivation

Cultural heritage reconstruction presents practical challenges that are less visible in curated benchmark datasets:

- Limited access to some viewpoints
- Occlusion around roofs, pillars, and narrow structural regions
- Different image characteristics between aerial and ground platforms
- The need to balance reconstruction quality with realistic data-collection constraints

This project provides hands-on experience with real-world 3D vision, multi-platform data acquisition, scene representation, and reconstruction evaluation.

## Current Status

The project is currently in progress. The team is iterating on:

- Image coverage and data quality
- Reconstruction and rendering settings
- Evaluation methodology
- Integration of drone and quadruped-robot observations

Representative results and additional technical details will be added after the experiments are finalized and approved for public release.

## Repository Scope

This is a **public portfolio repository** that summarizes the project and my contributions. The original team repositories, raw datasets, internal reports, and unpublished implementation details are not included because they are managed separately as part of the capstone project.

No private team code, raw cultural-heritage dataset, model checkpoints, or internal course materials are published here.

## Technologies

- 3D Gaussian Splatting
- Python
- PyTorch
- Computer Vision
- Multi-view 3D Reconstruction
- Drone and quadruped-robot data acquisition

## Project Context

- Type: Team capstone project
- Domain: 3D Vision / Cultural Heritage Digitization
- Status: Ongoing
# Cultural-Heritage-3D-Reconstruction
3D reconstruction of Korean cultural heritage using drone and quadruped-robot imagery with Gaussian Splatting.
