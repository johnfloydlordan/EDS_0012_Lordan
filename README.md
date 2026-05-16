# \# Name: John Floyd M. Lordan

# \# Course \& Section: BSECE-1A

# \# Topic ID: AGR-03 – NPK Synergy Ratios

# 

# \---

# 

# \# Automated Soil Nutrient Analytics and Statistical NPK Synergy Classification Using Python Pipelines

# 

# \## Project Description

# 

# This project implements a Python-based smart agricultural engineering analytics pipeline for analyzing NPK synergy ratios and crop nutrient behavior using agricultural telemetry datasets. The system processes agricultural telemetry data to evaluate nutrient interaction patterns, identify crop-related nutrient relationships, perform clustering analysis, and generate statistical and graphical engineering outputs.

# 

# The proposed analytics pipeline automates:

# 

# \- Raw agricultural telemetry ingestion

# \- Dataset preprocessing and cleaning

# \- Statistical computation and analysis

# \- Correlation mapping

# \- K-means clustering analysis

# \- Comparative nutrient analysis

# \- Static scientific visualizations

# \- Animated agricultural telemetry visualizations

# \- Outlier detection using Z-score analysis

# 

# \---

# 

# \# Dataset

# 

# Crop Recommendation Dataset

# Source: Kaggle

# 

# \---

# 

# \# Features

# 

# \- Automated data cleaning

# \- Duplicate and null-value removal

# \- Feature engineering

# \- NPK ratio computation

# \- NumPy statistical analytics

# \- Correlation heatmap analysis

# \- K-means clustering analysis

# \- Comparative nutrient analysis

# \- Static visualizations

# \- Animated telemetry visualizations

# \- Outlier detection using Z-score analysis

# \- Agricultural filtering logic for dataset uniqueness

# 

# \---

# 

# \# Unique Filter Logic

# 

# To ensure mathematical uniqueness of the agricultural telemetry dataset, the analytics pipeline applied the following filtering condition:

# 

# ```python

# df = df\[

# &#x20;   (df\['humidity'] > 60)

# ]

# ```

