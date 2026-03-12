# Simplified Property Task Generator

## Overview

This tool generates tasks based on **simplified properties** and an **appended neural network**.  
Given a neural network and an arbitrary property, the tool produces:

- A **simplified version of the original property**
- A corresponding **neural network with the property appended**

The goal is to reduce complexity while preserving the semantics of the original task, enabling more efficient analysis, verification, or downstream processing.

## Key Features

- Accepts arbitrary properties and neural network models
- Automatically simplifies properties
- Appends the simplified property to the given neural network
- Produces outputs suitable for further verification or experimentation

## Input

- **Neural Network**: The original network model
- **Property**: An arbitrary property defined over the network

## Output

- **Simplified Property**: A reduced and optimized version of the input property
- **Appended Network**: A modified neural network incorporating the simplified property

## Use Cases

- Neural network verification
- Property simplification for analysis
- Research and experimentation with property-aware networks

## Notes

This tool is designed to maintain the logical intent of the original property while reducing its complexity, making it easier to work with large or complex neural networks.

---
