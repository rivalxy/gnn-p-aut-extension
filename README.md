# Bachelor Thesis: Graph neural networks and extensibility of partial graph automorphisms

This repository contains the code, experiments, and validation statistics for my bachelor thesis, which focuses on training a Graph Neural Network (GNN) to determine whether a partial automorphism of a graph can be extended to a full automorphism.

## Overview

The project uses a dataset of graphs with high automorphism group sizes, generates partial automorphisms, and trains a GNN classifier to predict extendibility. The goal is to explore the learnability of graph symmetries and the structural patterns that guide automorphism extension.

## Repository Structure

* **/dataset/** - Graph datasets and generated partial automorphisms.
* **/kaggle/** - Notebooks used for training on kaggle.com
* **/results/** - Optuna search history and training history along with the best model weights saved as pytorch file.

## Thesis Context

The goal is to investigate potential applications of graph neural networks (GNNs) to problems in algebraic graph theory. The student will have the opportunity to engage with the state-of-the-art research in machine learning and algebraic graph theory and contribute to the field by generating new record graphs and training GNNs to predict algebraic properties.
