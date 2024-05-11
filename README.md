# Retrieval-Augmented Generation (RAG) System for LLMs

## Overview
This repository contains the implementation of a Retrieval-Augmented Generation (RAG) system designed to enhance the responses of Large Language Models (LLMs). By integrating cloud-deployed RAG with LLMs, our solution improves accuracy, contextual relevance, and informativeness of the model outputs in real-world applications.

## Features
- **Modular Architecture**: Combines LLMs with dynamic external data retrieval.
- **Enhanced Responses**: Increases the precision and relevance of model outputs.
- **Scalable Infrastructure**: Utilizes Docker and Kubernetes for robust deployment.

## Documentation
For more detailed information, refer to the report.

## To Run
On GKE, run 
kubectl apply -f deployment.yaml
kubectl get svc to see the External IP and port
Go to the url place of your browser and type:
For non-RAG LLM : <External IP>:<Port>/search_simple?query=<query>
For RAG LLM : <External IP>:<Port>/search?query=<query>


