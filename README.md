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
Follow these steps to deploy and access the application on Google Kubernetes Engine:

### 1. Apply the Deployment Configuration
- Open your terminal.
- Run the following command to apply the `deployment.yaml` configuration:
- kubectl apply -f deployment.yaml


### 2. Retrieve the External IP and Port
- To find out the external IP address and port, execute:
    kubectl get svc
- Note down the External IP and the port number provided.

### 3. Access the Application
- Open a web browser of your choice.
- Enter the URL in the address bar based on the model you are using:
- For non-RAG LLM:
  ```
  http://<External IP>:<Port>/search_simple?query=<query>
  ```
- For RAG LLM:
  ```
  http://<External IP>:<Port>/search?query=<query>
  ```
- Replace `<External IP>` and `<Port>` with the actual IP address and port number from step 2. Replace `<query>` with your search query.

Ensure you replace placeholders with actual values when executing commands. This guide assumes that your `deployment.yaml` is correctly configured and that you have the necessary permissions to deploy to GKE.



