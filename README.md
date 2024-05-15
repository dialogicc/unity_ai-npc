# Project Setup Guide

This guide will walk you through the steps needed to set up the project on a Windows machine.

## Prerequisites

- Anaconda
- Visual Studio Code (VS Code)
- Git
- Python

## Steps

1. **Install Anaconda:**
   Download and install Anaconda from [here]([https://www.anaconda.com/products/individual](https://docs.anaconda.com/free/miniconda/)).

2. **Configure Conda to use conda-forge:**
   Open Anaconda Prompt and run:
   ```sh
   conda config --add channels conda-forge
3. **Create Conda Environment:**
   Ensure you are in the project directory and run:
   ```sh
   conda env create -f environment.yaml
4. **Download and Install Python:**
   Download the latest version of Python from [here](https://www.python.org/downloads/) and install it.
5. **Select Conda Environment in VS Code:**
   - Open VS Code.
   - Click on the interpreter in the bottom left corner.
   - Select "Python: Select Interpreter".
   - Choose the Conda environment you created earlier.
   - If the environment is not listed, click "Enter interpreter path..." and navigate to the environment's folder.

6. **Install PyTorch:**
   Follow the instructions on [pytorch.org](https://pytorch.org/get-started/locally/) to install PyTorch in your environment. After installation, restart the kernel in VS Code.

7. **Install Ollama and Gemma:2b:**
   Follow the instructions provided by Ollama to install it. Then, install the Gemma:2b package.

8. **Fix TTS Installation Issues:**
   If you encounter issues with Parler TTS due to the Git path:
   - Install Git for Windows from [here](https://gitforwindows.org/).
   - After installation, retry installing Parler TTS.

9. **Adjust Python Path in Script:**
   Open the script and modify the FileName to match the path to your Conda environment's Python executable:
   ```csharp
   FileName = @"C:\Users\INF3_1\miniconda3\envs\master2\python.exe", // Path to the Conda environment's Python executable
