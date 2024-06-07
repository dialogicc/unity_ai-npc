# Project Setup Guide

This guide will walk you through the steps needed to set up the project on a Windows machine.

## Prerequisites

- Anaconda
- Visual Studio Code (VS Code)
- Python

## Steps

1. **Install Anaconda:**
   Download and install Anaconda from [here]([https://www.anaconda.com/products/individual](https://docs.anaconda.com/free/miniconda/)), search and run the "Anaconda Prompt (miniconda3)" console.

2. **Configure Conda to use conda-forge:**
   Run the following command in the Anaconda Prompt (miniconda3) console:
   ```sh
   conda config --add channels conda-forge
3. **Create Conda Environment:**
   
   Navigate to the right location
   Run the following command:

   *Windows:*
   ```
   cd %USERPROFILE%\Downloads
   ```
   *Unix:*
   ```
   cd ~/Downloads
   ```
   
   Ensure the .yaml file is located at "Downloads":
   ```sh
   conda env create -f conda_environment_windows.yaml
   ```
   If you encounter issues with Parler TTS due to the Git path:
      - Install Git for Windows from [here](https://gitforwindows.org/).
      - After installation, retry installing Parler TTS.  
      
5. **Download and Install Python:**
   Download the latest version of Python from [here](https://www.python.org/downloads/) and install it.

6. **Install Ollama and Gemma:2b:**
   Follow the instructions provided by Ollama to install it. Then, install the Gemma:2b package.

7. **Adjust Python Path in Script:**
   Open the script "Interaction" and modify the FileName to match the path to your Conda environment's Python executable. For example:
   ```csharp
   FileName = @"C:\Users\INF3_1\miniconda3\envs\master2\python.exe", // Path to the Conda environment's Python executable
8. **Run Unity-Project:**
   Open the Interactive_AI-folder in Unity and talk to the NPC by pressing 'e' to start and stop the audio-recording.
