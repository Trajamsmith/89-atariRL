# Atari RL
Exploring a now-classic problem in reinforcement learning — training an agent to play Atari games.

Initial implementations of these models were graciously borrowed from the following sources:
* Lunar Lander Deep Q -- https://www.youtube.com/watch?v=5fHngyN8Qhw
* Atari Dueling Q -- https://medium.com/analytics-vidhya/building-a-powerful-dqn-in-tensorflow-2-0-explanation-tutorial-d48ea8f3177a

## Installation
Create an environment using either `conda` or `pipenv`. Steps below assume `conda`. From the root directory:
1. `conda create -n atari-rl && conda activate atari-rl`
2. `pip install -r requirements.txt` (installs into the conda environment)


### Using Windows
OpenAI Gym is not officially supported on Windows, so exta installation steps are required. This process can be somewhat finicky, so if you have the option, prefer running these scripts on macOS or Linux!

The steps found [here](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) are what worked for me:
1. Install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. `pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py`
3. Install [Xming](https://sourceforge.net/projects/xming/)

