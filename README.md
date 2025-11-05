# Introduction and overview

This repository originally contained the [Contextual Dynamics Lab](http://www.context-lab.com)'s code related to the [CCN Algonauts 2021 challenge](http://algonauts.csail.mit.edu/challenge.html). However, since that challenge has *long* passed (without us entering it!) our focus has shifted to a new question:

> As you go through life, your subjective experience reflects ongoing processing and coordination among and between the many (many!) components of your brain. Conceptually, you can think of your complete subjective experience as a sort of “baseline” that we can use for (theoretical) comparrison.
>
> But how does an individual piece of your brain “experience” those same things? For example, presumably your retina and primary visual cortex mostly process visual aspects of your experiences, but perhaps they don’t “care” as much about other stuff (information from other senses, emotion, motor processing, etc.). Could we come up with a “transformation function” that takes in a stimulus (say, a movie) and outputs a modified version of the stimulus containing only the parts that one specific brain region “cares about”? For example, maybe this function would remove the audio track from the movie if we were to ask it how the movie was “seen” by primary visual cortex. Or maybe we’d get only auditory information if we processed it like primary auditory cortex. Or maybe we’d get a version that focused on (or highlighted in some way) faces, if we passed in face processing areas. And so on.
>
>One could also imagine trying to remove individual brain areas. E.g., suppose we asked how everywhere in the brain except face processing areas “saw” that movie. Would faces be blurred out or unintelligible?
>
> This project is in the very early (brainstorming and prototyping) phases: we’re trying to figure out how to solve a very difficult conceptual and theoretical problem! If we can figure out some approaches, there are myriad deep, interesting, and enormously useful applications. A few examples:
>
>   - We could understand, intuitively, the “role” of any brain area or set of brain areas
>   - We could simulate changes in function following brain lesions, damage, and/or resective surgery
>   - We could construct “optimized” stimuli that maximally target specific regions or sets of regions. This could be useful for other neuroscientific experiments, building/testing models, therapies, etc.

### Where did the name "giblet responses" come from?

The brain's "giblets" are its myriad constituent pieces (apologies for the somewhat morbid/gross metaphor). We want to understand how each giblet [responds](https://en.wikipedia.org/wiki/Region_of_interest) to a stimulus-- in this case, a movie.

## Setup instructions

### Quick Setup (Recommended)

Run the automated setup script:

```bash
./setup_environment.sh
```

This will:
- Install miniconda (if needed)
- Create the `giblet-py311` conda environment
- Install all dependencies
- Download the Sherlock dataset
- Verify the installation

For manual setup or troubleshooting, see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md).

### Cluster Training

For distributed training on a GPU cluster:

```bash
# Launch 8-GPU training on tensor01
./remote_train.sh --cluster cluster_name --config cluster_train_config.yaml --gpus 8 --name production_run

# Monitor training
./check_remote_status.sh --cluster cluster_name

# Attach to running session
ssh yourNetID@tensor01.dartmouth.edu
screen -r production_run
```

See [SETUP.md](SETUP.md) for complete cluster training guide including:
- Remote training automation
- Monitoring and debugging
- Resuming from checkpoints
- Retrieving results

### Joining the Team

1. Fork this repository and clone the fork to your computer
2. Add yourself to the "Team CDL contributors" list below (you'll need to add yourself to your fork and then submit a pull request).  By adding yourself to the team, you agree to:
    - Maintain clear, open, and regular communications with your fellow team members related to work on this challenge project.  This includes joining the associated [slack channel](https://context-lab.slack.com/archives/C020V4HJFT4) and checking/contributing to it regularly while the project is active.
    - Participate in the hackathons related to the project
    - Take responsibility for the methods, code, and results
    - Assist in writing up and submitting our results

## Team CDL contributors

Note: the tabulated list of team members below is ordered by join date; author order on any publications will be determined based on contributions of:
- Code and writing (using GitHub to track commits)
- Ideas (documented on Slack and/or GitHub)

### Team members:

| Name                   | Join date   | GitHub username | Email address        |
-------------------------|-------------|-----------------|----------------------|
| Jeremy R. Manning      | May 4, 2021 | [jeremymanning](https://github.com/jeremymanning) | jeremy@dartmouth.edu |

# Approach

Some scattered thoughts (expand and refine later...)
- hyperalignment-based decoding, similar to what we used [here](https://arxiv.org/abs/1701.08290)
- some sort of deep learning-based thing-- GAN?  Autoencoder + hyperalignment?  Custom model?
- use TFA and/or timecorr features?
- connect with other datasets (e.g. Huth et al. semantic decoding, neurosynth, etc.)?

Another half-baked idea:
 - Build an autoencoder with the following design:
     - Input and output layers: vectorized video frames + audio
     - Several (not sure how many) intermediate layers (leading to/from the "middle")
     - Middle layer: "compressed" representation-- should have the same number of features as there are fMRI voxels in the dataset
         - Potential future tweak: add in some convolutional layers to pool information across nearby voxels
 - Now train the autoencoder to optimize not just for matching the input/output, but also matching the middle layer to match the fMRI responses to each frame
     - The frame rate will be much higher than the fMRI samplerate, so we might want to use linear interpolation to generate intermediate images between the actual fMRI responses
     - BOLD responses are slow; perhaps there's a way to detangle the temporal blurring?
 - Now, for any (new) input image, we can predict brain responses using the encoder part of the autoencoder
 - We can also use the decoder part to predict video/audio from brain responses
 - We can also fix some of the responses (of some "voxels" in the middle layer) to 0 to "lesion" the responses, and then see how the decoder's outputs change

# Results

(insert description here, including links to Jupyter notebooks for reproducing any figures)

# Bibliography

1. Cichy RM, Dwivedi K, Lahner B, Lascelles A, Iamshchinina P, Graumann M, Andonian A, Murty NAR, Kay K, Roig G, Oliva A (2021) The Algonauts Project 2021 Challenge: How the Human Brain Makes Sense of a World in Motion. *arXiv*: 2104.13714. [[link](https://arxiv.org/abs/2104.13714)]
