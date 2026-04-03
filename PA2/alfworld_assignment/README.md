# CS588 Programming Assignment #2

**Due:** TBD
**Objective:** Understanding agentic AI through observing an LLM-based agent in ALFWorld

## Related Materials

- `PA2.zip`

## Developing Environment

- Python 3.10
- PyTorch (optional, not required for this assignment)
- ALFWorld
- Google Generative AI SDK (Gemini API, free tier)

## Setup

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate alfworld_agent
```

Or with pip only:

```bash
conda create -n alfworld_agent python=3.10 -y
conda activate alfworld_agent
pip install -r requirements.txt
```

### 2. Download ALFWorld data

```bash
alfworld-download
```

This downloads ~5 GB of game files to `~/.alfworld/`. Only needs to run once.

### 3. Get a Gemini API key (free)

1. Go to https://aistudio.google.com/apikey
2. Create a free API key.
3. Set it as an environment variable:

```bash
export GEMINI_API_KEY="your-key-here"
```

> **Free tier limits:** 15 requests/min, 1,500 requests/day.
> The script includes automatic rate-limit handling.

### 4. Verify setup

```bash
cd alfworld_assignment
python -c "import alfworld; import google.generativeai; print('OK')"
```

## Requirements

### 1. Background

ALFWorld is a text-based environment where an agent must complete household tasks such as "put a clean mug on the shelf." The agent receives:

- **Text observation:** description of what it sees (e.g., "On the desk 1, you see a mug 1, a pen 2.")
- **Admissible commands:** list of valid actions (e.g., "go to shelf 1", "take mug 1 from desk 1")

There are 6 task types: pick-and-place, heat-then-place, cool-then-place, clean-then-place, examine-under-lamp, and pick-two-objects. The agent must complete each task within 30 steps.

The provided code implements a **ReAct** agent (Yao et al., ICLR 2023) that uses a Thought → Action → Observation loop with the **Gemma 3 27B** model via the Gemini API.

### 2. Run the Agent

```bash
python src/main.py
```

**Before running:** change `STUDENT_ID` at the top of `src/main.py`.

This runs 15 episodes and saves all trajectories to `eval_results/`. The full run takes approximately 60–90 minutes due to API rate limits.

### 3. Trajectory Analysis (10 points)

Open the result JSON file in `eval_results/`. Each episode contains a `trajectory` array where each entry has `thought`, `action`, and `observation`.

Answer the following:

**Q1.** Report the overall success rate (how many of the 15 episodes succeeded).

**Q2.** Pick **two failed episodes**. For each, describe:
- What was the task?
- What did the agent do wrong? (e.g., stuck in a loop, picked the wrong object, skipped a step, repeated invalid actions)
- At which step did the agent first go off track?

**Q3.** Pick **one successful episode**. Describe how the agent's reasoning (in the "thought" field) helped it follow the plan correctly.

### 4. Analysis (5 points)

Discuss the limitations of this simple ReAct approach. Think about:

- What failure patterns do you observe across episodes?
- How does the 30-step limit affect the agent?
- What happens when the agent encounters "Nothing happens" (invalid actions)?
- How could these limitations be addressed? Propose at least one concrete improvement.

## Submission

Submit the following to KLMS:

1. A **report in PDF format** containing:
   - Your student ID
   - The overall success rate
   - Your answers to Q1–Q3 (with specific episode numbers and step references)
   - Your analysis for Requirement 4
   - A few example trajectories (copy relevant portions)
2. The `eval_results/` folder with your result JSON file

**Do not** include your Gemini API key anywhere in the submission.

## Policies

- Everyone must submit their own work.
- You may collaborate with others, but what you submit (results and analysis) must be your own.
- Since results depend on the LLM's behavior, your trajectories will differ from others — this is expected.
