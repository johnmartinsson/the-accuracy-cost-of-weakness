# Supplementary material
These are the supplementary materials for the paper: "The Accuracy Cost of Weakness: A Theoretical Analysis of Fixed-Segment Weak Labeling for Events in Time".

Under review at TMLR.

## Setup virtual environment
A careful person will execute the commands in setup_venv.sh one by one so that nothing unintended goes wrong along the way. Otherwise, simply run

    ./setup_venv.sh

This will install all dependencies in a virtual environment.

To actiave the environment run

    source ./weak-labeling/bin/activate

To deactivate the environment run

    deactivate

## Verify the theoretical analysis of FIX weak labeling in Section 4.2

**Start Jupyter Notebook**:

```bash
jupyter notebook symbolic_verification_of_analysis.ipynb
```

The notebook should open in your browser. Now, simply follow the instructions in the notebook to verify the Theorems.

If the packages are not found, then jupyter is probably using the wrong environment. Following these steps should fix that.

**Select the Correct Kernel**:
   - Open the notebook in Jupyter.
   - Go to the Kernel menu.
   - Select Change Kernel.
   - Choose "Python (weak-labeling)".

This should ensure that your Jupyter notebook uses the correct virtual environment.

## Reproduce the results

    ./produce_figures.sh 1000

This will reproduce all the results figures in Section 6, by running the simulation described in Section 5 with 1000 simulated audio recording samples. This takes hours when simulating 1000 audio recordings, you can reduce this to verify that everything runs correctly and that the images look reasonable, but the process has quite a lot of variance, so do not be surprised if the results look off. I have tried this with 10 samples, and there is a tendency for the curves to end up above the theory lines. But if you run it with 1000 samples the curves converge nicely.

The simulation scripts are "label_accuracy_simulations.py" and "cost_simulations.py". Read these if you want to fully understand the simulation setup.

## Illustrative figures

    ./illustrative_figures.py

This will reproduce most of the illustrative figures used in the paper.

## Description of other files

- definitions.py : the definitions presented in the paper
- theorems.py : the theorems presented in the paper
- baby_event_lengths.npy : the event lengths for crying baby events from the NIGENS dataset
- dog_event_lengths.npy : the event lengths for barking dog events from the NIGENS dataset

## Additional analysis scripts (Appendix A.4 and A.5)

There are two self-contained Python tools for inspecting audio-event
annotations and comparing them with a simple annotation-quality theory.

| Script | Purpose | Typical output |
|--------|---------|----------------|
| `audioset_label_analysis.py` | Compare *weak* (clip-level) vs *strong* (frame-level) labels in **AudioSet-Eval** for a chosen class and overlay the empirical accuracy on a theoretical γ-accuracy curve. | <br>• `event_length_distribution_<label>.png` – histogram of event durations<br>• `accuracy_curve_<label>.png` – theoretical accuracy vs γ with empirical line<br>• Console summary: measured accuracy and best-matching γ |
| `offset_distribution_analysis.py` | Download & extract the **NIGENS** corpus (on first run) and visualise the distribution of **offsets**<br>`offset = segment_end − event_start`<br> between events and overlapping query segments. The script automatically analyses the *dog* and *baby* subclasses. | • `dog_offset_distribution.png`<br>• `baby_offset_distribution.png`<br>Each file overlays one histogram per fraction `f` |

---

### 1. `audioset_label_analysis.py` (Appendix A.5)

```bash
# Requirements
pip install pandas numpy matplotlib requests

# Example: analyse the “Animal” super-class (/m/0jbk)
python audioset_label_analysis.py \
       --label /m/0jbk \
       --data-dir ./data    \   # cache AudioSet evaluation files here
       --output-dir ./plots     # write PNGs here
```

### 2. `offset_distribution_analysis.py` (Appendix A.4)

```bash
# Requirements (note: no pandas needed here)
pip install numpy matplotlib requests

# First run: download ~2 GB, extract, then plot offsets
python offset_distribution_analysis.py \
       --root ~/datasets \      # NIGENS.zip + extracted folder will live here
       --fractions 0.1 1 10     # choose any positive fractions f
```
