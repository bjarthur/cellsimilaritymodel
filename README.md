# Heritable short-term cellular memory emerges from stochastic biochemical reaction networks

## Project Abstract
Cells exhibit a mysterious form of selective heritable short-term memory, influencing outcomes as diverse as cell fate decisions in embryos and environmental responses in cancer cells and bacteria. Here, we present a simple theoretical framework explaining how this selective memory can arise from the reactions by which molecules are regulated in cells. Our key insight is that related cells retain more similar molecular concentrations relative to random cells when a greater variance of possible concentration states is created during a single cell generation than is created by cell division across a population. This persistence of molecular similarity down a lineage constitutes a form of heritable short-term memory. We identify the biochemical networks that produce, modify, and degrade molecules as an underexplored source of these additional molecular concentration states. Using experimentally informed simulations, we find that the strength and duration of molecular similarity down a lineage depend on tunable network properties, explaining why some cellular traits persist only briefly while others last generations. These contributions to molecular concentration variance from biochemical reaction networks act in concert with gene expression and other regulatory processes to shape the protein composition of cells. Our framework yields clear, testable predictions for determining how biochemical network architectures drive non-genetic cellular inheritance.

## Repository Structure
Description of how the code is organized
- `/docs` - Further documentation on setup
- `/figures` - Contains script for figure plotting
- `/simulations` - Code to generate data for each figure
- `/utils` - Utility files for running code

## Requirements
### Software Dependencies
List of required software and versions:
- python-version=3.13
- cmapy>=0.6.6
- matplotlib>=3.10.7
- numpy>=2.3.4
- opencv-python>=4.11.0.86
- scipy>=1.16.2
- Roboto font (optional, used for the figure panel labels; see `docs/setup.md`)

## Installation
See setup.md file in `/docs` 

## Contact
Contact Allyson Sgro at sgroa@janelia.hhmi.org. 