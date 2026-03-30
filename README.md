## Installation

### 1. Clone the Repository
First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/lmanan/tracking_library.git
cd tracking_library
```

### 2. Set Up the Conda Environment
Create and activate a dedicated Conda environment named `motile`:

```bash
conda create -n motile python==3.10
conda activate motile
```

### 3. Install Dependencies
Install the required packages, including `ilpy`, `scip`, and all dependencies:

```bash
conda install -c conda-forge -c funkelab -c gurobi ilpy
conda install -c conda-forge scip
```

### 4. Install tracking_library in Editable Mode
Finally, install the tracking_library repository in editable mode (useful for development):

```bash
pip install -e .
```
Full Setup Summary:

```bash
git clone https://github.com/lmanan/tracking_library.git
cd tracking_library
conda create -n motile python=3.10
conda activate motile
conda install -c conda-forge -c funkelab -c gurobi ilpy
conda install -c conda-forge scip
pip install -e .
```


## Getting Started

```bash
cd tracking_library/tracklet_stitching/manual
python infer_node_edge_selection.py
```


