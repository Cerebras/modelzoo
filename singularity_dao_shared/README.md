# Patterns and Predictions Shared Repository

This repository contains example code shared by Cerebras with Patterns and Predictions.

## Setup

In order to run the shared models, a copy of [modelzoo](https://github.com/Cerebras/modelzoo) should be cloned into this repo's parent directory.

## Usage

Please refer to modelzoo's [README](https://github.com/Cerebras/modelzoo#readme) for information on directory structure and running models on the Cerebras System.

When running with `singularity`, in order for `modelzoo` imports to work correctly, it is necessary to bind the path to this repo's parent directory by adding it to the `-B` flag.
