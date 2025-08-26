# isort: off
# MZ: import sys
# MZ: import os
# MZ: sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
# isort: on

if __name__ == '__main__':
    import warnings
    from cerebras.modelzoo.common.run_utils import run

    warnings.warn(
        "Running models using run.py is deprecated. Please switch to using the ModelZoo CLI. "
        "See https://training-docs.cerebras.ai/model-zoo/cli-overview for more details."
    )

    run()
