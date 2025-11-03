# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Setuptools installation script for cerebras modelzoo.
"""

from pathlib import Path

from setuptools import find_namespace_packages, setup
from setuptools.command.install import install


class ErrorOnInstall(install):
    """Check that cerebras_modelzoo was installed as editable."""

    def run(self):
        raise RuntimeError(
            "The package `cerebras_modelzoo` is not a mountable package on the "
            "appliance since it is not available in PyPI. It must be installed as an "
            "editable package using:\n"
            "  $ pip install -e /path/to/cerebras/modelzoo\n"
        )


def main():
    """
    Python wheel/setuptools installation script for cerebras modelzoo.
    """
    root_directory = Path(__file__).parent
    long_description = (root_directory / "PYPI-README.md").read_text()

    __version__ = "2.7.0"

    entry_points = []

    # Add modelzoo cli entry point
    mz_cli_entry = 'cszoo = \
        cerebras.modelzoo.cli.main:main'
    entry_points.append(mz_cli_entry)

    setup(
        name=f"cerebras_modelzoo",
        version=__version__,
        description='Cerebras Modelzoo',
        url='https://github.com/Cerebras/modelzoo',
        author='Cerebras Systems',
        author_email='support@cerebras.net',
        long_description=long_description,
        long_description_content_type='text/markdown',
        packages=find_namespace_packages(where='src/'),
        package_dir={'': 'src'},
        package_data={
            "cerebras.modelzoo": ['**/*.yaml'],
        },
        entry_points={
            'console_scripts': entry_points,
        },
        install_requires=[
            f"cerebras_pytorch=={__version__}",
            "argcomplete==3.5.0",
            "tabulate==0.9.0",
            "transformers==4.45.2",
            "tokenizers==0.20.1",
            "datasets==2.19.1",
            "filelock==3.14.0",
            "more-itertools==10.5.0",
            "datasketch==1.6.5",
            "ftfy==6.2.3",
            "networkit==10.1",
            "Keras-Preprocessing==1.1.2",
            "scipy==1.10.1",
            "regex>=2021.8.3",
            "nltk==3.9.1",
            "spacy==3.7.6",
            "matplotlib==3.9.2",
            "pyarrow==17.0.0",
            "pydantic==2.8.2",
            "pyYAML",
            "pandas==2.2.3",
            "jsonschema==4.23.0",
            "torch==2.4.0",
            "torchvision==0.19.0",
            "safetensors==0.4.5",
            "sentencepiece==0.2.0",
            # pylint: disable=line-too-long
            "lm-dataformat @ https://github.com/leogao2/lm_dataformat/archive/ac85cb7dae49ce25e9973a128ebd9167deaf64dd.zip",
            "lm-eval @ https://github.com/EleutherAI/lm-evaluation-harness/archive/refs/tags/v0.4.7.zip",
            "bigcode_eval @ https://github.com/bigcode-project/bigcode-evaluation-harness/archive/f0b81a9d079289881bd42f509811d42fe73e58cf.zip",
            "h5py==3.13.0",
            "tqdm==4.66.5",
            "Flask==3.0.3",
            "Pillow==9.4.0",
            "faiss-cpu==1.8.0.post1",
            "deepdiff>=8,<9",
            "cerebras_cloud_sdk>=1.0.0",
            "click==8.1.7",
            "termcolor==2.4.0",
            "slack-sdk==3.11.2",
        ],
        # PyPI package information.
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development",
            "Topic :: Software Development :: Libraries",
            "License :: OSI Approved :: Apache License",
            "Programming Language :: Python :: 3",
        ],
        license_files=('LICENSE',),
        cmdclass={
            'install': ErrorOnInstall,
        },
    )


if __name__ == "__main__":
    main()
