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

import argparse
import logging
import os
import pickle
from glob import glob

import networkit as nk
import tqdm


def construct_graph(set_of_duplicate_pairs):
    G = nk.Graph()
    mapper = {}
    for pair in tqdm.tqdm(set_of_duplicate_pairs):
        node1_name, node2_name = pair
        if node1_name not in mapper:
            mapper[node1_name] = G.addNode()
        if node2_name not in mapper:
            mapper[node2_name] = G.addNode()
        G.addEdge(mapper[node1_name], mapper[node2_name])
    return G, mapper


def find_connected_components(G):
    cc = nk.components.ConnectedComponents(G)
    cc.run()
    return cc.getComponents(), cc.numberOfComponents()


def generate_connected_components_mp(args):
    files = glob(f"{args.input_dir}/*")

    logging.info("Started graph building...")
    # load pickled duplicate pairs
    set_of_duplicate_pairs = set()
    for fp in files:
        with open(fp, "r") as f:
            for line in tqdm.tqdm(f):
                pair = tuple(line.strip().split(" :: "))
                if pair[0] != pair[1]:
                    set_of_duplicate_pairs.add(pair)

    length = len(set_of_duplicate_pairs)

    logging.info(f"Length of the set of duplicates: {length}")

    output_directory = os.path.dirname(args.input_dir)
    pickle_file = os.path.join(output_directory, 'duplicate_pairs.pickle')

    with open(pickle_file, 'wb') as file:
        pickle.dump(set_of_duplicate_pairs, file)

    # Generate a graph using IDs as nodes and a pair of IDs as an edge
    nk.setNumberOfThreads(60)
    G, mapper = construct_graph(set_of_duplicate_pairs)
    components, n_components = find_connected_components(G)
    logging.info(f"Number of connected components: {n_components}")

    reversed_mapper = {value: key for key, value in mapper.items()}

    # Dump pickled connected components on disk and load if needed
    with open(args.out_file, "wb") as fout:
        pickle.dump((components, n_components, reversed_mapper), fout)
    logging.info("Finished generating duplicates list.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory, where duplicates generated in previous step are present.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        help="Name of the output .pickle file, which contains the graph of duplicates found, in the form of connected components.",
    )
    args = parser.parse_args()
    generate_connected_components_mp(args)
