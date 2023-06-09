import argparse
import os
import sys

sys.path.append("./preprocessing")
from preprocessing import normalize_text, filter, shuffle_holdout, datasets
from dedup import to_hash, dedup_train, generate_duplicate_pairs, generate_connected_components, generate_duplicates_dict

ds_names = ["arxiv", "stackexchange", "book", "wikipedia", "github", "c4", "common_crawl"]
cc_years = ["2019-30", "2020-05", "2021-04", "2022-05" "2023-06"]

def main(input_dir):

    # norm text
    ds_dirs = ds_names.copy()
    ds_dirs.remove("common_crawl")
    for cc in cc_years:
        ds_dirs.append("common_crawl/"+cc)
    
    red_pj_norm = os.path.join(input_dir, "RedPajama_norm")
    for dataset in ds_dirs:
        norm_args = argparse.Namespace()
        norm_args.data_dir = os.path.join(input_dir, dataset)
        norm_args.target_dir = os.path.join(red_pj_norm, dataset)
        norm_args.zst = "common_crawl" in dataset
        norm_args.idx = -1
        normalize_text.normalize_text(norm_args)

    # filter docs
    short_docs = os.path.join(red_pj_norm, "red_pj_filter.pickle")
    filter_args = argparse.Namespace()
    filter_args.input_dir = red_pj_norm
    filter_args.output_file = short_docs
    filter_args.n_docs = 100 #TODO: update
    filter_args.dataset_name = "all"
    filter_args.threshold = 200
    filter.filter_dataset(filter_args)

    # generate minhash
    for dataset in ds_dirs:
        hash_args = argparse.Namespace()
        hash_args.dataset_name = "common_crawl" if "common_crawl" in dataset else dataset
        hash_args.input_dir = os.path.join(red_pj_norm, dataset)
        hash_args.output_dir = os.path.join(red_pj_norm, dataset)
        hash_args.n_docs = 100 #TODO: update
        hash_args.iter = 0
        hash_args.index_start = 0
        hash_args.index_end = None
        hash_args.w = 13
        hash_args.k = 10000
        to_hash.generate_hashes(hash_args)

    # generate duplicates
    dup_dir = os.path.join(red_pj_norm, "dup")
    os.makedirs(dup_dir, exist_ok=True)
    dup_pairs_args = argparse.Namespace()
    dup_pairs_args.input_dir = red_pj_norm
    dup_pairs_args.out_file = os.path.join(dup_dir, "duplicate_pairs.txt")
    dup_pairs_args.range = 13
    dup_pairs_args.bands = 9
    dup_pairs_args.processes = 45
    generate_duplicate_pairs.generate_pairs(dup_pairs_args)

    dup_connected_args = argparse.Namespace()
    dup_connected_args.input_dir = dup_dir
    dup_connected_args.out_file = os.path.join(dup_dir, "connected_components.pickle")
    generate_connected_components.generate_connected_components_mp(dup_connected_args)

    dup_docs = os.path.join(dup_dir, "duplicates.pickle")
    dup_dict_args = argparse.Namespace()
    dup_dict_args.input_file = os.path.join(dup_dir, "connected_components.pickle")
    dup_dict_args.out_file = dup_docs
    generate_duplicates_dict.generate_duplicates(dup_dict_args)

    # interleave & shuffle
    shuffle_holdout.pass_1_shuffle(
        datasets.RedPajamaReplication(
            datasets.redpj_datasets(red_pj_norm+"/"), dup_docs, short_docs
        ),
        output_dir_path=os.path.join(red_pj_norm, "pass1"),
    )

    # split train & holdout
    for j in range(1, 21):
        shuffle_holdout.pass_2_shuffle_holdout(
            input_dir_path=os.path.join(red_pj_norm, "pass1"),
            output_dir_path=os.path.join(red_pj_norm, "train"),
            output_holdout_dir_path=os.path.join(red_pj_norm, "holdout"),
            start_index=j-1,
            end_index=j,
            chunk_id=j,
        )

    # Deduplicate Train against Holdout
    for j in range(1, 21):
        dedup_train.deduplicate_train_holdout_sets(
            os.path.join(red_pj_norm, "train"), os.path.join(red_pj_norm, "holdout"), os.path.join(red_pj_norm, "train_deduped"), j,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Dataset input directory.")
    args = parser.parse_args()
    main(args.input_dir)
