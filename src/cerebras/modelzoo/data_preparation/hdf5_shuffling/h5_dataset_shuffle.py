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
import glob
import os
import time

import h5py
import numpy as np


def shuffle_dataset(args):
    np.random.seed(seed=115 + args.worker_id)

    # Recursively find all H5 files in inputdir
    h5filenames = glob.glob(
        os.path.join(args.inputdir, '**/*.h5'), recursive=True
    )
    img_flag = args.multi_modal
    inputdir = args.inputdir
    if inputdir[-1] != '/':
        inputdir += '/'
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
    workers_dir = os.path.join(args.outdir, 'workers')
    if not os.path.exists(workers_dir):
        os.makedirs(workers_dir, exist_ok=True)
    worker_file = os.path.join(workers_dir, f'worker-{args.worker_id}.txt')
    if os.path.exists(worker_file):
        os.remove(worker_file)
    shuf_split_dir = os.path.join(args.outdir, 'shuf_split')
    if not os.path.exists(shuf_split_dir):
        os.makedirs(shuf_split_dir, exist_ok=True)

    sorted(h5filenames)
    print(h5filenames)

    chunk_outdirs = []
    for i in range(args.num_output_chunks):
        chunk_outdir = os.path.join(args.outdir, f'{str(i)}')
        chunk_outdirs.append(chunk_outdir)
        if not os.path.exists(chunk_outdir):
            os.makedirs(chunk_outdir, exist_ok=True)

    spread_start_time = time.time()
    for idx, h5filename in enumerate(h5filenames):
        if idx % args.num_parallel != args.worker_id:
            continue
        done_filename = h5filename.replace(inputdir, "").replace('.h5', '.shuf')
        done_filename = os.path.join(shuf_split_dir, done_filename)
        done_filedir = os.path.dirname(done_filename)
        if not os.path.exists(done_filedir):
            os.makedirs(done_filedir, exist_ok=True)
        if os.path.exists(done_filename):
            print(f'H5 file is done: {h5filename}')
            continue
        chunk_samples = []
        if img_flag:
            img_chunk_samples = []
        for i in range(args.num_output_chunks):
            chunk_samples.append([])
            if img_flag:
                img_chunk_samples.append([])
        h5file = h5py.File(h5filename, 'r')
        num_samples = h5file['data'].shape[0]
        out_chunk_ids = np.random.choice(
            np.arange(args.num_output_chunks), size=num_samples
        )
        print(out_chunk_ids)
        for seq_id, out_id in enumerate(out_chunk_ids):
            chunk_samples[out_id].append(h5file['data'][seq_id : seq_id + 1])
            if img_flag:
                img_chunk_samples[out_id].append(
                    h5file["img_path"][seq_id : seq_id + 1]
                )
        for chunk_id, chunk in enumerate(chunk_samples):
            if len(chunk) == 0:
                continue
            chunk = np.concatenate(chunk, axis=0)
            if img_flag:
                img_chunk = np.concatenate(img_chunk_samples[chunk_id], axis=0)
            out_hash = f'{np.random.randint(2**48):012x}'
            out_filename = os.path.join(
                args.outdir, f'{chunk_id}', f'sdata-{out_hash}.h5'
            )
            # TODO: Make this a while loop
            assert not os.path.exists(out_filename)
            outfile = h5py.File(out_filename, 'w')

            outfile.create_dataset(
                'data',
                data=chunk,
                shape=chunk.shape,
                chunks=(1, chunk.shape[1], chunk.shape[2]),  # hdf5 chunk size
                compression='gzip',
            )
            if img_flag:
                # shape/chunk arguments not needed for strings
                outfile.create_dataset(
                    "img_path",
                    data=img_chunk,
                    compression='gzip',
                )
            outfile.close()
        h5file.close()
        with open(done_filename, 'w') as done_file:
            done_file.write('Done :D\n')

    with open(worker_file, 'w') as worker_file:
        worker_file.write('Done :D\n')

    spread_end_time = time.time()
    print(
        f'Spreading samples took {spread_end_time-spread_start_time} seconds',
        flush=True,
    )

    done = False
    while not done:
        done_workers = 0
        for i in range(args.num_parallel):
            if os.path.exists(
                os.path.join(args.outdir, 'workers', f'worker-{i}.txt')
            ):
                done_workers += 1
        if done_workers == args.num_parallel:
            done = True
        time.sleep(2)

    sync_end_time = time.time()
    print(
        f'Syncing with other workers took {sync_end_time-spread_end_time} seconds',
        flush=True,
    )

    for chunk_id, chunk_outdir in enumerate(chunk_outdirs):
        if chunk_id % args.num_parallel != args.worker_id:
            continue
        subchunk_filenames = glob.glob(os.path.join(chunk_outdir, 'sdata-*.h5'))
        sorted(subchunk_filenames)  # For determinism
        chunk_data = []
        if img_flag:
            img_chunk_data = []
        for subchunk_filename in subchunk_filenames:
            h5file = h5py.File(subchunk_filename, 'r')
            chunk_data.append(np.array(h5file['data']))
            if img_flag:
                img_chunk_data.append(np.array(h5file["img_path"]))
            h5file.close()
        chunk = np.concatenate(chunk_data, axis=0)
        if img_flag:
            img_chunk = np.concatenate(img_chunk_data, axis=0)
        indices = np.arange(chunk.shape[0])
        np.random.shuffle(indices)
        chunk = chunk[indices]
        if img_flag:
            img_chunk = img_chunk[indices]
        chunk_filename = os.path.join(args.outdir, f'data-{chunk_id:05d}.h5')
        chunk_file = h5py.File(chunk_filename, 'w')
        chunk_file.attrs["n_examples"] = chunk.shape[0]
        chunk_file.create_dataset(
            'data',
            data=chunk,
            shape=chunk.shape,
            chunks=(1, chunk.shape[1], chunk.shape[2]),  # hdf5 chunk size
            compression='gzip',
        )
        if img_flag:
            chunk_file.create_dataset(
                "img_path",
                data=img_chunk,
                compression='gzip',
            )
        chunk_file.close()

        # Remove all subchunk files, since they have been consolidated
        for subchunk_filename in subchunk_filenames:
            os.remove(subchunk_filename)

    consolidate_end_time = time.time()
    print(
        f'Consolidating shuffled samples took {consolidate_end_time-sync_end_time} seconds',
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'inputdir', help='Path to the H5 files directory to shuffle.'
    )
    parser.add_argument(
        'outdir', help='Directory path in which to place shuffled data files'
    )
    parser.add_argument(
        'num_output_chunks',
        help='Number of output data chunks to create',
        type=int,
    )
    parser.add_argument('--num_parallel', default=1, type=int)
    parser.add_argument('--worker_id', default=0, type=int)
    parser.add_argument(
        '--multi_modal',
        action="store_true",
        help="Flag to specify if we our data has an image-path in addition to the tokens. Image-paths are strings and thus have to be stored as separate datasets within an h5 file.",
    )
    args = parser.parse_args()
    shuffle_dataset(args)
