
import argparse
import glob
import h5py
import numpy as np
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('inputdir', help='Path to the H5 files directory to shuffle.')
parser.add_argument('outdir', help='Directory path in which to place shuffled data files')
parser.add_argument('num_output_chunks', help='Number of output data chunks to create', type=int)
parser.add_argument('--num_parallel', default=1, type=int)
parser.add_argument('--worker_id', default=0, type=int)
args = parser.parse_args()

np.random.seed(seed=115 + args.worker_id)

# Recursively find all H5 files in inputdir
h5filenames = glob.glob(os.path.join(args.inputdir, '**/*.h5'), recursive=True)

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
    for i in range(args.num_output_chunks):
        chunk_samples.append([])
    h5file = h5py.File(h5filename, 'r')
    num_samples = h5file['data'].shape[0]
    out_chunk_ids = np.random.choice(np.arange(args.num_output_chunks), size=num_samples)
    print(out_chunk_ids)
    for seq_id, out_id in enumerate(out_chunk_ids):
        chunk_samples[out_id].append(h5file['data'][seq_id:seq_id+1])
    for chunk_id, chunk in enumerate(chunk_samples):
        if len(chunk) == 0:
            continue
        chunk = np.concatenate(chunk, axis=0)
        out_hash = f'{np.random.randint(2**48):012x}'
        out_filename = os.path.join(args.outdir, f'{chunk_id}', f'sdata-{out_hash}.h5')
        # TODO: Make this a while loop
        assert not os.path.exists(out_filename)
        outfile = h5py.File(out_filename, 'w')
        outfile.create_dataset(
            'data',
            data=chunk,
            shape=chunk.shape,
            chunks=(1, chunk.shape[1], chunk.shape[2]), # hdf5 chunk size
            compression='gzip',
        )
        outfile.close()
    h5file.close()
    with open(done_filename, 'w') as done_file:
        done_file.write('Done :D\n')

with open(worker_file, 'w') as worker_file:
    worker_file.write('Done :D\n')

spread_end_time = time.time()
print(f'Spreading samples took {spread_end_time-spread_start_time} seconds', flush=True)

done = False
while not done:
    done_workers = 0
    for i in range(args.num_parallel):
        if os.path.exists(os.path.join(args.outdir, 'workers', f'worker-{i}.txt')):
            done_workers += 1
    if done_workers == args.num_parallel:
        done = True
    time.sleep(2)

sync_end_time = time.time()
print(f'Syncing with other workers took {sync_end_time-spread_end_time} seconds', flush=True)

for chunk_id, chunk_outdir in enumerate(chunk_outdirs):
    if chunk_id % args.num_parallel != args.worker_id:
        continue
    subchunk_filenames = glob.glob(os.path.join(chunk_outdir, 'sdata-*.h5'))
    sorted(subchunk_filenames) # For determinism
    chunk_data = []
    for subchunk_filename in subchunk_filenames:
        h5file = h5py.File(subchunk_filename, 'r')
        chunk_data.append(np.array(h5file['data']))
        h5file.close()
    chunk = np.concatenate(chunk_data, axis=0)
    np.random.shuffle(chunk)
    chunk_filename = os.path.join(args.outdir, f'data-{chunk_id:05d}.h5')
    chunk_file = h5py.File(chunk_filename, 'w')
    chunk_file.attrs["n_examples"] = chunk.shape[0]
    chunk_file.create_dataset(
        'data',
        data=chunk,
        shape=chunk.shape,
        chunks=(1, chunk.shape[1], chunk.shape[2]), # hdf5 chunk size
        compression='gzip',
    )
    chunk_file.close()

    # Remove all subchunk files, since they have been consolidated
    for subchunk_filename in subchunk_filenames:
        os.remove(subchunk_filename)

consolidate_end_time = time.time()
print(f'Consolidating shuffled samples took {consolidate_end_time-sync_end_time} seconds', flush=True)
