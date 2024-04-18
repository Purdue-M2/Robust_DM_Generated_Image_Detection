import utils
import h5py
import numpy as np
from multiprocessing import Pool, cpu_count
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)

import gc
gc.enable()


def func(doc):
    # Print the current document ID
    print(f"Processing document {doc}", end=' ')

    # Retrieve data from the 'train' directory
    (img_id, reaL_prompt, fake_prompt, img_url,
     image_gen0_meta, image_gen1_meta, image_gen2_meta, image_gen3_meta) = \
        utils.get_data(f'train/{doc:d}.parquet') #Use test/{doc:d} for loading test dataset

    print(f"Data retrieved for document {doc}")

    # Create a list of tuples with the required data
    pack = [(img_id[i],
             reaL_prompt[i], fake_prompt[i], img_url[i],
             image_gen0_meta[i], image_gen1_meta[i], image_gen2_meta[i], image_gen3_meta[i])  # Use the same fake_prompt for all four fake images
            for i in range(len(img_id))]

    results = []

    # Process each tuple in the pack using utils.get_image_and_text
    for i in pack:
        result = utils.get_data(i)
        if result: results.append(result)

    print(f"Processed data for document {doc}")

    # Unpack the results into separate lists
    img_id, original_prompt, positive_prompt, real, image_gen0, image_gen1, image_gen2, image_gen3 = zip(*results)

    # Convert lists to numpy arrays
    real = np.array(real)
    image_gen0 = np.array(image_gen0)
    image_gen1 = np.array(image_gen1)
    image_gen2 = np.array(image_gen2)
    image_gen3 = np.array(image_gen3)

    # Encode Unicode strings to UTF-8 bytes
    original_prompt_bytes = [prompt.encode('utf-8') for prompt in original_prompt]
    positive_prompt_bytes = [prompt.encode('utf-8') for prompt in positive_prompt]

    # Define compression settings for h5py datasets
    c = {'compression': 'gzip', 'compression_opts': 1}

    # Create an h5 file and store the data in a single folder
    output_folder = 'h5_train' #h5_test for test set
    os.makedirs(output_folder, exist_ok=True)

    with h5py.File(os.path.join(output_folder, f'{doc:04d}.h5'), 'w') as fw:
        fw.create_dataset('img_id', data=img_id, **c)
        fw.create_dataset('real', data=real, **c)
        fw.create_dataset('image_gen0', data=image_gen0, **c)
        fw.create_dataset('image_gen1', data=image_gen1, **c)
        fw.create_dataset('image_gen2', data=image_gen2, **c)
        fw.create_dataset('image_gen3', data=image_gen3, **c)
        fw.create_dataset('original_prompt', data=original_prompt_bytes, **c)
        fw.create_dataset('positive_prompt', data=positive_prompt_bytes, **c)

    print(f"Data written to h5 file for document {doc}")


if __name__ == '__main__':
    # Set the number of CPU workers based on 90% of available CPU cores
    cpu_workers = int(cpu_count() * 0.8)
    print('>> cpu_workers =', cpu_workers)
    print()

    # Get a list of document IDs based on available files in 'train' directory
    train_files = [file[:-8] for file in os.listdir('train') if file.endswith('.parquet')]
    train_docs = [int(doc) for doc in train_files]  # Assuming document IDs are integers

    # Use multiprocessing to process documents in parallel
    with Pool(cpu_workers) as pool:
        pool.map(func, train_docs)



