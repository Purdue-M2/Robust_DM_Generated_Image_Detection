import h5py
import clip_feature
import os
import numpy as np

def clip_all(folder):
    for doc in range(20):
        try:
            with h5py.File(f'h5_{folder}/{doc:04d}.h5', 'r') as fr:
                img_id = fr['img_id'][:]
                real = fr['real'][:]
                image_gen0 = fr['image_gen0'][:]
                image_gen1 = fr['image_gen1'][:]
                image_gen2 = fr['image_gen2'][:]
                image_gen3 = fr['image_gen3'][:]
                original_prompt = fr['original_prompt'][:]
                positive_prompt = fr['positive_prompt'][:]
        except:
            continue

        print(f'size(h5_{folder}/{doc:04d}.h5) : {len(img_id)}')

        original_prompt = [str(text) for text in original_prompt]
        positive_prompt = [str(text) for text in positive_prompt]

        real_img_clip = clip_feature.clip_image_encode(real)
        image_gen0_clip = clip_feature.clip_image_encode(image_gen0)
        image_gen1_clip = clip_feature.clip_image_encode(image_gen1)
        image_gen2_clip = clip_feature.clip_image_encode(image_gen2)
        image_gen3_clip = clip_feature.clip_image_encode(image_gen3)
        original_prompt_clip = clip_feature.clip_text_encode(original_prompt)
        positive_prompt_clip = clip_feature.clip_text_encode(positive_prompt)

        c = {'compression': 'gzip', 'compression_opts': 1}
        with h5py.File(f'clip_{folder}/{doc:04d}.h5', 'w') as fw:
            fw.create_dataset('img_id', data=img_id, **c)
            fw.create_dataset('original_prompt', data=np.array(original_prompt_clip), **c)
            fw.create_dataset('image_gen0', data=image_gen0_clip, **c)
            fw.create_dataset('image_gen1', data=image_gen1_clip, **c)
            fw.create_dataset('image_gen2', data=image_gen2_clip, **c)
            fw.create_dataset('image_gen3', data=image_gen3_clip, **c)
            fw.create_dataset('positive_prompt', data=np.array(positive_prompt_clip), **c)

if __name__ == '__main__':
    clip_all('train')  # or 'train' depending on your use case
    clip_all('val')  
