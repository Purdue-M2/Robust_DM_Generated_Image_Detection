import clip
import torch
from PIL import Image

def clip_image_encode(imgs):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-L/14', device=device)
    imgs_size = len(imgs)
    batch_imgs = torch.zeros((imgs_size, 3, 224, 224))
    for i in range(imgs_size):
        batch_imgs[i] = preprocess(Image.fromarray(imgs[i])).unsqueeze(0)  
    batch_imgs_tensor = batch_imgs.to(device)
    
    with torch.no_grad():
        batch_features = model.encode_image(batch_imgs_tensor)

    return batch_features.cpu()
if __name__ == '__main__':

    pass


def clip_text_encode(texts):
    print('Starting text encoding process...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-L/14', device=device)
    # Maximum number of tokens accepted by CLIP model
    max_length = 77
    # Truncate texts to the maximum length accepted by CLIP model
    truncated_texts = [text if len(text) <= max_length else text[:max_length] for text in texts]
    # Tokenize texts. CLIP's tokenize method now works without throwing an error
    text_tokens = clip.tokenize(truncated_texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    print('Text encoding completed.')

    return text_features.cpu().numpy()

if __name__ == '__main__':
    pass



