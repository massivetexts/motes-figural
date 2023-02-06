import torch 
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import pandas as pd
from tqdm import tqdm

def autoset_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.has_mps:
        print("CLIP doesn't work on M1 GPUs yet; check here for updates: https://github.com/openai/CLIP/issues/247")
        #device = "mps"
        device = "cpu"
    else:
        device = "cpu"
    return device

class FiguralImage():

    def __init__(self, path):
        self.path = path
        self.im = Image.open(path)
        self._original =  self.im.copy()

    def contrast(self, factor=1):
        if factor > 1:
            enhancer = ImageEnhance.Contrast(self.im)
            self.im = enhancer.enhance(factor)
        return self

    def crop(self, crop_prop=1):
        ''' Crops bottom'''
        if crop_prop < 1:
            w,h = self.im.size
            self.im = self.im.crop((0,0,w, int(crop_prop*h)))
        return self

    def elaboration(self, cutoff=245, raw=False, crop_prop=0.85):
        '''
        Provides an ink count or proportion for image, resized to 200x200 (i.e. 40000px)
        cutoff: Anything darker than a greyscale value of `cutoff` is counted as 'ink'
        raw: return count of ink, else proportion

        TODO: have norms of starting amount of ink
        '''
        greyim = ImageOps.grayscale(self._original).resize((200,200)).crop((0,0,200,int(crop_prop*200)))
        greyvals = np.array(greyim)
        ink_count = (greyvals <= cutoff).sum()

        if raw:
            return ink_count
        else:
            return ink_count / greyvals.size

def image_loader(impaths, contrast_factor=1, crop_bottom=False):
    '''A generator for loading images gradually'''
    for impath in impaths:
        im = FiguralImage(impath).contrast(contrast_factor)
        if crop_bottom:
            im = im.crop(0.85)
        yield im

def preprocess_imlist(imgs, preprocessor, device="cpu"):
    '''Pre-process a list of PIL images and stack, for batch inferences'''
    image_input = []
    for im in tqdm(imgs):
        if type(im) == FiguralImage:
            im = im.im
        try:
            image_input.append(preprocessor(im))
        except:
            print('preprocesserror with an image; no handling yet.')
    return torch.tensor(np.stack(image_input)).to(device)

def collage(impaths, cols, rows='auto', thumbnail=None):
    ''' Collage from a set of image paths'''
    w,h = None, None
    canvas = None
    if rows == 'auto':
        rows = 1 + len(impaths) // cols
    for i, impath in enumerate(impaths[:cols*rows]):
        im = Image.open(impath)
        if w == None:
            w, h = im.size
            canvas = Image.new(mode=im.mode, size=(w*cols, h*rows), color='white')
        col, row = i % cols, i // cols
        canvas.paste(im, (w*col, h*row))
    if thumbnail:
        canvas.thumbnail(thumbnail)
    return canvas



def grammar(x):
    ''' Very basic add indefinite articles (e.g. 'mountain' > 'a mountain', 'egg' > 'an egg') '''
    no_article = ['lightning', 'water'] # mass nouns and such. hand coded based on data
    if (x[-1] == 's') or (x in no_article):
        return x
    elif x[0] in list("aeiou"):
        return f"an {x}"
    else:
        return f"a {x}"

def get_zerosims(image_inputs, model, zero_terms, device='cpu', idx=None):
    ''' Return average image similarity to other images
    
    zero terms should be a list of zero-originality terms for the given activity.
    '''
    import clip
    zero_terms_tokens = clip.tokenize(zero_terms).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        txt_features = model.encode_text(zero_terms_tokens)

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    txt_features = txt_features / txt_features.norm(dim=1, keepdim=True)
    simmat = image_features @ txt_features.t()
    x = simmat.cpu().numpy()
    stats = np.vstack([x.min(1), x.mean(1), np.sort(x, axis=1)[:, :3].mean(1)])

    # return series if index provided
    if idx:
        return pd.DataFrame(stats, index=['min_zlist', 'mean_zlist', 'lowest3_zlist'], columns=idx).T
    else:
        return stats

def get_avg_sims(image_inputs, model, idx=None):
    ''' Return average image similarity to other images'''
    with torch.no_grad():
        image_features = model.encode_image(image_inputs)

    # normalize tensors
    image_features_n = image_features / image_features.norm(dim=1, keepdim=True)

    # cosine similarity
    simmat = image_features_n @ image_features_n.t()
    
    # correct for self-sim
    avg_sims = simmat.sum(1).sub(1).div(simmat.shape[0]-1).cpu().numpy()

    # return series if index provided
    if idx:
        return pd.Series(avg_sims, index=idx)
    else:
        return avg_sims

def similarity_to_target(image_inputs, target_path, model, preprocess, device, idx=None):
    ''' Return similarity of images to a target image (usually a blank image)'''
    blankloader = image_loader([target_path], contrast_factor=4, crop_bottom=True)
    blank_inputs = preprocess_imlist(blankloader, preprocess, device=device)

    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
        blank_img_features = model.encode_image(blank_inputs)

    # normalize tensors
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    blank_img_features = blank_img_features / blank_img_features.norm(dim=1, keepdim=True)

    # cosine similarity
    simmat = image_features @ blank_img_features.t()
    sims = simmat[:, 0].cpu().numpy()
    # return series if index provided
    if idx:
        return pd.Series(sims, index=idx)
    else:
        return sims