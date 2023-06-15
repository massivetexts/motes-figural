import torch 
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import clip

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

class FiguralScorer():

    '''A class for scoring figural responses as a batch'''

    def __init__(self, impaths, model, preprocess, device='cpu', contrast_factor=4, crop_bottom=False):
        self.impaths = impaths
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.image_features = None
        self.contrast_factor = contrast_factor
        self.crop_bottom = crop_bottom

    def image_loader(self):
        '''A generator for loading images gradually'''
        return self._generic_image_loader(self.impaths)

    def _generic_image_loader(self, ims):
        for impath in ims:
            im = FiguralImage(impath).contrast(self.contrast_factor)
            if self.crop_bottom:
                im = im.crop(0.85)
            yield im

    def preprocessed_image_loader(self, imgs, batch_size=500):
        '''Pre-process a list of PIL images and stack, for batch inferences.
        If batch_size is provided, will batch the images into tensors of that size, else the generator will yield a single large tensor.
        
         Returns a generator of tensors.
         '''
        if not batch_size:
            batch_size = len(self.impaths)

        batch = []
        while True:
            # build batch with imgs, up to batch_size or until imgs is exhausted
            while len(batch) < batch_size:
                try:
                    batch.append(next(imgs))
                except StopIteration:
                    break
            # if batch is empty, imgs is exhausted
            if not batch:
                break
            # else, preprocess and yield
            else:
                yield torch.tensor(np.stack([self.preprocess(im.im) for im in batch])).to(self.device)
                batch = []
    
    def get_image_features(self, normalize=True, batch_size=500, force=False, use_tqdm=True):
        ''' Get image features for all images in impaths, preprocessed in batches. Saves the features to self.image_features. '''
        if not force and self.image_features is not None:
            return self.image_features
        
        imload = self.image_loader()
        preprocessed_imload = self.preprocessed_image_loader(imload, batch_size=batch_size)
        if use_tqdm:
            preprocessed_imload = tqdm(preprocessed_imload,
                                       total=len(self.impaths)//batch_size + (1 if len(self.impaths) % batch_size else 0),
                                       desc='Getting CLIP features', leave=False)

        for batch in preprocessed_imload:
            with torch.no_grad():
                batch_features = self.model.encode_image(batch)
            if self.image_features is None:
                self.image_features = batch_features
            else:
                self.image_features = torch.cat([self.image_features, batch_features])
        if normalize:
            self.image_features = self.image_features / self.image_features.norm(dim=1, keepdim=True)
        
        return self.image_features
    
    def get_zerosims(self, zero_terms, idx=True, meta=True):
        ''' Return average image similarity to other images
    
        zero terms should be a list of zero-originality terms for the given activity.
        '''
        zero_terms_tokens = clip.tokenize(zero_terms).to(self.device)

        with torch.no_grad():
            txt_features = self.model.encode_text(zero_terms_tokens)

        txt_features = txt_features / txt_features.norm(dim=1, keepdim=True)
        img_features = self.get_image_features(normalize=True)
        simmat = img_features @ txt_features.t()
        x = simmat.cpu().numpy()
        stats = np.vstack([x.min(1), x.mean(1), np.sort(x, axis=1)[:, :3].mean(1)])
        if idx:
            sims = pd.DataFrame(stats, index=['min_zlist', 'mean_zlist', 'lowest3_zlist'], columns=self.impaths).T.reset_index()
            sims.columns = ['path', 'min_zlist', 'mean_zlist', 'lowest3_zlist']
            sims['id'] = sims.path.apply(lambda x: x.stem)
            sims.path = sims.path.apply(lambda x: str(x))
            if meta:
                sims = self._add_meta_to_df(sims)
            return sims
        else:
            return stats

    def get_avg_sims(self, idx=True, meta=True):
        ''' Return average image similarity to other images.
        
        idx=True returns a DataFrame with index=impaths, else returns numpy'''
        img_features = self.get_image_features(normalize=True)
        simmat = img_features @ img_features.t()
        avg_sims = simmat.sum(1).sub(1).div(simmat.shape[0]-1).cpu().numpy()
        if idx:
            sims = pd.Series(avg_sims, index=self.impaths).reset_index()
            sims.columns = ['path', 'avg_sim'] 
            sims['id'] = sims.path.apply(lambda x: x.stem)
            sims.path = sims.path.apply(lambda x: str(x))
            if meta:
                sims = self._add_meta_to_df(sims)
            return sims
        else:
            return avg_sims
        
    def _add_meta_to_df(self, df):
        '''Add metadata to a dataframe'''
        meta = dict(contrast_factor=self.contrast_factor, crop_bottom=self.crop_bottom)
        for k,v in meta.items():
            df[k] = v
        return df
        
    def get_sims_to_target(self, target_path, idx=True, meta=True):
        ''' Return similarity of images to a target image (usually a blank image)'''
        blankloader = self._generic_image_loader([target_path])
        blank_inputs = list(self.preprocessed_image_loader(blankloader, batch_size=None))[0]

        img_features = self.get_image_features(normalize=True)
        with torch.no_grad():
            blank_img_features = self.model.encode_image(blank_inputs)

        # normalize tensors
        blank_img_features = blank_img_features / blank_img_features.norm(dim=1, keepdim=True)

        simmat = img_features @ blank_img_features.t()
        sims = simmat[:, 0].cpu().numpy()

        if idx:
            sims = pd.Series(sims, index=self.impaths).reset_index()
            sims.columns = ['path', 'blank_sim'] 
            sims['id'] = sims.path.apply(lambda x: x.stem)
            sims.path = sims.path.apply(lambda x: str(x))
            if meta:
                sims = self._add_meta_to_df(sims)
            # add columns with meta to sims
            return sims
        else:
            return sims