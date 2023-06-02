import torch 
from pathlib import Path
from PIL import Image

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

def task_ref(root_dir, activity_name_match=None):
    '''For a task/activity listing, return a dict reference of images files'''
    impaths = dict()
    print('Directories of images:', end='\t')
    for subdir in Path(root_dir).iterdir():
        if not subdir.is_dir():
            continue
        if activity_name_match and not activity_name_match in subdir.name:
            continue
        imgs = list(subdir.glob('*jpg')) + list(subdir.glob('*png'))
        impaths[subdir.name] = imgs
        print(subdir.name, f"({len(imgs)} files)", end='\t')
    print()
    return impaths


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