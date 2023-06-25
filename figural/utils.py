import torch 
from pathlib import Path
from PIL import Image
import pandas as pd
from scipy.stats import pearsonr
import yaml
from pathlib import Path

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

def grouped_corr(results, groupcols, targetcol, predcol, inverse_col=False):
    # get the correlation of targetcol and predcol, grouped by groupcols, returning both the correlation and p-value as columns
    print(f"# tests without GT:{results[targetcol].isna().sum()}")
    subset = results[~results[targetcol].isna() & ~results[predcol].isna()]
    def corr_func(x):
        if len(x) < 2:
            return (None, None)
        corr, pval = pearsonr(x[targetcol], x[predcol])
        if inverse_col:
            corr = -corr
        return (corr.round(2), pval.round(4))
    return subset.groupby(groupcols).apply(corr_func).apply(pd.Series).rename(columns={0: 'corr', 1: 'pval'})

def prep_audra_gt(df, task):
    df.columns = ['id', 'O']
    df['task'] = task
    df['test'] = 'audra'
    df['activity'] = "Images_" + df['id'].apply(lambda x: x.split('_')[1])
    return df

def task_ref(root_dir, activity_name_match=None, print_dir=False):
    '''For a task/activity listing, return a dict reference of images files'''
    impaths = dict()
    if print_dir:
        print('Directories of images:', end='\t')

    for subdir in Path(root_dir).iterdir():
        if not subdir.is_dir():
            continue
        if activity_name_match and not activity_name_match in subdir.name:
            continue
        imgs = list(subdir.glob('*jpg')) + list(subdir.glob('*png'))
        impaths[subdir.name] = imgs
        if print_dir:
            print(subdir.name, f"({len(imgs)} files)", end='\t')
    if print_dir:
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
    no_article = ['lightning', 'water', 'multiplication'] # mass nouns and such. hand coded based on data
    if (x[-1] == 's') or (x in no_article):
        return x
    elif x[0] in list("aeiou"):
        return f"an {x}"
    else:
        return f"a {x}"
    

def load_data_and_gt(configpath='../../config.yaml', results_path='../../data/metrics/all_data.csv',
                     seed=1234, test_prop=0.1):
    import numpy as np
    from figural.utils import prep_audra_gt

    rng = np.random.default_rng(seed=seed)

    meta = load_config(configpath)
    root_dir = Path(meta['root_dir'])

    all_gt = []
    for test in meta['tests']:
        for task in test['tasks']:
            if test['name'] == 'audra':
                df = pd.read_csv(root_dir / task['truth'], header=None)
                df = prep_audra_gt(df, task['name'])
            else:
                df = pd.read_csv(root_dir / task['truth'])
                df['test'] = 'ttct'
            all_gt.append(df)
    all_gt = pd.concat(all_gt)
    print("Ground Truth size: ", all_gt.shape)

    data = pd.read_csv(results_path, index_col=0)
    data.activity = data.activity.str.replace('.1_common', '', regex=False)
    data = data.merge(all_gt, how='left')

    # exclude items where the registration (aligning the images) failed. These shouldn't be in this data, but doublecheck!
    data['reg_err'] = False
    activity1errs = ['9508e-68724', 'b5b5f-79047', '9508e-47261', '9508e-1438', 'dee97-71558 M', 'b5b5f-74951', 'b5b5f-78819', 'dee97-69433 M', '.DS_Store', 'dee97-64546', 'dee97-51927', 'dee97-74768', 'b5b5f-71542', 'b5b5f-77829', 'b5b5f-80287', 'dee97-76132', '9508e-71576', 'dee97-61601 M', '9508e-71601', 'dee97-69412 M', 'b5b5f-63869', 'dee97-71246', 'b5b5f-79207', 'b5b5f-76123', 'dee97-73321', 'b5b5f-78888', '9508e-67863', 'b5b5f-67808', 'dee97-7534', 'dee97-76285 M', 'b5b5f-74804', 'b5b5f-80188', 'b5b5f-79277', 'b5b5f-76145', 'b5b5f-76543', 'dee97-74015 M', '9508e-72771', 'b5b5f-74369']
    data.loc[data['id'].isin(activity1errs) & (data.activity == 'activity1'), 'reg_err'] = True
    data = data[~data.reg_err]
    for col in ['F', 'O', 'T', 'E', 'R', 'C']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # remove some data errors
    data.loc[data['F'] > 1, 'F'] = np.NaN
    data.loc[data['T'] > 3, 'T'] = np.NaN
    data.loc[data['R'] > 2, 'R'] = np.NaN
    data.loc[data['O'] > 1, 'O'] = np.NaN

    # add - if using - a test/train sample
    data['testset'] = (rng.random(size=len(data)) < test_prop)
    return data

def load_config(configpath='../../config.yaml'):
    # load config.yaml
    with open(configpath) as f:
        meta = yaml.safe_load(f)
    return meta

def load_scorers(model, preprocess, meta, load_features=True, include_cropped=False,
                 save_dir=None, device='cpu'):
    # load all tasks from the config.yaml definition and their scorers.
    # and extract CLIP features now to save time later
    from figural.scoring import FiguralScorer
    from tqdm.auto import tqdm

    loadedtasks = []
    root_dir = Path(meta['root_dir'])

    contrast_factor = 4
    for test in meta['tests']:
        for task in test['tasks']:
            impaths = task_ref(root_dir / task['directory'])

            for activity, paths in tqdm(impaths.items(), desc=f'{test["name"]}/{task["name"]}', leave=False):
                if (test['name'] == 'ttct') and include_cropped:
                    crop_options = [False, True]
                else:
                    crop_options = [False]

                for crop_option in crop_options:
                    if save_dir:
                        # create a filename based on the test, task, activity, and crop option
                        save_location = Path(save_dir) / f'{test["name"]}_{task["name"]}_{activity}_{"crop" if crop_option else "nocrop"}.pt'
                    else:
                        save_location = None
                    scorer = FiguralScorer(paths, model, preprocess, device=device,
                                        contrast_factor=contrast_factor,
                                        save_location=save_location,
                                        crop_bottom=crop_option)
                    if load_features:
                        scorer.get_image_features()

                    loadedtasks.append(dict(
                        test = test['name'],
                        task = task['name'],
                        activity = activity,
                        blank = task['blanks'][activity],
                        paths = paths,
                        scorer = scorer,
                        crop_bottom = crop_option,
                        contrast_factor = contrast_factor
                    ))
    return loadedtasks