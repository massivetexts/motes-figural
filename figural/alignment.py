import numpy as np
import cv2
from pdf2image import convert_from_path
from PIL import Image


def name_from_path(path):
    import hashlib
    parent_folder_name = path.parent.stem
    parent_hash = hashlib.sha1(parent_folder_name.encode('utf-8')).hexdigest()[:5]
    name = f"{parent_hash}-{path.stem}"
    return name
class Booklet():

    # cropmap specifies where each activity is
    # # via (page, leftcropprop, topcropprop, rightcropprop, lowercropprop)
    # crops are proportions of page w/h, *based* on the target booklet.
    cropmap = dict()
    cropmap['booklet_b'] = dict(
        activity1=(0, 0.52, 0.09, 0.94, 0.72),
        activity2a=(1,0.089,0.3,0.263,.579),
        activity2b=(1,0.263,0.31,0.437,.579),
        activity2c=(1,0.089,0.579,0.263,.849),
        activity2d=(1,0.263,0.579,0.437,.849),
        activity2e=(1,0.581,0.043,0.755,.312),
        activity2f=(1,0.755,0.042,0.929,.310),
        activity2g=(1,0.581,0.311,0.755,.580),
        activity2h=(1,0.755,0.310,0.929,.580),
        activity2i=(1,0.581,0.580,0.756,.850),
        activity2j=(1,0.755,0.58,0.929,.850),
    )
    cropmap['booklet_a'] = dict(
        activity1=(0, 0.52, 0.09, 0.94, 0.72),
        activity2a=(1,0.075,0.315,0.245,.585),
        activity2b=(1,0.25,0.32,0.42,.585),
        activity2c=(1,0.07,0.59,0.245,.855),
        activity2d=(1,0.25,0.59,0.42,.86),
        activity2e=(1,0.575,0.05,0.75,.323),
        activity2f=(1,0.75,0.06,0.92,.32),
        activity2g=(1,0.575,0.33,0.745,.592),
        activity2h=(1,0.75,0.33,0.91,.59),
        activity2i=(1,0.575,0.60,0.74,.865),
        activity2j=(1,0.75,0.60,0.92,.87),
    )

    def __init__(self, path, first_page=2, dpi=100, booklet='a', poppler_path=None, target_booklet=None):
        ''' first_page should be the first page /after/ title page. i.e. activity 1 page. '''
        self.path = path
        self.first_page = first_page
        self._images = []
        self._originals = []
        self.dpi = dpi
        self.booklet = booklet
        self.poppler_path = poppler_path
        self.read_pdf()
        # determine if the booklet is 1 physical page per PDF page, or 2
        w, h = self._originals[0].size
        if (len(self._originals) > 5) and (w < h):
            self.format = 'single'
            self._single = self._originals
            self._double = []
        else:
            self.format = 'double'
            self._single = []
            self._double = self._originals
        self._targetbooklet = target_booklet

    def read_pdf(self, grayscale=False):
        self._originals = convert_from_path(self.path, self.dpi, first_page=self.first_page,
                                            grayscale=grayscale, poppler_path=self.poppler_path)

    def double(self):
        ''' Read original scans doubled '''
        if self.format == 'single':
            self._double = self._double_from_single(self._originals)
            return self._double
        elif len(self._double) > 0:
            return self._double
        else:
            return self._originals

    def _double_from_single(self, imgs):
        ''' Take a single format list of images and double the pages'''
        collector = []
        n = len(imgs)
        for left, right in zip(imgs[0:-(n%2):2], imgs[1:-(n%2):2]):
            (lw, lh), (rw, rh) = (left.size, right.size)
            new = Image.new(mode=left.mode, size=(lw+rw, max(lh, rh)))
            new.paste(left, (0, max((rh-lh)//2,0) ))
            new.paste(right, (lw, max((lh-rh)//2,0)))
            collector.append(new)
        if n % 2 == 1:
            collector.append(imgs[-1])
        return collector

    def get_all_activities(self, mod=0, square=True, use_originals=False, **kwargs):
        imgs = dict()
        for activity in self.cropmap[f'booklet_{self.booklet}'].keys():
            img = self.get_activity(activity.replace('activity', ''), mod=mod,
                                    square=square, use_originals=use_originals, **kwargs)
            imgs[activity] = img
        return imgs

    def get_page(self, activity, use_originals=False):
        pagenum, leftprop, upperprop, rightprop, lowerprop = self.cropmap[f'booklet_{self.booklet}']['activity'+str(activity)]
        images = self._originals if use_originals else self._images
        return images[pagenum]

    def get_activity(self, activity, mod=0, square=True, use_originals=False, **kwargs):
        '''
        Get a custom crop from the page.
        mod: int in pixels. Use the modifier to tighten (positive number) 
        or loosen (negative number) the bounding box
        square: crop to a square, anchored in center
        '''
        if not use_originals and (len(self._images) == 0):
            print('Running alignment first')
            self.align(**kwargs)
        images = self._originals if use_originals else self._images
        pagenum, leftprop, upperprop, rightprop, lowerprop = self.cropmap[f'booklet_{self.booklet}']['activity'+str(activity)]
        page = images[pagenum]
        w, h = page.size
        left, upper, right, lower = int(leftprop*w), int(upperprop*h), int(rightprop*w), int(lowerprop*h)
        cropped = page.crop((left+mod, upper+mod, right-mod, lower-mod))
        if square:
            w, h = cropped.size
            l = min(w, h)
            crops = [(w-l)//2, (h-l)//2, (w-l)//2+l, (h-l)//2+l]
            cropped = cropped.crop(crops)
        return cropped

    def single(self):
        ''' Read original scans singled '''
        if self.format == 'single':
            return self._originals
        elif len(self._single) > 0:
            return self._single
        else:
            collector = []
            for image in self._originals:
                w, h = image.size
                if w > h:
                    collector.append(image.crop((0,0,w//2,h)))
                    collector.append(image.crop((w//2,0,w,h)))
                else:
                    collector.append(image)
            self._single = collector
            return self._single

    def __getitem__(self, key):
        if len(self._images) == 0:
            return self._originals[key]
        return self._images[key]

    def __len__(self):
        if len(self._images) == 0:
            return len(self._originals)
        return len(self._images)

    def __iter__(self):
        if len(self._images) == 0:
            for page in self._originals:
                yield page
        else:
            for page in self._images:
                yield page

    def align(self, good_match_percent=.15, format='auto', max_features=500, include_matches=False):
        aligned = []
        matches_coll = []

        if format == 'auto':
            format = self.format
        elif (format == 'smart') and (self.format == 'single'):
            # if the scans were single, the 'smart' thing is to align as singles rather than
            # patching together
            format = self.format
        
        if format == 'double':
            sources = self.double()
            targets = self._targetbooklet.double()
        elif format == 'single':
            sources = self.single()
            targets = self._targetbooklet.single()
        elif format == 'smart':
            sd, ss = self.double(), self.single()
            sources = [sd[0]] + ss[2:]
            td, ts = self._targetbooklet.double(), self._targetbooklet.single()
            targets = [td[0]] + ts[2:]

        pagen = min(len(sources), len(targets))

        for source, target in zip(sources[:pagen], targets[:pagen]):
            registered_source, h = alignImages(source, target, good_match_percent=good_match_percent,
                                               max_features=max_features, include_matches=include_matches)
            if include_matches:
                matches, h = h
                matches_coll.append(matches)
            aligned.append(Image.fromarray(registered_source))

        if format == 'double':
            self._images = aligned
        elif format == 'single':
            # patch back together into double format
            #return aligned, h
            self._images = self._double_from_single(aligned)
        elif format == 'smart':
            self._images = [aligned[0]] + self._double_from_single(aligned[1:])
        else:
            raise Exception(f"Not a supported format: {format}")

        if include_matches:
            return self._images, matches_coll
        return self._images

    def composite(self, format='auto', force=False, **kwargs):
        ''' Combine pages with target to see strength of alignment '''
        if format == 'auto':
            format = self.format

        if (len(self._images) == 0) or force:
            self.align(format=format, **kwargs)

        composites = []
        targets = self._targetbooklet.double()
        pagen = min(len(self._images), len(targets))
        for regsource, target in zip(self._images[:pagen], targets[:pagen]):
            mask = Image.new("L", target.size, 128)
            im_comp = Image.composite(regsource, target, mask)
            composites.append(im_comp)
        return composites

def alignImages(source, target, max_features=500, good_match_percent=.15, include_matches=False):
    ''' Image registration with opencv. 
    
    `source` image is the one that will be aligned, to target `target` image.
    
    Returns the remapped source, and the homography map.

    Base on code from https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
    '''

    if type(source) is not np.ndarray:
        source = np.array(source)
    if type(target) is not np.ndarray:
        target = np.array(target)

    # Convert images to grayscale
    if source.ndim == 3:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    if target.ndim == 3:
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(source, None)
    keypoints2, descriptors2 = orb.detectAndCompute(target, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches = list(matches)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]

    if include_matches:
        imMatches = cv2.drawMatches(source, keypoints1, target, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width = target.shape[:2]
    sourceReg = cv2.warpPerspective(source, h, (width, height))

    if include_matches:
        return sourceReg, (Image.fromarray(imMatches), h)
    else:
        return sourceReg, h