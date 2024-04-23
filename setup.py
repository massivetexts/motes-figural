import os
from setuptools import setup

setup(name='MOTES Figural',
      packages=["figural"],
      version='0.0.1',
      description="Tools for scoring figural originality responses.",
      url="https://github.com/massivetexts/motes-figural",
      author="Peter Organisciak",
      author_email="peter.organisciak@du.edu",
      license="MIT",
      install_requires=["pytesseract", "opencv-python", "pdf2image", "ftfy", 'regex', "tqdm",
                        "clip @ git+https://github.com/openai/CLIP.git", "xgboost", "torch"],
      dependency_links=[
        'git+https://github.com/openai/CLIP.git#egg=clip'
    ]
)