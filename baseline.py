import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import html
import json
import os
from typing import List

import numpy as np
from pyvi.ViTokenizer import tokenize
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from underthesea import sent_tokenize
