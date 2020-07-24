import numpy as np
import pandas as pd
import nltk
import utils 
from utils import create_common_sense_distance
import pickle
from tqdm import tqdm

nltk.download('punkt')

stories = utils.read_data('data/nlp2_val.csv')
stories_test = utils.read_data('data/nlp2_test.csv')

embedding = pd.read_csv('numberbatch-en-19.08.txt', sep=' ', skiprows=1, header=None)
embedding.set_index(0, inplace=True)

dictionary_commonsense_val={}
for i in tqdm(stories.index):
    story=stories.loc[i]
    idx=story['storyid']
    story_data=story.iloc[1:-1]
    ending1=create_common_sense_distance('ending1',story_data,embedding)
    ending2=create_common_sense_distance('ending2',story_data,embedding)
    dictionary_commonsense_val[idx]={'ending1':ending1,'ending2':ending2}

pickle.dump( dictionary_commonsense_val, open( "data/dictionary_commonsense_val.pickle","wb") )

dictionary_commonsense_test={}
for i in tqdm(stories_test.index):
    story=stories_test.loc[i]
    idx=story['storyid']
    story_data=story.iloc[1:-1]
    ending1=create_common_sense_distance('ending1',story_data,embedding)
    ending2=create_common_sense_distance('ending2',story_data,embedding)
    dictionary_commonsense_test[idx]={'ending1':ending1,'ending2':ending2}

pickle.dump( dictionary_commonsense_test, open( "data/dictionary_commonsense_test.pickle","wb") )