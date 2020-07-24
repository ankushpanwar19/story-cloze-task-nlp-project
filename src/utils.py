import pandas as pd
import text_to_uri as ttu
from nltk.stem import PorterStemmer 
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize

def read_data(fpath):
    df = pd.read_csv(fpath)

    columns_rename = {
        'InputStoryid': 'storyid',
        'InputSentence1': 'sentence1',
        'InputSentence2': 'sentence2',
        'InputSentence3': 'sentence3',
        'InputSentence4': 'sentence4',
        'InputSentence5': 'sentence5',
        'RandomFifthSentenceQuiz1': 'ending1',
        'RandomFifthSentenceQuiz2': 'ending2',
        'AnswerRightEnding': 'answer'
    }
    df = df.rename(index=str, columns=columns_rename)
    return df

def run_step(batch, net, tokenizer, loss_name, device, compute_loss=True):
    e1_inputs = tokenizer(text=batch['full_story'], 
                        text_pair=batch['ending1'],
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",)
    e2_inputs = tokenizer(text=batch['full_story'], 
                    text_pair=batch['ending2'],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt")

    for i in e1_inputs.keys():
        e1_inputs[i]=e1_inputs[i].to(device)
        e2_inputs[i]=e2_inputs[i].to(device)

    output = net(e1_inputs, e2_inputs)
    if not compute_loss:
        return output
        
    loss = loss_name(output, batch['labels'])
    return output, loss


def create_common_sense_distance(ending_name, story,embedding):
        # row = story.iloc[idx,1:-1]
        emb_words=embedding.index
        stemmer=PorterStemmer()
        words_e = word_tokenize(story[ending_name])[:-1]
        dist = []
        for i in range(4):
            dis_j = 0
            num = 0
            words_s = word_tokenize(story.iloc[i])[:-1]

            for word_e in words_e:
                max_d = 0
                num += 1
                # cnt = 0
                word_e_process = ttu.standardized_uri('en', word_e)
                if word_e_process in emb_words:
                    word_e_emb = embedding.loc[word_e_process].values
                    for word_s in words_s:
                        if stemmer.stem(word_e) != stemmer.stem(word_s):
                            word_s_process = ttu.standardized_uri('en', word_s)
                            if word_s_process in emb_words:
                                word_s_emb = embedding.loc[word_s_process].values

                                d = cosine(word_e_emb, word_s_emb)
                                if d > max_d:
                                    max_d=d
                dis_j += max_d
            dis_j /= num
            dist.append(dis_j)

        return dist