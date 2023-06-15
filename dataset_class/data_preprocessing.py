import gc, os
import pandas as pd
import numpy as np
import torch
import configuration as configuration
from torch import Tensor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from tqdm.auto import tqdm


def ner_tokenizing(cfg: configuration.CFG, text: str):
    """
    Preprocess text for NER Pipeline
    if you want to set param 'return_offsets_mapping' == True, you must use FastTokenizer
    you must use PretrainedTokenizer which is supported FastTokenizer
    Converting text to torch.Tensor will be done in Custom Dataset Class
    Params:
        return_offsets_mapping:
            - bool, defaults to False
            - Whether or not to return (char_start, char_end) for each token.
            => useful for NER Task
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        text: text from dataframe or any other dataset, please pass str type
    """
    inputs = cfg.tokenizer(
        text,
        return_offsets_mapping=True,  # only available for FastTokenizer by Rust, not erase /n, /n/n
        max_length=cfg.max_len,
        padding='max_length',
        truncation=True,
        return_tensors=None,
        add_special_tokens=True,
    )
    return inputs


def subsequent_tokenizing(cfg: configuration.CFG, text: str) -> any:
    """
    Tokenize input sentence to longer sequence than common tokenizing
    Append padding strategy NOT Apply same max length, similar concept to dynamic padding
    Truncate longer sequence to match LLM max sequence
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        text: text from dataframe or any other dataset, please pass str type
    Reference:
        https://www.kaggle.com/competitions/AI4Code/discussion/343714
        https://github.com/louis-she/ai4code/blob/master/tests/test_utils.py#L6

    """
    inputs = cfg.tokenizer.encode_plus(
        text,
        max_length=128,
        padding=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False,  # No need to special token to subsequent text sequence
    )
    return inputs['input_ids']


def adjust_sequences(sequences: list, max_len: int):
    """
    Similar to dynamic padding concept
    Append slicing index from original, because original source code is implemented weired
    So it generates some problem for applying very longer sequence
    Add -1 value to slicing index, so we can get result what we want
    Args:
        sequences: list of each cell's token sequence in one unique notebook id, must pass tokenized sequence input_ids
        => sequences = [[1,2,3,4,5,6], [1,2,3,4,5,6], ... , [1,2,3,4,5]]
        max_len: max length of sequence into LLM Embedding Layer, default is 2048 for DeBERTa-V3-Large
    Reference:
         https://github.com/louis-she/ai4code/blob/master/ai4code/utils.py#L70
    """
    length_of_seqs = [len(seq) for seq in sequences]
    total_len = sum(length_of_seqs)
    cut_off = total_len - max_len
    if cut_off <= 0:
        return sequences, length_of_seqs

    for _ in range(cut_off):
        max_index = length_of_seqs.index(max(length_of_seqs))
        length_of_seqs[max_index] -= 1
    sequences = [sequences[i][:l-1] for i, l in enumerate(length_of_seqs)]

    return sequences, length_of_seqs


def subsequent_decode(cfg: configuration.CFG, token_list: list) -> any:
    """
    Return decoded text from subsequent_tokenizing & adjust_sequences
    For making prompt text
    Args:
        cfg: configuration.CFG, needed to load tokenizer from Huggingface AutoTokenizer
        token_list: token list from subsequent_tokenizing & adjust_sequences
    """
    output = cfg.tokenizer.decode(token_list)
    return output


def check_null(df: pd.DataFrame) -> pd.Series:
    """ check if input dataframe has null type object...etc """
    return df.isnull().sum()


def kfold(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """ KFold """
    fold = KFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(df)):
        df.loc[vx, "fold"] = int(num)
    return df


def group_kfold(df: pd.DataFrame, cfg: configuration.CFG) -> pd.DataFrame:
    """ GroupKFold """
    fold = GroupKFold(
        n_splits=cfg.n_folds,
    )
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(X=df, y=df['pct_rank'], groups=df['ancestor_id'])):
        df.loc[vx, "fold"] = int(num)
    return df


def stratified_groupkfold(df: pd.DataFrame, cfg: configuration.CFG) -> pd.DataFrame:
    """ Stratified Group KFold from sklearn.model_selection """
    fold = StratifiedGroupKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cfg.seed
    )
    df['fold'] = -1
    for num, (tx, vx) in enumerate(fold.split(df, df['target'], df['topics_ids'])):
        df.loc[vx, 'fold'] = int(num)  # Assign fold group number

    df['fold'] = df['fold'].astype(int)  # type casting for fold value
    return df


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data_folder from csv file like as train.csv, test.csv, val.csv
    """
    df = pd.read_csv(data_path)
    return df


def get_n_grams(train: pd.DataFrame, n_grams: float, top_n: float = 10):
    """
    Return Top-10 n-grams from the each discourse type
    Source code from Reference URL, but I modified some part
    you can compare each discourse type's result, we can find really unique words for each discourse type
    Args:
        train: original train dataset from competition
        n_grams: set number of n-grams (window size)
        top_n: value of how many result do you want to see, sorted by descending counts value, default is 10

    [Reference]
    https://www.kaggle.com/code/erikbruin/nlp-on-student-writing-eda/notebook
    """
    df_words = pd.DataFrame()
    for dt in tqdm(train['discourse_type'].unique()):
        df = train.query('discourse_type == @dt')
        texts = df['discourse_text'].tolist()
        vec = CountVectorizer(
            lowercase = True,
            stop_words = 'english',
            ngram_range=(n_grams, n_grams)
        ).fit(texts)
        bag_of_words = vec.transform(texts)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        cvec_df = pd.DataFrame.from_records(words_freq,\
                                            columns= ['words', 'counts']).sort_values(by="counts", ascending=False)
        cvec_df.insert(0, "Discourse_type", dt)
        cvec_df = cvec_df.iloc[:top_n,:]
        df_words = df_words.append(cvec_df)
    return df_words


def get_ner_labels(df: pd.DataFrame, text_df: pd.DataFrame) -> None:
    """
    Make NER labels feature for each token in sequence
    Args:
        df: original train dataset from train.csv
        text_df: text dataset from train.txt
    Reference:
        https://www.kaggle.com/code/cdeotte/pytorch-bigbird-ner-cv-0-615/notebook
    """
    all_entities = []
    for idx, i in enumerate(df.iterrows()):
        if idx % 100 == 0:
            print(idx, ', ', end='')
        total = i[1]['text'].split().__len__()
        entities = ["O"] * total
        for j in df[df['id'] == i[1]['id']].iterrows():
            discourse = j[1]['discourse_type']
            list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
            entities[list_ix[0]] = f"B-{discourse}"
            for k in list_ix[1:]:
                entities[k] = f"I-{discourse}"
        all_entities.append(entities)
    text_df['entities'] = all_entities
    text_df.to_csv('train_NER.csv',index=False)


def labels2ids():
    """
    Encoding labels to ids for neural network with BIO Styles
    labels2dict = {
    'O': 0, 'B-Lead': 1, 'I-Lead': 2, 'B-Position': 3, 'I-Position': 4, 'B-Claim': 5,
    'I-Claim': 6, 'B-Counterclaim': 7, 'I-Counterclaim': 8, 'B-Rebuttal': 9, 'I-Rebuttal': 10,
    'B-Evidence': 11, 'I-Evidence': 12, 'B-Concluding Statement': 13, 'I-Concluding Statement': 14
     }
    """
    output_labels = [
        'O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim',
        'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement',
        'I-Concluding Statement'
    ]
    labels_to_ids = {v: k for k, v in enumerate(output_labels)}
    return labels_to_ids


def ids2labels():
    """
    Decoding labels to ids for neural network with BIO Styles
    labels2dict = {
    'O': 0, 'B-Lead': 1, 'I-Lead': 2, 'B-Position': 3, 'I-Position': 4, 'B-Claim': 5,
    'I-Claim': 6, 'B-Counterclaim': 7, 'I-Counterclaim': 8, 'B-Rebuttal': 9, 'I-Rebuttal': 10,
    'B-Evidence': 11, 'I-Evidence': 12, 'B-Concluding Statement': 13, 'I-Concluding Statement': 14
     }

    """
    output_labels = [
        'O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim',
        'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement',
        'I-Concluding Statement'
    ]
    ids_to_labels = {k: v for k, v in enumerate(output_labels)}
    return ids_to_labels


def txt2df(data_path: str) -> pd.DataFrame:
    """
    Convert txt to dataframe for inference & submission
    Args:
        data_path: txt file path from competition host
    Reference:
        https://www.kaggle.com/code/chasembowers/sequence-postprocessing-v2-67-lb/notebook
    """
    text_id, text = [], []
    for f in tqdm(list(os.listdir(data_path))):
        text_id.append(f.replace('.txt', ''))
        text.append(open(data_path + f, 'r').read())

    df = pd.DataFrame({'id': text_id, 'text': text})
    return df


def sequence_length(cfg: configuration.CFG, text_list: list) -> list:
    """ Get sequence length of all text data for checking statistics value """
    length_list = []
    for text in tqdm(text_list):
        tmp_text = ner_tokenizing(cfg, text)['attention_mask']
        length_list.append(tmp_text.count(1))
    return length_list


def sorted_quantile(array: list, q: float):
    """
    This is used to prevent re-sorting to compute quantile for every sequence.
    Args:
        array: list of element
        q: accumulate probability which you want to calculate spot
    Reference:
        https://stackoverflow.com/questions/60467081/linear-interpolation-in-numpy-quantile
        https://www.kaggle.com/code/chasembowers/sequence-postprocessing-v2-67-lb/notebook
    """
    array = np.array(array)
    n = len(array)
    index = (n - 1) * q
    left = np.floor(index).astype(int)
    fraction = index - left
    right = left
    right = right + (fraction > 0).astype(int)
    i, j = array[left], array[right]
    return i + (j - i) * fraction


def split_mapping(unsplit):
    """ Return array which is mapping character index to index of word in list of split() words """
    splt = unsplit.split()
    offset_to_wordidx = np.full(len(unsplit),-1)
    txt_ptr = 0
    for split_index, full_word in enumerate(splt):
        while unsplit[txt_ptr:txt_ptr + len(full_word)] != full_word:
            txt_ptr += 1
        offset_to_wordidx[txt_ptr:txt_ptr + len(full_word)] = split_index
        txt_ptr += len(full_word)
    return offset_to_wordidx