# -*- coding: utf-8 -*-

# Checks for required Python packages and installs them if not already installed.
!pip install --quiet importlib
import importlib

req_packages:list = ['typing','numpy','pandas','re','os','shutil','zlib','collections','warnings','difflib','fuzzywuzzy','jellyfish','python-Levenshtein','scipy','nltk','pandarallel']

for package_name in req_packages:
  try:
    importlib.import_module(package_name)
  except:
    try:
      !pip install --quiet {package_name}
    except Exception as e:
      print(f"Required package {package_name} was not installed!: {str(e)}")
del importlib
print("All required packages are installed.")

# Import installed packages.
import time
from typing import List,Counter
import numpy as np
import pandas as pd
import re
import os
import shutil

import zlib
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

from difflib import SequenceMatcher as difflib_sequencematcher
from fuzzywuzzy import fuzz
import jellyfish
import Levenshtein
from scipy.spatial.distance import jensenshannon

import nltk
nltk.download(['punkt', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk.metrics.distance import edit_distance

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
print("All required packages are imported.")

class TextScoring():
  """
  A class for computing text similarity metrics between elements in a pandas DataFrame.

  """
  def __init__(self,dataframe_object:pd.DataFrame,output_folder:str='Mapped_Attributes',col_name_1:str='doc1_elements',col_name_2:str='doc2_elements',metrics_list:list=['all']):
    """
    Initialize the TextScoring class.

    Args:
    - dataframe_object (DataFrame): Pandas DataFrame object containing the data to process.
    - output_folder (str, optional): Name of the output folder where results will be stored. Defaults to 'Mapped_Attributes'.
    - col_name_1 (str, optional): Name of the column in the DataFrame representing document 1 elements. Defaults to 'doc1_elements'.
    - col_name_2 (str, optional): Name of the column in the DataFrame representing document 2 elements. Defaults to 'doc2_elements'.
    - metrics_list (list, optional): List of specific text scoring metrics to use. Defaults to ['all'].
      Options for metrics:
      - 'basic_jaccard_similarity': Basic Jaccard similarity coefficient.
      - 'weighted_jaccard_similarity': Weighted Jaccard similarity coefficient.
      - 'damerau_levenshtein_distance': Damerau-Levenshtein distance.
      - 'dice_coefficient': Dice coefficient.
      - 'sequencematcher': SequenceMatcher similarity.
      - 'editdistance': Edit distance (Levenshtein distance).
      - 'fuzz_partial_ratio': Partial ratio using fuzzy matching.
      - 'fuzz_ratio': Ratio using fuzzy matching.
      - 'fuzz_token_set_ratio': Token set ratio using fuzzy matching.
      - 'fuzz_token_sort_ratio': Token sort ratio using fuzzy matching.
      - 'fuzz_wratio': Weighted ratio using fuzzy matching.
      - 'hamming_distance': Hamming distance.
      - 'jaro_similarity': Jaro similarity.
      - 'jaro_winkler_similarity': Jaro-Winkler similarity.
      - 'jensen_shannon_divergence': Jensen-Shannon divergence.
      - 'levenshtein_distance': Levenshtein distance.
      - 'levenshtein_similarity': Levenshtein similarity.
      - 'minhash_containment_distance': MinHash containment distance.
      - 'monge_elkan_similarity': Monge-Elkan similarity.
      - 'normalized_compression_distance': Normalized compression distance.
      - 'overlap_coefficient': Overlap coefficient.
      - 'ratcliff_obershelp_similarity': Ratcliff-Obershelp similarity.

    Returns:
    - None
    """
    # Define text scoring metrics and corresponding methods
    self.text_metrics:dict={
      'get_basic_jaccard_similarity':self.get_basic_jaccard_similarity,
      'get_weighted_jaccard_similarity':self.get_weighted_jaccard_similarity,
      'get_damerau_levenshtein_distance':self.get_damerau_levenshtein_distance,
      'get_dice_coefficient':self.get_dice_coefficient,
      'get_difflib_sequencematcher':self.get_difflib_sequencematcher,
      'get_editdistance':self.get_editdistance,
      'get_fuzz_partial_ratio':self.get_fuzz_partial_ratio,
      'get_fuzz_ratio':self.get_fuzz_ratio,
      'get_fuzz_token_set_ratio':self.get_fuzz_token_set_ratio,
      'get_fuzz_token_sort_ratio':self.get_fuzz_token_sort_ratio,
      'get_fuzz_wratio':self.get_fuzz_wratio,
      'get_hamming_distance':self.get_hamming_distance,
      'get_jaro_similarity':self.get_jaro_similarity,
      'get_jaro_winkler_similarity':self.get_jaro_winkler_similarity,
      'get_jensen_shannon_divergence':self.get_jensen_shannon_divergence,
      'get_levenshtein_distance':self.get_levenshtein_distance,
      'get_levenshtein_similarity':self.get_levenshtein_similarity,
      'get_minhash_containment_distance':self.get_minhash_containment_distance,
      'get_monge_elkan_similarity':self.get_monge_elkan_similarity,
      'get_normalized_compression_distance':self.get_normalized_compression_distance,
      'get_overlap_coefficient':self.get_overlap_coefficient,
      'get_ratcliff_obershelp_similarity':self.get_ratcliff_obershelp_similarity,
    }

    self.df1:pd.DataFrame = dataframe_object
    self.output_folder:str = self.trim_characters(stxt=output_folder).replace(' ','_')
    self.col_name_1:str = col_name_1
    self.col_name_2:str = col_name_2

    # Determine which scoring metrics to use based on metrics_list
    self.scoring_metrics:dict = self.text_metrics if 'all' in metrics_list else {'get_'+str(k):self.text_metrics['get_'+str(k)] for k in metrics_list if ('get_'+str(k) in self.text_metrics.keys())}

  def __repr__(self):
    """
    Returns a string representation of the class instance.
    """
    return f"TextScoring()"

  def __str__(self):
    """
    Returns a description of the class.
    """
    return "Class Similarity Score for Elements given in DataFrame"

  def get_jaro_winkler_similarity(self,sent_1:str,sent_2:str)->float:
    """
    Calculates the Jaro-Winkler similarity between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: similarity score (percentage).
    """
    return (jellyfish.jaro_winkler_similarity(sent_1,sent_2))*100

  def get_minhash_containment_distance(self,sent_1:str,sent_2:str)->float:
    """
    Calculates the MinHash containment distance between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: MinHash containment distance score.
    """
    sent_1_len,sent_2_len,sent_1_2_len=len(sent_1),len(sent_2),len(set(sent_1).intersection(set(sent_2)))
    if sent_1_len>sent_2_len:
      return sent_1_2_len / sent_1_len
    else:
      return sent_1_2_len / sent_2_len

  def get_difflib_sequencematcher(self,sent_1:str,sent_2:str)->float:
    """
    Computes the similarity ratio between two strings using difflib's SequenceMatcher.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: Similarity ratio between the two strings, scaled to percentage (0 to 100).
      The ratio measures how similar the sequences are, where 100 means identical.
    """
    return difflib_sequencematcher(isjunk=None,autojunk=True,a=sent_1,b=sent_2).ratio()*100

  def get_fuzz_ratio(self,sent_1:str,sent_2:str)->float:
    """
    Computes the fuzz ratio between two strings using Levenshtein Distance.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: similarity score
    """
    return fuzz.ratio(sent_1,sent_2)

  def get_fuzz_partial_ratio(self,sent_1:str,sent_2:str)->float:
    """
    Computes the partial fuzz ratio between two strings using partial matching.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: similarity score
    """
    return fuzz.partial_ratio(sent_1,sent_2)

  def get_fuzz_token_sort_ratio(self,sent_1:str,sent_2:str)->float:
    """
    Computes the fuzz ratio between two strings after sorting internal tokens.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: similarity score
    """
    return fuzz.token_sort_ratio(sent_1,sent_2)

  def get_fuzz_token_set_ratio(self,sent_1:str,sent_2:str)->float:
    """
    Computes the fuzz ratio between two strings using token sets.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: similarity score
    """
    return fuzz.token_set_ratio(sent_1,sent_2)

  def get_fuzz_wratio(self,sent_1:str,sent_2:str)->float:
    """
    Computes the WRatio between two strings using token sorting.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: similarity score
    """
    return fuzz.WRatio(sent_1,sent_2)

  def get_editdistance(self,sent_1:str,sent_2:str)->float:
    """
    Computes the Edit Distance between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: editdistance score
    """
    return 1 - (edit_distance(s1=sent_1,s2=sent_2) / max(len(sent_1),len(sent_2)))

  def get_basic_jaccard_similarity(self,sent_1:str,sent_2:str)->float:
    """
    Computes the basic Jaccard similarity coefficient between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: Basic Jaccard similarity coefficient score.
      The score is computed as the size of the intersection of the token sets
      divided by the size of the union of the token sets.
      Returns 0.0 if the union set size is zero.
    """
    set_1:set = set(self.get_nltk_word_tokenize(txt=sent_1))
    set_2:set = set(self.get_nltk_word_tokenize(txt=sent_2))
    union_set:int = len(set_1.union(set_2))
    return len(set_1.intersection(set_2)) / union_set if union_set > 0 else 0.0

  def get_weighted_jaccard_similarity(self,sent_1:str,sent_2:str)->float:
    """
    Computes the weighted Jaccard similarity coefficient between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: Weighted Jaccard similarity coefficient score.
      The score is computed as the weighted sum of the intersection of token frequencies
      divided by the weighted sum of the union of token frequencies.
    """
    set_1:set = set(self.get_nltk_word_tokenize(txt=sent_1))
    set_2:set = set(self.get_nltk_word_tokenize(txt=sent_2))
    intersection_set:set = set_1.intersection(set_2)
    union_set:set = set_1.union(set_2)
    weighted_intersection:int = sum(min(sent_1.count(token),sent_2.count(token)) for token in intersection_set)
    weighted_union:int = sum(max(sent_1.count(token),sent_2.count(token)) for token in union_set)
    return weighted_intersection/weighted_union

  def get_dice_coefficient(self,sent_1:str,sent_2:str)->float:
    """
    Computes the Dice coefficient between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: Dice coefficient score.
      The score is computed as twice the size of the intersection of the token sets
      divided by the sum of the sizes of both token sets.
    """
    set_1:set = set(self.get_nltk_word_tokenize(txt=sent_1))
    set_2:set = set(self.get_nltk_word_tokenize(txt=sent_2))
    intersection_set:int = len(set_1.intersection(set_2))
    return (2.0 * intersection_set) / (len(set_1) + len(set_2))

  def get_overlap_coefficient(self,sent_1:str,sent_2:str)->float:
    """
    Computes the overlap coefficient between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: Overlap coefficient score.
      The score is computed as the size of the intersection of the token sets
      divided by the size of the smaller of the two token sets.
    """
    set_1:set = set(self.get_nltk_word_tokenize(txt=sent_1))
    set_2:set = set(self.get_nltk_word_tokenize(txt=sent_2))
    intersection_set:int = len(set_1.intersection(set_2))
    return intersection_set / min(len(set_1),len(set_2))

  def get_levenshtein_distance(self,sent_1:str,sent_2:str)->float:
    """
    Computes the Levenshtein distance between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: Levenshtein distance between the two strings.
      The distance represents the minimum number of single-character edits
      (insertions, deletions, substitutions) required to change one string into the other.
    """
    return jellyfish.levenshtein_distance(sent_1,sent_2)

  def get_damerau_levenshtein_distance(self,sent_1:str,sent_2:str)->float:
    """
    Computes the Damerau-Levenshtein distance between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: Damerau-Levenshtein distance between the two strings.
      The distance measures the minimum number of operations (insertions, deletions,
      substitutions, and transpositions of adjacent characters) required to transform
      one string into the other.
    """
    return jellyfish.damerau_levenshtein_distance(sent_1,sent_2)

  def get_hamming_distance(self,sent_1:str,sent_2:str)->float:
    """
    Computes the Hamming distance between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: Hamming distance between the two strings.
      The distance is the number of positions at which the corresponding characters
      are different between the two strings. The strings must be of equal length.
    """
    return jellyfish.hamming_distance(sent_1,sent_2)

  def get_jaro_similarity(self,sent_1:str,sent_2:str)->float:
    """
    Computes the Jaro similarity between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: Jaro similarity score between the two strings.
      The score ranges from 0 (no similarity) to 1 (exact match),
      measuring the similarity between the strings based on character matching.
    """
    return jellyfish.jaro_similarity(sent_1,sent_2)

  def get_levenshtein_similarity(self,sent_1:str,sent_2:str)->float:
    """
    Computes the normalized Levenshtein similarity between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: Normalized Levenshtein similarity score between the two strings.
      The score ranges from 0 (no similarity) to 1 (exact match),
      computed as 1 - (Levenshtein distance / maximum length of the two strings).
    """
    return 1 - (Levenshtein.distance(sent_1,sent_2) / max(len(sent_1),len(sent_2)))

  def get_normalized_compression_distance(self,sent_1:str,sent_2:str)->float:
    """
    Computes the Normalized Compression Distance (NCD) between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: Normalized Compression Distance (NCD) score between the two strings.
      The score is computed as 1 - (compressed_combined - min(compressed_1, compressed_2)) /
      max(compressed_1, compressed_2), where compressed lengths are obtained using zlib compression.
      NCD measures the similarity between strings based on their compressed size,
      normalized by the size of the smallest compressed string.
    """
    combined_length:int = len(sent_1 + sent_2)
    compressed_1:int = len(zlib.compress(sent_1.encode(encoding='utf-8',errors='replace')))
    compressed_2:int = len(zlib.compress(sent_2.encode(encoding='utf-8',errors='replace')))
    compressed_combined:int = len(zlib.compress((sent_1 + sent_2).encode(encoding='utf-8',errors='replace')))
    ncd_score:float = (compressed_combined - min(compressed_1,compressed_2)) / max(compressed_1,compressed_2)
    return 1 - ncd_score

  def get_nltk_word_tokenize(self,txt:str)->list:
    """
    Tokenizes a given text using NLTK's word_tokenize function and filters out non-alphanumeric tokens.

    Args:
    - txt (str): Input text to tokenize.

    Returns:
    - list: List of alphanumeric tokens extracted from the text.
    """
    return [x for x in word_tokenize(text=txt,language='english') if x.isalnum()]

  def get_token_probabilities(self,stxt:str)->dict:
    """
    Computes the probability of each token in the given text.

    Args:
    - stxt (str): Input text to analyze.

    Returns:
    - dict: Dictionary where keys are tokens and values are their probabilities (relative frequencies).
      The probability of each token is computed as its count divided by the total number of tokens.
    """
    tokens:List[str] = self.get_nltk_word_tokenize(txt=stxt)
    token_counts:Counter[str] = Counter(tokens)
    total_count:int = len(tokens)
    return {token: count / total_count for token,count in token_counts.items()}

  def get_jensen_shannon_divergence(self,sent_1:str,sent_2:str)->float:
    """
    Computes the Jensen-Shannon divergence between the token probability distributions of two texts.

    Args:
    - sent_1 (str): First text.
    - sent_2 (str): Second text.

    Returns:
    - float: Jensen-Shannon divergence score between the token distributions of the two texts.
      The score ranges from 0 (identical distributions) to 1 (completely different distributions).
      It measures the similarity between two probability distributions using the Jensen-Shannon divergence metric.
    """
    text_prob_1:dict = self.get_token_probabilities(stxt=sent_1)
    text_prob_2:dict = self.get_token_probabilities(stxt=sent_2)
    tokens:set = set(text_prob_1.keys()).union(set(text_prob_2.keys()))
    text_probs_1:np.ndarray = np.array([text_prob_1.get(token, 0) for token in tokens])
    text_probs_2:np.ndarray = np.array([text_prob_2.get(token, 0) for token in tokens])
    return 1 - jensenshannon(p=text_probs_1,q=text_probs_2)

  def get_ratcliff_obershelp_similarity(self,sent_1:str,sent_2:str)->float:
    """
    Computes the Ratcliff/Obershelp similarity between two strings.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.

    Returns:
    - float: Ratcliff/Obershelp similarity score between the two strings.
      The score ranges from 0 (no similarity) to 1 (exact match),
      computed as the length of the longest common subsequence divided by the maximum length of the two strings.
    """
    if not sent_1 or not sent_2:
      return 0.0
    match_:list = [[0] * (len(sent_2) + 1) for _ in range(len(sent_1) + 1)]
    for i in range(1,len(sent_1) + 1):
      for j in range(1,len(sent_2) + 1):
        if sent_1[i - 1] == sent_2[j - 1]:
          match_[i][j] = match_[i - 1][j - 1] + 1
        else:
          match_[i][j] = max(match_[i][j - 1],match_[i - 1][j])
    return match_[len(sent_1)][len(sent_2)] / max(len(sent_1),len(sent_2))

  def get_monge_elkan_similarity(self,sent_1:str,sent_2:str,n_pairs:int=2)->float:
    """
    Computes the Monge-Elkan similarity between two strings based on n-gram similarity.

    Args:
    - sent_1 (str): First string.
    - sent_2 (str): Second string.
    - n_pairs (int): Number of n-grams to use for similarity comparison (default is 2).

    Returns:
    - float: Monge-Elkan similarity score between the two strings.
      The score ranges from 0 to 1, where higher values indicate greater similarity.
      It measures similarity based on the best match of n-grams between the two strings,
      using 1 minus the normalized edit distance.
    """
    similarity_total:float = 0.0
    text_total_pairs:float = 0.0
    for gram_1 in ngrams(sequence=self.get_nltk_word_tokenize(txt=sent_1),n=n_pairs):
      best_sim_:float = 0
      for gram_2 in ngrams(sequence=self.get_nltk_word_tokenize(txt=sent_2),n=n_pairs):
        sim_:float = self.get_editdistance(sent_1=gram_1,sent_2=gram_2)
        if sim_ > best_sim_:
          best_sim_ = sim_
      similarity_total += best_sim_
      text_total_pairs += 1
    similarity_score:float = (similarity_total / text_total_pairs) if text_total_pairs > 0 else 0.0
    return similarity_score

  def trim_characters(self,stxt:str='')->str:
    """
    Removes non-alphanumeric characters from a string.

    Args:
    - stxt (str): Input string.

    Returns:
    - str: String with non-alphanumeric characters removed.
    """
    return re.compile(pattern=r'\s+').sub(repl=r' ',string=str(re.compile(pattern=r'[^a-zA-Z\d]').sub(repl=r' ',string=str(stxt)))).strip()

  def create_final_folder(self)->None:
    """
    Creates Output Folder.
    If the folder already exists, it is first removed along with all its contents, and then a new empty folder is created.

    Returns:
    - None
    """
    if os.path.exists(path=self.output_folder):
      # Forcefully delete a directory and its contents
      shutil.rmtree(path=self.output_folder)
    os.mkdir(path=self.output_folder)
    return None

  def create_final_zip(self)->None:
    """
    Creates a ZIP archive of all the contents.
    This method walks through the directory structure, adds all files to a ZIP archive, and stores it as '.zip'.

    Returns:
    - None
    """
    # Creates ZIP
    with zipfile.ZipFile(file=self.output_folder+'.zip',mode='w',compression=zipfile.ZIP_DEFLATED) as zip_file:
      for all_root,all_dirs,all_files in os.walk(self.output_folder):
        for file_1 in all_files:
          temp_file_path = os.path.join(all_root,file_1)
          zip_file.write(
            temp_file_path,
            os.path.relpath(temp_file_path,self.output_folder)
            )

    zip_file_path:str = self.output_folder+'.zip'
    target_folder_path:str = self.output_folder
    os.rename(os.path.abspath(zip_file_path),os.path.abspath(os.path.join(target_folder_path,zip_file_path)))
    return None

  def pre_processing_text_values(self,txt:str='',is_lower:bool=True,remove_characters:bool=True)->str:
    """
    Pre-processes text values by lowercasing, removing non-alphanumeric characters, and tokenizing.

    Args:
    - txt (str): Input text.
    - is_lower (bool, optional), default = True: Convert text to lowercase.
    - remove_characters (bool, optional), default = True: Remove non-alphanumeric characters.

    Returns:
    - str: Pre-processed text.
    """
    if is_lower:
      txt:str=str(txt).lower().strip()
    else:
      txt:str=str(txt).strip()

    if remove_characters:
      txt:str=self.trim_characters(stxt=txt)
    else:
      pass

    return ' '.join([x for x in word_tokenize(txt) if x.isalnum()])

  def get_all_similarity_scores(self,row1:pd.Series)->pd.Series:
    """
    Calculates similarity scores between Doc1 and Doc2 Elements using various metrics.

    Args:
    - row1 (pd.Series): Input row containing text columns for comparison.

    Returns:
    - pd.Series: Series with similarity scores appended.
    """
    doc1_,doc2_=row1[self.col_name_1],row1[self.col_name_2]
    for metric_name,metric_func in self.scoring_metrics.items():
      if metric_name in ['get_minhash_containment_distance']:
        row1[metric_name.replace('get_','high_score_')]=round(number=metric_func(sent_1=doc1_,sent_2=doc2_)*100,ndigits=4)
      elif metric_name in ['get_monge_elkan_similarity']:
        for i in range(1,4):
          row1[metric_name.replace('get_','high_score_')+'_'+str(i)]=round(number=metric_func(sent_1=doc1_,sent_2=doc2_,n_pairs=i),ndigits=4)
      elif metric_name in [
          'get_editdistance',
          'get_levenshtein_distance',
          'get_damerau_levenshtein_distance',
          'get_hamming_distance',
          'get_normalized_compression_distance',
          'get_difflib_sequencematcher',
          'get_minhash_containment_distance',
      ]:
        row1[metric_name.replace('get_','low_score_')]=round(number=metric_func(sent_1=doc1_,sent_2=doc2_),ndigits=4)
      else:
        row1[metric_name.replace('get_','high_score_')]=round(number=metric_func(sent_1=doc1_,sent_2=doc2_),ndigits=4)
    return row1

  def main(self)->None:
    """
    Main function to perform text scoring and write results to a CSV file.
    """
    start_time:float = time.time()

    self.create_final_folder()

    # single core processing
    # self.df1:pd.DataFrame = self.df1.apply(lambda x: self.get_all_similarity_scores(row1=x),axis=1)

    # using pandarallel for multiprocessing
    self.df1:pd.DataFrame = self.df1.parallel_apply(lambda x: self.get_all_similarity_scores(row1=x),axis=1)

    self.df1.to_csv(path_or_buf=self.output_folder+'/Similarity_Scores.csv',index=False,mode='w',encoding='utf-8') # save in CSV file format
    # self.create_final_zip()
    print(f"Elapsed time: {((time.time() - start_time) / 60):.2f} minutes")
    return None

def custom_ram_cleanup_func()->None:
  """
  Clean up global variables except for specific exclusions and system modules.

  This function deletes all global variables except those specified in
  `exclude_vars` and variables starting with underscore ('_').

  Excluded variables:
  - Modules imported into the system (except 'sys' and 'os')
  - 'sys', 'os', and 'custom_ram_cleanup_func' itself

  Returns:
  None
  """

  import sys
  all_vars = list(globals().keys())
  exclude_vars = list(sys.modules.keys())
  exclude_vars.extend(['In','Out','_','__','___','__builtin__','__builtins__','__doc__','__loader__','__name__','__package__','__spec__','_dh','_i','_i1','_ih','_ii','_iii','_oh','exit','get_ipython','quit','sys','os','custom_ram_cleanup_func',])
  for var in all_vars:
      if var not in exclude_vars and not var.startswith('_'):
          del globals()[var]
  del sys
  return None

# Example usage:
if __name__ == "__main__":

  import pandas as pd

  # Sample DataFrame
  df = pd.DataFrame(data={
      'doc1_elements': ['apple', 'banana', 'cherry'],
      'doc2_elements': ['apples', 'bannnana', 'charries'],
  })

  # Create TextScoring instance and compute similarity scores
  TextScoring(
      dataframe_object=df,
      output_folder='Example2',
      col_name_1='doc1_elements',
      col_name_2='doc2_elements',
      metrics_list=['fuzz_token_set_ratio', 'jaro_winkler_similarity']
  ).main()

  custom_ram_cleanup_func()
  del custom_ram_cleanup_func

