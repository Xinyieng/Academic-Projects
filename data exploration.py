# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:26:33 2019
Test to convert list to 

@author: alyam
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_num_words_per_sample(sample_texts):
    """Returns the median number of words per sample given corpus.

    # Arguments
        sample_texts: list, sample texts.

    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)

def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s.split()) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()

def count_relevant(dataframe):
    count = 0
    if(dataframe.loc[:,'target'] == 1):
        print("in if")
        count+=1
    return count

growth = pd.read_csv("C:\\Users\\alyam\\PycharmProjects\\Principal Final Deliverables\\Data\\growth-data-all.csv")
stability = pd.read_csv("C:\\Users\\alyam\\PycharmProjects\\Principal Final Deliverables\\Data\\stability-data-all.csv")
opportunity = pd.read_csv("C:\\Users\\alyam\\PycharmProjects\\Principal Final Deliverables\\Data\\opportunity-data-all.csv")
strategy = pd.read_csv("C:\\Users\\alyam\\PycharmProjects\\Principal Final Deliverables\\Data\\strategy-data-all.csv")

print("Growth Data Exploration")
print("Total number of observation: ", len(growth.index))
print("Samples division: ", growth.groupby('target')['target'].count())
print("Median number of words: ", get_num_words_per_sample(growth['Sentences'].tolist()))
plot_sample_length_distribution(growth['Sentences'].tolist())

print("Stability Data Exploration")
print("Total number of observation: ", len(stability.index))
print("Samples division: ", stability.groupby('target')['target'].count())
print("Median number of words: ", get_num_words_per_sample(stability['Sentences'].tolist()))
plot_sample_length_distribution(stability['Sentences'].tolist())

print("Opportunity Data Exploration")
print("Total number of observation: ", len(opportunity.index))
print("Samples division: ", opportunity.groupby('target')['target'].count())
print("Median number of words: ", get_num_words_per_sample(opportunity['Sentences'].tolist()))
plot_sample_length_distribution(opportunity['Sentences'].tolist())

print("Strategy Data Exploration")
print("Total number of observation: ", len(strategy.index))
print("Samples division: ", strategy.groupby('target')['target'].count())
print("Median number of words: ", get_num_words_per_sample(strategy['Sentences'].tolist()))
plot_sample_length_distribution(strategy['Sentences'].tolist())