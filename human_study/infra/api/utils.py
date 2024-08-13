import pandas as pd
import numpy as np
import os
import csv
import time
from difflib import SequenceMatcher
import regex as re
from itertools import chain, zip_longest

INPUT_FILE = '../data/wqe_experiment.csv'
df = pd.read_csv(INPUT_FILE, encoding='utf-8')
def get_row(counter):
    return df.iloc[counter].to_dict()

ANNOTATIONS_FILE = '../data/annotations.csv'
headers = ['time', 'row', 'user', 'id', 'mutator', 'watermark', 'mutation_A', 'mutation_B', 'choice']
if not os.path.exists(ANNOTATIONS_FILE):
    with open(ANNOTATIONS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    

def get_total():
    return len(df)
def write(data):
    data['time'] = time.time()
    if not set(headers).issubset(data.keys()):
        return False
    with open(ANNOTATIONS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [data[key] for key in headers]
        writer.writerow(row)
        return True

def intersperse_lists(list1, list2):
    return ''.join(chain(*zip_longest(list1, list2)))

def diff(response1, response2, align=False):
    text1 = re.split(r'(\S+)', response1)
    line1 = text1[1::2]
    whitespace1 = text1[::2]
    text2 = re.split(r'(\S+)', response2)
    line2 = text2[1::2]
    whitespace2 = text2[::2]
    if align:
        whitespace1 = [' ']*len(whitespace1)
        whitespace2 = [' ']*len(whitespace2)

    lightgreen = '#9effb9'
    lightred = '#ff9ead'
    lightpurple = '#d5b1f2'
    essay1_html = []
    essay2_html = []
    def color(s, color):
        return f'<span style="background-color: {color};">{s}</span>'
    matcher = SequenceMatcher(None, line1, line2)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            essay1_html.append(intersperse_lists(whitespace1[i1:i2], line1[i1:i2]))
            essay2_html.append(intersperse_lists(whitespace1[i1:i2], line1[i1:i2]))
        elif tag == 'delete':
            essay1_html.append(whitespace1[i1])
            whitespace1[i1] = ''
            text = intersperse_lists(whitespace1[i1:i2], line1[i1:i2])
            essay1_html.append(color(text, lightred))
            if align:
                essay2_html.append(' '*(len(text)+1))
        elif tag == 'insert':
            essay2_html.append(whitespace2[j1])
            whitespace2[j1] = ''
            text = intersperse_lists(whitespace2[j1:j2], line2[j1:j2])
            essay2_html.append(color(text, lightgreen))
            if align:
                essay1_html.append(' '*(len(text)+1))
        elif tag == 'replace':
            essay1_html.append(whitespace1[i1])
            whitespace1[i1] = ''
            essay2_html.append(whitespace2[j1])
            whitespace2[j1] = ''
            text1 = intersperse_lists(whitespace1[i1:i2], line1[i1:i2])
            text2 = intersperse_lists(whitespace2[j1:j2], line2[j1:j2])
            if align:
                max_length = max(len(text1), len(text2))
                text1 = text1.ljust(max_length)
                text2 = text2.ljust(max_length)
            essay1_html.append(color(text1, lightpurple))
            essay2_html.append(color(text2, lightpurple))
    essay1_html.append(whitespace1[-1])
    essay2_html.append(whitespace2[-1])
    return ''.join(essay1_html), ''.join(essay2_html)