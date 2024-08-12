import time
import csv
import os
from flask import Flask, request
import pandas as pd
import numpy as np

from utils import write, diff, get_row, get_total
from db import init_db, set_user_counter, get_user_counter

init_db()
app = Flask(__name__)

@app.route('/time')
def get_current_time():
    return {'time': time.time()}

@app.route('/controlledtest/', defaults={'user': None})
@app.route('/controlledtest/<user>', methods=['GET', 'POST'])
def get_controlledtest(user):
    if request.method == 'POST':
        success = write(request.json)
        if success:
            set_user_counter(user)
            return "Success", 201
        return "Bad Request", 400
    else:
        if(user is None):
            return {
                'row': 0,
                'total': get_total(),
                'flip': None,
                'prompt': 'Please set your name and press enter to continue.',
                'promptID': 0,
                'mutator': '',
                'watermark': '',
                'response_A': '',
                'response_B': '',
                'align_A': '',
                'align_B': '',
                'mutation_A': 0,
                'mutation_B': 0,
            }
        counter = get_user_counter(user)
        if(counter >= get_total()):
            return {
                'row': counter,
                'total': get_total(),
                'flip': None,
                'prompt': 'Nothing left to annotate',
                'promptID': 0,
                'mutator': '',
                'watermark': '',
                'response_A': '',
                'response_B': '',
                'align_A': '',
                'align_B': '',
                'mutation_A': 0,
                'mutation_B': 0,
            }
        data = get_row(counter)
        A,B = data['original'], data['mutated']
        mutation_A, mutation_B = 0, data['step']
        # flip a coin to decide which mutation to show first
        flip = request.args.get('flip')
        if flip == "false":
            pass
        elif flip == "true" or np.random.rand() > 0.5:
            A,B = B,A
            mutation_A, mutation_B = mutation_B, mutation_A
            flip = "true"
        else:
            flip = "false"

        response_A, response_B = diff(A,B)
        align_A, align_B = diff(A, B, align=True)
        return {
            'row': counter,
            'total': get_total(),
            'flip': flip=="true",
            'prompt': data['prompt'],
            'promptID': data['id'],
            'mutator': data['mutator'],
            'watermark': data['watermark'],
            'response_A': response_A,
            'response_B': response_B,
            'align_A': align_A,
            'align_B': align_B,
            'mutation_A': mutation_A,
            'mutation_B': mutation_B,
        }

# @app.route('/blindtest', methods=['GET', 'POST'])
# def get_blindtest():
#     if request.method == 'POST':
#         success = write(request.json)
#         if success:
#             return "Success", 201
#         return "Bad Request", 400

#     else:
#         random_group = get_random_group()
#         mutations = random_group['mutated_text'].reset_index(drop=True).tolist()
#         choice = np.random.choice(len(mutations), 2, replace=False).tolist()
#         A = mutations[choice[0]]
#         B = mutations[choice[1]]
#         response_A, response_B = diff(A,B)
#         align_A, align_B = diff(A, B, align=True)
#         return {
#             'prompt': random_group['prompt'].iloc[0],
#             'mutator': random_group['mutator'].iloc[0],
#             'response': random_group['response'].iloc[0],
#             'response_A': response_A,
#             'response_B': response_B,
#             'align_A': align_A,
#             'align_B': align_B,
#             'mutation_A': choice[0],
#             'mutation_B': choice[1],
#         }