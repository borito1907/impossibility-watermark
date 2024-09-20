import time
import csv
import os
import pandas as pd
import numpy as np
from flask import Flask, request
from flask_socketio import SocketIO
from socket_handlers import setup_event_handlers
from utils import write, diff, get_row, get_total
from db import init_db, set_user_counter, get_user_counter

init_db()
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
setup_event_handlers(socketio)

@app.route('/time')
def get_current_time():
    return {'time': time.time()}

@app.route('/diff')
def get_diff():
    A = request.args.get('raw_A')
    B = request.args.get('raw_B')
    response_A, response_B = diff(A,B)
    align_A, align_B = diff(A, B, align=True)
    return {
        'response_A': response_A,
        'response_B': response_B,
        'align_A': align_A,
        'align_B': align_B,
    }

@app.route('/getrow/<int:row>')
def inspect_row(row):
    if(row >= get_total()):
        return "Bad Request", 400
    data = get_row(row)
    A,B = data['original'], data['mutated']
    mutation_A, mutation_B = 0, data['step']
    response_A, response_B = diff(A,B)
    align_A, align_B = diff(A, B, align=True)
    return {
        'row': row,
        'total': get_total(),
        'flip': False,
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
            A,B = "The quick brown fox jumps over the lazy dog.", "The really, really quick fox jumps over the crazy dog."
            # flip a coin to decide which mutation to show first
            flip = request.args.get('flip')
            if flip == "false":
                pass
            elif flip == "true" or np.random.rand() > 0.5:
                A,B = B,A
                flip = "true"
            else:
                flip = "false"

            response_A, response_B = diff(A,B)
            align_A, align_B = diff(A, B, align=True)
            return {
                'row': get_total(),
                'total': get_total(),
                'flip': flip=="true",
                'prompt': 'Nothing left to annotate',
                'promptID': 0,
                'mutator': '',
                'watermark': '',
                'response_A': response_A,
                'response_B': response_B,
                'align_A': align_A,
                'align_B': align_B,
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