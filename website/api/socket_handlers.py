from flask import request
from flask_socketio import emit
from utils import rooms, sids, get_random_string, diff  # Import the necessary utilities

def handle_connect():
    pass

def handle_create(data):
    roomid = data.get('roomid')
    if roomid is None:
        roomid = get_random_string(4)
        while roomid in rooms:
            roomid = get_random_string(4)
    sids[request.sid] = roomid
    rooms[roomid] = [request.sid]
    emit('set_room_id', roomid, to=request.sid)


def handle_join(data):
    roomid = data.get('roomid')
    if roomid in rooms:
        if sids[request.sid] in rooms and request.sid in rooms[sids[request.sid]]:
            rooms[sids[request.sid]].remove(request.sid)
        sids[request.sid] = roomid
        rooms[roomid].append(request.sid)
        emit('joined', roomid, to=request.sid)
        emit('new_user', request.sid, to=rooms[roomid][0])
    else:
        emit('error', f"{roomid} does not exist", to=request.sid)

def handle_disconnect():
    if sids[request.sid] in rooms:
        room = rooms[sids[request.sid]]
        if room[0] == request.sid:
            emit('disconnected', sids[request.sid], to=room[1:])
            del rooms[sids[request.sid]]
        else:
            room.remove(request.sid)
    del sids[request.sid]

def handle_create_task(data):
    room = data.get('roomid')
    A = data.get('original_text')
    B = data.get('mutated_text')
    response_A, response_B = diff(A, B)
    align_A, align_B = diff(A, B, align=True)

    data = {
        'prompt': data.get('instruction'),
        'taskid': data.get('taskid'),
        'response_A': response_A,
        'response_B': response_B,
        'align_A': align_A,
        'align_B': align_B,
    }
    emit('receive_task', data, to=rooms[room][1:])

def handle_resend_task(data):
    sid = data.get('sid')
    data = data.get('data')
    A = data.get('original_text')
    B = data.get('mutated_text')
    response_A, response_B = diff(A, B)
    align_A, align_B = diff(A, B, align=True)
    data = {
        'prompt': data.get('instruction'),
        'taskid': data.get('taskid'),
        'response_A': response_A,
        'response_B': response_B,
        'align_A': align_A,
        'align_B': align_B,
    }
    emit('receive_task', data, to=sid)

def handle_finish_task(data):
    room = data.get('roomid')
    emit('receive_answer', data, to=rooms[room][0])

def setup_event_handlers(socketio):
    socketio.on_event('connect', handle_connect)
    socketio.on_event('create', handle_create)
    socketio.on_event('join', handle_join)
    socketio.on_event('disconnect', handle_disconnect)
    socketio.on_event('create_task', handle_create_task)
    socketio.on_event('finish_task', handle_finish_task)
    socketio.on_event('resend_task', handle_resend_task)
