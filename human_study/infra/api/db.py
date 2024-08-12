import sqlite3
import threading

lock = threading.Lock()

def init_db():
    conn = sqlite3.connect('counter.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS counter (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_counter (
        user_id TEXT PRIMARY KEY,
        counter INTEGER
    )
    ''')
    conn.commit()
    conn.close()

def set_user_counter(user_id):
    with lock:
        conn = sqlite3.connect('counter.db')
        cursor = conn.cursor()
        cursor.execute('BEGIN TRANSACTION')
        cursor.execute('INSERT INTO counter DEFAULT VALUES;')
        cursor.execute('SELECT id FROM counter ORDER BY id DESC LIMIT 1')
        counter = cursor.fetchone()[0] - 1
        cursor.execute('''
            INSERT INTO user_counter (user_id, counter) VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET counter = ?
        ''', (user_id,counter,counter,))
        conn.commit()
        conn.close()
    return counter

def get_user_counter(user_id):
    conn = sqlite3.connect('counter.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT counter FROM user_counter WHERE user_id = ?
    ''', (user_id,))
    user_counter = cursor.fetchone()
    conn.close()
    if user_counter is None:
        return set_user_counter(user_id)
    return user_counter[0]