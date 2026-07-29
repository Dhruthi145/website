"""
auth.py — User authentication with SQLite
Handles signup, login, logout, session management
"""
import sqlite3, hashlib, os, uuid
from flask import Blueprint, request, jsonify, session, g

auth_bp = Blueprint('auth', __name__)
DB_PATH = 'decora_users.db'

def get_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            avatar_color TEXT DEFAULT '#a855f7'
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS designs (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            prompt TEXT,
            wall_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    db.commit()
    db.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def get_current_user():
    token = request.cookies.get('auth_token') or request.headers.get('X-Auth-Token')
    if not token:
        return None
    db = get_db()
    row = db.execute(
        'SELECT u.* FROM users u JOIN sessions s ON u.id=s.user_id WHERE s.id=?',
        (token,)
    ).fetchone()
    db.close()
    return dict(row) if row else None

@auth_bp.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name  = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip().lower()
    pwd   = data.get('password') or ''

    if not name or not email or not pwd:
        return jsonify({'error': 'All fields required'}), 400
    if len(pwd) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    colors = ['#a855f7','#6366f1','#ec4899','#f59e0b','#10b981','#3b82f6']
    color  = colors[len(email) % len(colors)]
    uid    = str(uuid.uuid4())

    db = get_db()
    try:
        db.execute(
            'INSERT INTO users (id,name,email,password_hash,avatar_color) VALUES (?,?,?,?,?)',
            (uid, name, email, hash_password(pwd), color)
        )
        token = str(uuid.uuid4())
        db.execute('INSERT INTO sessions (id,user_id) VALUES (?,?)', (token, uid))
        db.commit()
    except sqlite3.IntegrityError:
        db.close()
        return jsonify({'error': 'Email already registered'}), 409
    db.close()

    resp = jsonify({'user': {'id':uid,'name':name,'email':email,'avatar_color':color}})
    resp.set_cookie('auth_token', token, httponly=True, max_age=30*24*3600)
    resp.headers['X-Auth-Token'] = token
    return resp

@auth_bp.route('/api/auth/login', methods=['POST'])
def login():
    data  = request.get_json()
    email = (data.get('email') or '').strip().lower()
    pwd   = data.get('password') or ''

    db  = get_db()
    row = db.execute(
        'SELECT * FROM users WHERE email=? AND password_hash=?',
        (email, hash_password(pwd))
    ).fetchone()

    if not row:
        db.close()
        return jsonify({'error': 'Invalid email or password'}), 401

    user  = dict(row)
    token = str(uuid.uuid4())
    db.execute('INSERT INTO sessions (id,user_id) VALUES (?,?)', (token, user['id']))
    db.commit()
    db.close()

    resp = jsonify({'user': {
        'id': user['id'], 'name': user['name'],
        'email': user['email'], 'avatar_color': user['avatar_color']
    }})
    resp.set_cookie('auth_token', token, httponly=True, max_age=30*24*3600)
    resp.headers['X-Auth-Token'] = token
    return resp

@auth_bp.route('/api/auth/logout', methods=['POST'])
def logout():
    token = request.cookies.get('auth_token')
    if token:
        db = get_db()
        db.execute('DELETE FROM sessions WHERE id=?', (token,))
        db.commit()
        db.close()
    resp = jsonify({'ok': True})
    resp.delete_cookie('auth_token')
    return resp

@auth_bp.route('/api/auth/me', methods=['GET'])
def me():
    user = get_current_user()
    if not user:
        return jsonify({'user': None}), 401
    return jsonify({'user': {
        'id': user['id'], 'name': user['name'],
        'email': user['email'], 'avatar_color': user['avatar_color']
    }})

@auth_bp.route('/api/designs/save', methods=['POST'])
def save_design():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Not logged in'}), 401
    data = request.get_json()
    db   = get_db()
    db.execute(
        'INSERT INTO designs (id,user_id,session_id,prompt,wall_count) VALUES (?,?,?,?,?)',
        (str(uuid.uuid4()), user['id'], data.get('session_id',''),
         data.get('prompt',''), data.get('wall_count',0))
    )
    db.commit()
    db.close()
    return jsonify({'ok': True})

@auth_bp.route('/api/designs/history', methods=['GET'])
def history():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Not logged in'}), 401
    db   = get_db()
    rows = db.execute(
        'SELECT * FROM designs WHERE user_id=? ORDER BY created_at DESC LIMIT 20',
        (user['id'],)
    ).fetchall()
    db.close()
    return jsonify({'designs': [dict(r) for r in rows]})