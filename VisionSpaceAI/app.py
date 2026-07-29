from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge
import os, uuid, zipfile, io, traceback, logging
from auth import auth_bp, init_db
#python 3.14.3

from video_processor import extract_frames, select_frames
from image_preprocessor import preprocess_frames
from object_detector import detect_objects
from feature_extractor import extract_features
from preference_encoder import encode_preferences
from feature_fusion import fuse_features
from prompt_generator import generate_prompt
from design_generator import generate_designs

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder='.')
app.secret_key = os.environ.get('SECRET_KEY', 'decora-ai-secret-2024')
app.register_blueprint(auth_bp)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

os.makedirs('uploads', exist_ok=True)
os.makedirs('static/outputs', exist_ok=True)

ALLOWED = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v', '.flv', '.wmv'}


def cors(r):
    r.headers['Access-Control-Allow-Origin'] = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return r


@app.after_request
def add_cors(resp):
    return cors(resp)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify({'error': 'File too large — max 500 MB'}), 413


@app.errorhandler(413)
def too_large2(_):
    return jsonify({'error': 'File too large — max 500 MB'}), 413


@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/app')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_video():
    if request.method == 'OPTIONS':
        return cors(jsonify({}))

    log.info(f"Upload request — content-length: {request.content_length}")

    if 'video' not in request.files:
        log.error("No 'video' field in files")
        return jsonify({'error': "No 'video' field found. Check the form field name."}), 400

    file = request.files['video']
    if not file or not file.filename:
        return jsonify({'error': 'Empty file selection'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify({'error': f'Format "{ext}" not supported. Use: mp4, mov, avi, webm'}), 400

    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
    except Exception as e:
        return jsonify({'error': f'Save failed: {e}'}), 500

    size = os.path.getsize(filepath)
    if size == 0:
        os.remove(filepath)
        return jsonify({'error': 'File saved as 0 bytes — may be corrupt'}), 400

    log.info(f"Saved {filename} ({size/1024/1024:.1f} MB)")
    return jsonify({'video_id': filename, 'size_mb': round(size / 1024 / 1024, 1)})


@app.route('/process', methods=['POST', 'OPTIONS'])
def process():
    if request.method == 'OPTIONS':
        return cors(jsonify({}))

    data = request.form
    video_id = data.get('video_id', '').strip()
    if not video_id:
        return jsonify({'error': 'video_id missing'}), 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_id)
    if not os.path.exists(video_path):
        return jsonify({'error': f'Video not found on server. Try re-uploading.'}), 404

    preferences = {
        'style':         data.get('style',         'modern'),
        'room_type':     data.get('room_type',      'living_room'),
        'color_theme':   data.get('color_theme',    'neutral'),
        'wall_color':    data.get('wall_color',     'white'),
        'curtain_style': data.get('curtain_style',  'none'),
        'wall_decor':    data.get('wall_decor',     'none'),
        'lighting_mood': data.get('lighting_mood',  'warm_ambient'),
        'flooring':      data.get('flooring',       'keep_existing'),
        'add_plants':    data.get('add_plants',     'none'),
        'ceiling_style': data.get('ceiling_style',  'standard_white'),
        'budget':        float(data.get('budget',   5000)),

    }

    session_id = str(uuid.uuid4())
    out_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    os.makedirs(out_dir, exist_ok=True)

    try:
        import cv2 as _cv2

        frames = extract_frames(video_path)
        if not frames:
            return jsonify({'error': 'No frames extracted — video may be corrupt or codec unsupported'}), 422

        wall_frames  = select_frames(frames)
        wall_count   = len(wall_frames)
        preprocessed = preprocess_frames(wall_frames)
        detections   = detect_objects(preprocessed)
        img_features = extract_features(preprocessed)
        enc_prefs    = encode_preferences(preferences)
        fused        = fuse_features(img_features, detections, enc_prefs)
        prompt, neg  = generate_prompt(fused, preferences, detections)
        outputs      = generate_designs(preprocessed, prompt, neg, out_dir)

        source_urls = []
        for i, frame in enumerate(wall_frames):
            p = os.path.join(out_dir, f'before_{i}.jpg')
            _cv2.imwrite(p, frame)
            source_urls.append(url_for('static', filename=f'outputs/{session_id}/before_{i}.jpg'))

        image_urls = [
            url_for('static', filename=f'outputs/{session_id}/{os.path.basename(p)}')
            for p in outputs
        ]

        return jsonify({
            'prompt': prompt, 'before_images': source_urls,
            'after_images': image_urls, 'wall_count': wall_count,
            'session_id': session_id,
            'detected_objects': [d['label'] for d in detections[:10]]
        })

    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/designs/get/<session_id>', methods=['GET'])
def get_session(session_id):
    out_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    if not os.path.exists(out_dir):
        return jsonify({'error': 'Design session files not found'}), 404
    
    # Fetch prompt from database
    from auth import get_db
    db = get_db()
    row = db.execute('SELECT prompt, wall_count FROM designs WHERE session_id=?', (session_id,)).fetchone()
    db.close()
    
    prompt = row['prompt'] if row else "AI Space Redesign"
    wall_count = row['wall_count'] if row else 0
    
    # Reconstruct images list
    files = os.listdir(out_dir)
    afters = sorted([f for f in files if f.startswith('design_')])
    befores = sorted([f for f in files if f.startswith('before_')])
    
    # Build response in same format as /process
    return jsonify({
        'session_id': session_id,
        'before_images': [url_for('static', filename=f'outputs/{session_id}/{f}') for f in befores],
        'after_images': [url_for('static', filename=f'outputs/{session_id}/{f}') for f in afters],
        'prompt': prompt,
        'wall_count': wall_count or len(afters)
    })


@app.route('/download/<session_id>')
def download_zip(session_id):
    out_dir = os.path.join(app.config['OUTPUT_FOLDER'], session_id)
    if not os.path.exists(out_dir):
        return jsonify({'error': 'Session not found'}), 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for f in os.listdir(out_dir):
            zf.write(os.path.join(out_dir, f), f)
    buf.seek(0)
    return send_file(buf, mimetype='application/zip',
                     as_attachment=True, download_name='interior_designs.zip')


if __name__ == '__main__':
    init_db()
    import socket
    def free_port(candidates):
        for p in candidates:
            s = socket.socket()
            try: s.bind(('', p)); s.close(); return p
            except: s.close()
        return 5000
    port = free_port([5001, 5002, 8080, 8000, 5000])
    print(f"\n  Open -> http://localhost:{port}\n")
    app.run(debug=True, host='0.0.0.0', port=port)