from flask import Flask, request, jsonify, render_template
import openai
import re
import os
import requests
import tiktoken
from dotenv import load_dotenv
load_dotenv()
from markupsafe import Markup

app = Flask(__name__)

# Load API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
SUPADATA_API_KEY = os.getenv('SUPADATA_API_KEY')

# 1. Extract YouTube Video ID
def extract_video_id(url):
    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu\.be\/|\/v\/|\/e\/|watch\?v=|watch\?.+&v=)([\w-]{11})",
        r"youtu\.be\/([\w-]{11})",
        r"youtube\.com\/embed\/([\w-]{11})",
        r"youtube\.com\/v\/([\w-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# 2. Fetch transcript from SupaData
def get_supadata_transcript(video_id):
    url = f'https://api.supadata.ai/v1/youtube/transcript?videoId={video_id}'
    headers = {'x-api-key': SUPADATA_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f'SupaData API error: {response.text}')
    data = response.json()
    if 'content' not in data:
        raise Exception('Transcript not found in SupaData response')
    return ' '.join([entry['text'] for entry in data['content']])

# 3. Enhanced summary prompt
PROMPT_TEMPLATE_PATH = 'summary_prompt.md'

def generate_summary_prompt(transcript_text):
    with open(PROMPT_TEMPLATE_PATH, 'r') as f:
        template = f.read()
    return template.format(transcript_text=transcript_text)

# 4. Call OpenAI to summarize
def estimate_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def summarize_with_openai(transcript_text):
    model = "gpt-4o"
    input_tokens = estimate_tokens(transcript_text, model=model)

    # Output summary length capped to 10,000 or mirrors input
    max_output_tokens = min(10000, input_tokens)

    prompt = generate_summary_prompt(transcript_text)

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You summarize long YouTube transcripts professionally."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_output_tokens,
        temperature=0.01,
    )
    return response.choices[0].message.content.strip()

def summary_to_html(summary_text):
    # Convert markdown-style section titles to <h2>
    html = summary_text
    # Section titles: lines that are all-caps or start with a capital and are followed by a line break
    html = re.sub(r'^([A-Z][A-Za-z0-9 ,\-]+):?$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    # Bold phrases: **text** or *text* to <strong>text</strong>
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.*?)\*', r'<strong>\1</strong>', html)
    # Numbered lists: lines starting with 1. 2. etc.
    html = re.sub(r'^(\d+)\.\s+(.*)$', r'<li>\2</li>', html, flags=re.MULTILINE)
    # Bulleted lists: lines starting with - or *
    html = re.sub(r'^[\-\*]\s+(.*)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    # Wrap consecutive <li> in <ul> or <ol>
    html = re.sub(r'((<li>.*?</li>\s*)+)', lambda m: '<ul>' + m.group(1) + '</ul>' if '<li>' in m.group(1) else m.group(1), html, flags=re.DOTALL)
    # Section titles: add spacing
    html = re.sub(r'<h2>(.*?)</h2>', r'<h2 style="font-size:22px;font-weight:700;margin:2em 0 0.7em 0;">\1</h2>', html)
    # Wrap in summary-output div
    html = f'<div class="summary-output">{html}</div>'
    return Markup(html)

@app.route('/')
def index():
    return render_template('index.html')  # Simple input form (optional)

@app.route('/api/summarize', methods=['POST'])
def summarize():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL'}), 400
    try:
        transcript = get_supadata_transcript(video_id)
    except Exception as e:
        return jsonify({'error': f'Could not fetch transcript: {str(e)}'}), 500
    try:
        summary = summarize_with_openai(transcript)
        summary_html = summary_to_html(summary)
    except Exception as e:
        return jsonify({'error': f'Could not summarize transcript: {str(e)}'}), 500
    return jsonify({'summary': str(summary_html)})

if __name__ == '__main__':
    app.run(debug=True)
