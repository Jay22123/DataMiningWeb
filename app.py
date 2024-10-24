import numpy as np
from matplotlib.ticker import MaxNLocator
import random
import re
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, jsonify, make_response, send_from_directory
import os
from Read_Engine import Reader
from DataProcess import Processor
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端，避免啟動 GUI

app = Flask(__name__)
reader = Reader()
processor = Processor()

# 設定上傳檔案的目錄
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 列出上傳的檔案並讓用戶選擇
@app.route('/', methods=['GET', 'POST'])
def list_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    
    if request.method == 'POST':
        # 取得用戶選擇的檔案名稱
        selected_file = request.form.get('selected_file')
        if selected_file:
            # 例如：處理下載檔案的動作
            return redirect(url_for('uploaded_file', filename=selected_file))

    # 初次載入時列出檔案，顯示選擇表單
    return render_template('index.html', files=files)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 首頁，顯示兩個上傳檔案的按鈕


@app.route('/')
def index():
    return render_template('index.html')

# 處理左側檔案上傳


@app.route('/upload_left', methods=['POST'])
def upload_file_left():
    selected_file = request.form.get('selected_left_file')

    if not selected_file:
        return jsonify({'error': 'No file selected'}), 400

    # 檢查選擇的檔案是否為 XML 檔案
    if not selected_file.endswith('.xml'):
        return jsonify({'error': 'Please select an XML file.'}), 400

    # 檢查檔案是否存在於 uploads 資料夾中
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
    if not os.path.exists(file_path):
        return jsonify({'error': f'File {selected_file} does not exist.'}), 404

    # 從前端獲取 use_porter 值，並轉換為布爾值
    use_porter = request.form.get('use_porter') == 'true'
    search_word = request.form.get('search_word', '')


    message = reader.ReadDocument(file_path)

    # 如果提供了 search_word，對文本進行標註
    analysis_text = message["Content"]
    if search_word:
        # 使用正則表達式找出所有匹配 search_word 的單詞，並替換為帶有標籤的形式
        highlighted_text = re.sub(f'({search_word})', r'<span style="background-color: yellow;">\1</span>', analysis_text, flags=re.IGNORECASE)
    else:
        highlighted_text = analysis_text

    results = processor.analyze_text(message)
    tokens, frequencies = processor.zipf(message, isPorter=use_porter)

    top_20_tokens = tokens[:20]
    top_20_frequencies = frequencies[:20]
    plt.figure(figsize=(10, 6))
    plt.plot(top_20_frequencies, marker='o')
    plt.xticks(range(len(top_20_tokens)), top_20_tokens, rotation=45)
    plt.title('Top 20 Tokens by Frequency')
    plt.xlabel('Tokens')
    plt.ylabel('Frequency')

    # 在每個點上標註 frequencies 的值
    for i, freq in enumerate(top_20_frequencies):
        plt.text(i, freq, str(freq), ha='center', va='bottom', fontsize=9)

    plt.grid(True)

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # 保存圖像到uploads目錄
    image_token_path = os.path.join(
        app.config['UPLOAD_FOLDER'], 'left_token_plot.png')

    # 自動調整佈局，避免標籤被截斷
    plt.tight_layout()

    plt.savefig(image_token_path)
    plt.close()

    # 計算排名（1, 2, 3,... 對應 tokens）
    ranks = np.arange(1, len(tokens) + 1)

    # 對排名和頻率取對數
    log_ranks = np.log(ranks)
    log_frequencies = np.log(frequencies)

    # 繪製對數-對數圖
    plt.figure(figsize=(10, 6))
    plt.plot(log_ranks, log_frequencies, marker='o')
    plt.title('Zipf Distribution of Tokens')
    plt.xlabel('Log Rank')
    plt.ylabel('Log Frequency')
    plt.grid(True)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # 保存圖像到uploads目錄
    image_zipf_path = os.path.join(
        app.config['UPLOAD_FOLDER'], 'left_zipf_plot.png')

    # 自動調整佈局，避免標籤被截斷
    plt.tight_layout()
    plt.savefig(image_zipf_path)
    plt.close()

    # 添加緩存破壞符
    cache_buster = random.randint(1, 10000)  # 生成隨機數避免緩存

    return jsonify(
        {'message': 'File uploaded successfully',
         'analysis': highlighted_text,
         'char_count_including_spaces':  results["char_count_including_spaces"],
         'char_count_excluding_spaces':  results["char_count_excluding_spaces"],
         'word_count':  results["word_count"],
         'sentence_count':  results["sentence_count"],
         'non_ascii_char_count':  results["non_ascii_char_count"],
         'non_ascii_word_count':  results["non_ascii_word_count"],
         'token_plot_url': f'/{image_token_path}?cb={cache_buster}',
         'zipf_plot_url': f'/{image_zipf_path}?cb={cache_buster}'})


# 處理右側檔案上傳
@app.route('/upload_right', methods=['POST'])
def upload_file_right():
    if 'file_right' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file_right']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.endswith('.xml'):
        return jsonify({'error': 'Please upload an XML file.'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # 從前端獲取 use_porter 值，並轉換為布爾值
    use_porter = request.form.get('use_porter') == 'true'
    search_word = request.form.get('search_word', '')

    message = reader.ReadDocument(file_path)

    # 如果提供了 search_word，對文本進行標註
    analysis_text = message["Content"]
    if search_word:
        # 使用正則表達式找出所有匹配 search_word 的單詞，並替換為帶有標籤的形式
        highlighted_text = re.sub(f'({search_word})', r'<span style="background-color: yellow;">\1</span>', analysis_text, flags=re.IGNORECASE)
    else:
        highlighted_text = analysis_text


    results = processor.analyze_text(message)
    tokens, frequencies = processor.zipf(message, isPorter=use_porter)

    # 繪製 token 圖像
    top_20_tokens = tokens[:20]
    top_20_frequencies = frequencies[:20]
    plt.figure(figsize=(10, 6))
    plt.plot(top_20_frequencies, marker='o')
    plt.xticks(range(len(top_20_tokens)), top_20_tokens, rotation=45)
    plt.title('Top 20 Tokens by Frequency')
    plt.xlabel('Tokens')
    plt.ylabel('Frequency')

    # 在每個點上標註 frequencies 的值
    for i, freq in enumerate(top_20_frequencies):
        plt.text(i, freq, str(freq), ha='center', va='bottom', fontsize=9)

    plt.grid(True)

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # 保存圖像到uploads目錄
    image_token_path = os.path.join(
        app.config['UPLOAD_FOLDER'], 'right_token_plot.png')

    # 自動調整佈局，避免標籤被截斷
    plt.tight_layout()

    plt.savefig(image_token_path)

    plt.close()

    # 計算排名（1, 2, 3,... 對應 tokens）
    ranks = np.arange(1, len(tokens) + 1)

    # 對排名和頻率取對數
    log_ranks = np.log(ranks)
    log_frequencies = np.log(frequencies)

    # 繪製對數-對數圖
    plt.figure(figsize=(10, 6))
    plt.plot(log_ranks, log_frequencies, marker='o')
    plt.title('Zipf Distribution of Tokens')
    plt.xlabel('Log Rank')
    plt.ylabel('Log Frequency')
    plt.grid(True)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # 保存圖像到uploads目錄
    image_zipf_path = os.path.join(
        app.config['UPLOAD_FOLDER'], 'right_zipf_plot.png')

    # 自動調整佈局，避免標籤被截斷
    plt.tight_layout()
    plt.savefig(image_zipf_path)
    plt.close()

    # 添加緩存破壞符
    cache_buster = random.randint(1, 10000)  # 生成隨機數避免緩存

    return jsonify(
        {'message': 'File uploaded successfully',
         'analysis':  highlighted_text,
         'char_count_including_spaces':  results["char_count_including_spaces"],
         'char_count_excluding_spaces':  results["char_count_excluding_spaces"],
         'word_count':  results["word_count"],
         'sentence_count':  results["sentence_count"],
         'non_ascii_char_count':  results["non_ascii_char_count"],
         'non_ascii_word_count':  results["non_ascii_word_count"],
         'token_plot_url': f'/{image_token_path}?cb={cache_buster}',
         'zipf_plot_url': f'/{image_zipf_path}?cb={cache_buster}'})


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=5000)
