import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize  # 引入 NLTK 的分詞模組
from nltk.corpus import stopwords  # 引入 NLTK 的停用詞庫模組
from nltk.text import Text
from nltk.stem import PorterStemmer

from collections import Counter
import numpy as np
import os
import re


class Processor():
    def __init__(self) -> None:
        self._data = []
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('punkt')

    # 字串處理

    def analyze_text(self, text):

        text = str(text)
        words = word_tokenize(text.lower())
        cleaned_words = [re.sub(r'[^\w\s]', '', word)
                         for word in words if re.sub(r'[^\w\s]', '', word)]

        # 2. char數(含空格)
        char_count_including_spaces = len(text)

        # 3. char數(不含空格)
        char_count_excluding_spaces = len(text.replace(" ", ""))

        # 4. word數量
        word_count = len(cleaned_words)

        # 5. 句子數量
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)

        # 6. 非 ASCII  char
        non_ascii_chars = [char for char in text if ord(char) > 127]
        non_ascii_char_count = len(non_ascii_chars)

        # 7. 非 ASCII word
        non_ascii_words = [word for word in words if any(
            ord(char) > 127 for char in word)]
        non_ascii_word_count = len(non_ascii_words)

        return {
            "char_count_including_spaces": char_count_including_spaces,
            "char_count_excluding_spaces": char_count_excluding_spaces,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "non_ascii_char_count": non_ascii_char_count,
            "non_ascii_word_count": non_ascii_word_count
        }

    def zipf(self, text, isPorter=True):

        text = str(text)
        words = word_tokenize(text.lower())
        cleaned_words = [re.sub(r'[^\w\s]', '', word)
                         for word in words if re.sub(r'[^\w\s]', '', word)]

        sr = stopwords.words('english')

        filtered_tokens = [token for token in cleaned_words if token not in sr]

        if isPorter:
            tokens = self.porterAlg(filtered_tokens)
        else:
            tokens = filtered_tokens

        # 計算 token 的頻率分佈
        freq_dist = Counter(tokens)

        # 根據頻率對 token 進行排序
        sorted_token_freq = sorted(
            freq_dist.items(), key=lambda x: x[1], reverse=True)

        # 提取前 20 個 token 和對應的頻率
        tokens, frequencies = zip(*sorted_token_freq)

        # 返回 tokens 和對應的頻率
        return tokens, frequencies

    def porterAlg(self, cleaned_words):

        print("In PorterAlg")

        stemmer = PorterStemmer()

        stemmed_tokens = [stemmer.stem(token) for token in cleaned_words]

        return stemmed_tokens
