import os
import time
import codecs
from config import read_config
from tflite_classifier import nlog as log
from tflite_classifier.apply_bpe import BPE
from nltk.tokenize import word_tokenize

class TfliteClassifier:
    def __init__(self):
        log.info("Initializing Tflite classifier main object")
        start_time = time.time()
        self.init_config()
        log.info("Config initialized: time elapsed so far: %s seconds" % (time.time() - start_time))
        self.init_language()
        log.info("Language initialized: time elapsed so far: %s seconds" % (time.time() - start_time))

    def init_config(self):
        self.config = read_config(setup_file=None)

    def get_config(self, key, default=None):
        return self.config.get(str(key), default)

    def set_config(self, key, value):
        self.config[str(key)] = value

    def reload_config(self):
        self.init_config()

    def init_language(self):
        self.default_language = self.get_config('DEFAULT_LANGUAGE')
        self.available_languages = self.get_config('AVAILABLE_LANGUAGES')
        if self.default_language not in self.available_languages:
            log.warning('%s not in available languages: %s', self.default_language, self.available_languages)

    def _process_text(self, text):
        #from tflite_classifier.tokenizer import normalize_tokenize
        #return normalize_tokenize(text)
        text = text.lower()
        punct_list = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
        return [i.strip("".join(punct_list)) for i in word_tokenize(text) if i not in punct_list]

    def apply_bpe(self, text, classifier=None):
        if classifier == 'todo':
            print("getting todo codes file for prediction.")
            self.bpe = BPE(codecs.open('embeddings/todo_codes.txt', encoding='utf-8'))
        if classifier == 'current_todo':
            print("getting curent todo codes file for prediction.")
            self.bpe = BPE(codecs.open('embeddings/current_todo_codes.txt', encoding='utf-8'))
        return self.bpe.process_line(text)

    @staticmethod
    def create_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def get_local_directory_path(self):
        val = self.get_config('CLASSIFIER_DIR')
        val = val if val else 'TFLITE_MODEL'
        return self.create_dir(val)

    def get_todo_word_index_path(self):
        val = os.path.join(self.get_config('CLASSIFIER_DIR'), 'todo_models/architecture/lstm')
        return val

    def get_todo_model_path(self):
        val = os.path.join(self.get_config('CLASSIFIER_DIR'), 'todo_models/architecture/lstm/classify_email')
        return val

    def get_todo_vocab_path(self):
        val = os.path.join(self.get_config('CLASSIFIER_DIR'), 'todo_models/architecture/lstm/classify_email')
        return val

    def get_current_todo_word_index_path(self):
        val = os.path.join(self.get_config('CLASSIFIER_DIR'), 'current_todo_models/architecture/lstm')
        return val

    def get_current_todo_model_path(self):
        val = os.path.join(self.get_config('CLASSIFIER_DIR'), 'current_todo_models/architecture/lstm/classify_email')
        return val

    def get_current_todo_vocab_path(self):
        val = os.path.join(self.get_config('CLASSIFIER_DIR'), 'current_todo_models/architecture/lstm/classify_email')
        return val

def _load_application():
    return TfliteClassifier()

tflite_classifier = _load_application()

def get_application():
    return tflite_classifier
