import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tflite_classifier.application import get_application

tflite_classifier = get_application()

class TodoEmailTFlite:

    def __init__(self):
        self.model = None
        self.max_sentence_length = 50
        self.load_models()

    def load_models(self, loc=None, vocab_path=None):

        # loading idx2Label
        #self.idx2Label = np.load(os.path.join(tflite_classifier.get_vocab_path(),"idx2Label.npy"),allow_pickle=True).item()

        self.idx2Label = {count: word.strip() for count, word in
                          enumerate(open(os.path.join(tflite_classifier.get_todo_vocab_path(), "idx2Label.vocab")).readlines())}

        print("self.idx2Label::", self.idx2Label)
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=os.path.join(tflite_classifier.get_todo_model_path(),"model.tflite"))
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        #self.word2Idx = np.load(os.path.join(tflite_classifier.get_word_index_path(), "word2Idx.npy"), allow_pickle=True).item()
        self.word2Idx = {word.strip(): count for count, word in
                         enumerate(open(os.path.join(tflite_classifier.get_todo_word_index_path(), "word2Idx.vocab")).readlines())}
    def createTensor(self,sentence, word2Idx):
        unknownIdx = word2Idx['UNKNOWN_TOKEN']

        wordIndices = []

        for word in sentence:
            word = str(word)
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx

            wordIndices.append(wordIdx)
        wordIndices = pad_sequences([wordIndices], self.max_sentence_length, padding='post')[0]
        return [wordIndices]

    def predict(self, sentence):
        words = sentence.split()
        input_data = self.createTensor(words, self.word2Idx)
        input_data = np.asarray([input_data], dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data[0])
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output_details[0]['index'])
        score = max(pred[0])
        pred = pred[0].argmax()
        pred = self.idx2Label[pred].strip()
        return pred, score, 200