import re
from flask import request, jsonify, Flask
from tflite_classifier import nlog as log
from tflite_classifier.application import get_application
from tflite_classifier.todo_classifier_predict import TodoEmailTFlite
from tflite_classifier.new_todo_classifier_predict import NewTodoEmailTFlite

todo_ae = TodoEmailTFlite()

current_todo_ae = NewTodoEmailTFlite()

tflite_classifier = get_application()


class EmailFlask(Flask):

    def __init__(self, name):
        super().__init__(name)
        self.add_routes()

    def add_basic_routes(self):
        self.add_url_rule('/', 'index', self.index)
        self.add_url_rule('/health', 'health', self.health)
        self.add_url_rule('/readiness', 'readiness', self.readiness_check)
        self.add_url_rule('/liveness', 'liveness', self.liveness_check)
        self.add_url_rule('/status', 'status', self.status)

    def index(self):
        return 'Welcome to i.am+ ML Server : ' + self.name

    @staticmethod
    def health():
        return jsonify(['ok'])

    # pylint: disable=no-self-use
    def liveness_check(self):
        return jsonify(['ok'])

    def readiness_check(self):
        if self.is_ready():
            return '', 204

        return jsonify(self.get_status()), 503

    def status(self):
        return jsonify(self.get_status())

    def is_ready(self) -> bool:
        return True

    @staticmethod
    def get_status() -> str:
        return 'Invalid intent or models are not present'

    def add_routes(self):
        self.add_basic_routes()
        self.add_url_rule('/todo_classifier',
                          'todo_classifier',
                          self.todo_classifier,
                          methods=["POST"])
        self.add_url_rule('/new_todo_classifier',
                          'new_todo_classifier',
                          self.current_todo_classifier,
                          methods=["POST"])

    def classify_main(self, request_json, classifier):

        query = request_json.get('query')

        if classifier == 'todo_classifier':
            action_sentences = []
            final_answer, final_score, final_status_code = None, None, None
            email_sentences = re.split(';|\.|\?|\!', query)
            email_sentences = list(filter(lambda x: x != '', email_sentences))
            print("email_sentences::", email_sentences)
            for sentence in email_sentences:
                query_ = " ".join(tflite_classifier._process_text(sentence))
                print(query_)
                answer, score, status_code = todo_ae.predict(tflite_classifier.apply_bpe(query_, classifier='todo'))
                if answer == 'action':
                    print("reply sentence is::", sentence)
                    action_sentences.append({sentence:float(score)})
                    final_answer, final_score, final_status_code = answer, score, status_code
            if final_answer:
                answer, score, status_code = final_answer, final_score, final_status_code

        elif classifier == 'current_todo_classifier':
            action_sentences = []
            final_answer, final_score, final_status_code = None, None, None
            email_sentences = re.split(';|\.|\?|\!', query)
            email_sentences = list(filter(lambda x: x != '', email_sentences))
            print("email_sentences::", email_sentences)
            for sentence in email_sentences:
                query_ = " ".join(tflite_classifier._process_text(sentence))
                answer, score, status_code = current_todo_ae.predict(tflite_classifier.apply_bpe(query_, classifier='current_todo'))

                if answer == 'action':
                    print("reply sentence is::", sentence)
                    action_sentences.append({sentence:float(score)})
                    final_answer, final_score, final_status_code = answer, score, status_code
            if final_answer:
                answer, score, status_code = final_answer, final_score, final_status_code
        else:
            return {}, 200

        if status_code == 200:
            result_json = {'intent': answer, 'action_sentences': action_sentences, 'algorithm': 'lstm', 'language': 'en'}
            return result_json, status_code
        else:
            result_json = self.get_status()

        return result_json, status_code

    def todo_classifier(self):
        request_json = request.json or {}
        result, status_code = self.classify_main(request_json, 'todo_classifier')
        return jsonify(result), status_code

    def current_todo_classifier(self):
        request_json = request.json or {}
        result, status_code = self.classify_main(request_json, 'current_todo_classifier')
        return jsonify(result), status_code


app = EmailFlask(__name__)
log.info('** email initialization complete.')
