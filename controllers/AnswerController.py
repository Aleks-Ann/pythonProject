from flask import jsonify
from flask_restful import Resource, reqparse

from services.AnswerService1 import AnswerService1
from services.IAnswerService import IAnswerService

class AnswerController(Resource):

    def __init__(self, answer_service: IAnswerService = AnswerService1()):
        self.answer_service = answer_service

        self.parser = reqparse.RequestParser()
        self.parser.add_argument('request', type=dict, required=True)
        self.parser.add_argument('session', type=dict, required=True)
        self.parser.add_argument('version', type=str, required=True)

    def post(self):
        args = self.parser.parse_args()
        answer = self.answer_service.answer(args['request']['original_utterance'])
        response = {
            'session': args['session'],
            'version': args['version'],
            'response': {
                'end_session': True
            }
        }
        session_id = args['session']['session_id']
        if args['request']['original_utterance']:
            response['response']['text'] = answer
        else:
            response['response']['text'] = "Я чат-бот, отвечу на вопросы по поступлению в ВШЭ"
        return jsonify(response)

