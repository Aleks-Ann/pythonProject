from flask import Flask

from flask_ngrok import run_with_ngrok
from flask_restful import Api

from controllers.AnswerController import AnswerController

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yandexlyceum_secret_key'
run_with_ngrok(app)

api = Api(app, catch_all_404s=True)


def main():

    api.add_resource(AnswerController, '/')

    app.run()


if __name__ == '__main__':
    main()