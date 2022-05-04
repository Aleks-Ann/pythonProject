from services.IAnswerService import IAnswerService
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.neighbors import KDTree
import requests
from bs4 import BeautifulSoup
import pandas as pd

m = "cointegrated/rubert-tiny"
tokenizer = AutoTokenizer.from_pretrained(m)  # Создаем токенизатор
model = AutoModel.from_pretrained(m)  # Создаем модель


class AnswerService1(IAnswerService):

    def answer(self, text: str):
        # 1.Запросить все вопросы у репозитория
        soup = BeautifulSoup(requests.get('https://ma.hse.ru/faq').text)
        questions = []
        answers = []
        for div in soup.findAll('div', {'class': 'faq'}):
            questions.append(div.find('div', {'class': 'faq__question'}).text.strip())
            answers.append(div.find('div', {'class': 'faq__answer'}).text.strip())
        print(len(questions))
        data = pd.DataFrame({'q': questions, 'a': answers})
        # 2
        vectors = np.stack([embed_rubert(t) for t in data.q])
        index = KDTree(vectors)

        distances, indices = index.query(embed_rubert([text])[np.newaxis, :], k=3)
        if distances[0][0] > 0.77:
            return f'Кажется, у меня нет ответа на ваш вопрос. Может быть, вы хотите знать, "{data.q[np.random.choice(indices[0])]}"?'
        else:
            return data.a[indices[0][0]]

    def normalize(v):
        return v / sum(v ** 2) ** 0.5


def embed_rubert(text, mean=False):
    # Получаем токены
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
    # В режиме экспдуатации сети подаем эмбеддинги на вход модели
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings)
    return sentence_embeddings[0].numpy()

# distances, indices = index.query(embed_rubert([text])[np.newaxis, :], k=3)
# for i, d in zip(indices[0], distances[0]):
# print(i, d, data.q[i])
# return text