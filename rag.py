import os

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


class RagOpenAI:
    def __init__(self, question):
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY')
        self.question = question


    def vectorize_text(self, text):
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def find_most_similar(self, question_vector, vectors, documents):
        similarities = []

        for index, vector in enumerate(vectors):
            similarity = cosine_similarity([question_vector], [vector])[0][0]
            similarities.append([similarity, index])

        similarities.sort(reverse=True, key=lambda x: x[0])
        top_documents = [documents[index] for similarity, index in similarities[:2]]

        return top_documents

    def ask_question(self, question, context):
        prompt = f'''以下の質問に以下の情報をベースにして答えて下さい。
      [ユーザーの質問]
      {question}
    
      [情報]
      {context}
      '''
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4o",
        )
        return response.choices[0].message.content

    def call_func_rag(self):
        # vectors.txtからvectorsを読み込む
        with open('vectors.txt', 'r', encoding='utf-8') as file:
            vectors = file.read()

        # 内容をリスト形式に変換する
        vectors = eval(vectors)

        # chunks.txtからtext_chunksを読み込む
        with open('chunks.txt', 'r', encoding='utf-8') as file:
            text_chunks = file.read()
        text_chunks = eval(text_chunks)

        question_vector = self.vectorize_text(self.question)
        similar_document = self.find_most_similar(question_vector, vectors, text_chunks)
        answer = self.ask_question(self.question, similar_document)
        return answer


