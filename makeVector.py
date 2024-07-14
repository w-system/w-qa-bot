import os

from openai import OpenAI

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))


def get_text_content():
    with open('rag-text.txt', 'r', encoding='utf-8') as file:
        text = file.read().replace('\n', '')
    return text


def chunk_text(text):
    chunks = text.split('=====')
    return chunks


def vectorize_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


if __name__ == '__main__':
    test_text = get_text_content()
    text_chunks = chunk_text(test_text)
    vectors = [vectorize_text(doc) for doc in text_chunks]
    # test_chunksをテキストへ出力
    with open('chunks.txt', 'w', encoding='utf-8') as file:
        file.write(str(text_chunks))

    # vectorsをテキストへ出力
    with open('vectors.txt', 'w', encoding='utf-8') as file:
        file.write(str(vectors))
