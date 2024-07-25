from flask import Flask, request, jsonify, Response, stream_with_context
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from pypdf import PdfReader
import re
import json
import openai
import faiss, sqlite3
import numpy as np

def get_document_text(file_path):
    extension = file_path[-3:]
    if extension == "txt" or extension == "tex":
        with open(file_path, 'r') as file:
            document_text = file.read()
    elif extension == "pdf":
        document_text = ""
        pdf = PdfReader(file_path)
        for page in pdf.pages:
            try:
                document_text += page.extract_text() + "\n\n"
                # document_text += page.extract_text(extraction_mode="layout") + "\n\n"
            except:
                print("PDF read failure: " + file_path)
    elif extension == "docx":
        doc = docx.Document(file_path)
        document_text = '\n\n'.join([p.text for p in doc.paragraphs])
    elif extension == "pptx":
        pres = pptx.Presentation(file_path)
        document_text = '\n\n'.join([get_slide_text(slide) for slide in pres.slides])
    else:
        print("Unsupported file type: " + file_path)
        return
    if document_text and document_text != "":
        # print("success: " + file_path)
        if re.match("Exam|exam|Midterm|midterm",file_path):
            source_type = "Tests, midterms, and exams from past years"
        elif re.match("Module",file_path):
            source_type = "Teaching materials"
        elif re.match("Textbook",file_path):
            source_type = "Textbook"
        elif re.match("Transcripts",file_path):
            source_type = "Transcripts of class sessions"
        elif re.match("Discussion",file_path):
            source_type = "Media articles"
        else:
            source_type = "Other"
        return document_text
    else:
        print("failure: " + file_path)
        return


# 0. Create dictionary of course documents that the LLM can request.
conn = sqlite3.connect('my_db.db')
cursor = conn.cursor()
cursor.execute('SELECT file_path, description FROM documents')
data = cursor.fetchall()
document_descriptions = [ { "file_path":row[0] , "description":row[1] } for row in data ]
no_selection_text = "No selection."

# 1. Load vector embeddings and set up function to retrieve context
conn = sqlite3.connect('my_db.db')
cursor = conn.cursor()
cursor.execute('SELECT id, embedding FROM chunks')
data = cursor.fetchall()
ids = [row[0] for row in data]
embeddings = np.array([np.frombuffer(row[1]) for row in data])
conn.close()
# Calculate length of each individual embedding
dimension = embeddings.shape[1]
# Use inner product, but not cosine similarity (I conjecture that inner product is better for our use case)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
# if desired we can write the index to a file; this may become valuable at some level of scale
# faiss.write_index(index,filename)
# To do: I would like to loop over the keywords and do a semantic search on each separately, 
# and then rerank to prioritize those matching earlier keywords, those from lecture materials as opposed to outside materials, and those with higher match quality,
# but need to think about the exact way to encode all of that.
def retrieve_context(keywords):
    keywords_list = ', '.join(keywords)
    print("Keywords:",keywords_list)
    query_embedding = np.array([ openai.embeddings.create(model="text-embedding-ada-002",input=keywords_list).data[0].embedding ])
    distances_array, indices_array = index.search(query_embedding, 10)
    distances = distances_array[0]
    indices = indices_array[0]
    retrieved_ids = [ids[index] for index in indices]
    conn = sqlite3.connect('my_db.db')
    cursor = conn.cursor()
    placeholders = ','.join('?' for _ in retrieved_ids)
    query = f'SELECT id, chunk_text FROM chunks WHERE id IN ({placeholders})'
    cursor.execute(query,retrieved_ids)
    data = cursor.fetchall()
    conn.close()
    context = '\n\n'.join([row[1] for row in data])
    print("Context:",context)
    return context

# 2. Initialize LLM client and set up function to build prompt and query LLM
client = openai.OpenAI()
def query_LLM(query,chat_history_messages):

    # Gather context.
    helper_query_system_string = "I am teaching a college finance course. \
    A student has asked a question. \
    You cannot speak to the student, so you will not try to answer the question. \
    Instead, you are helping me build a prompt to a different LLM that will answer the question. \
    This LLM does not know anything about my course except what I provide in the prompt. \
    Below I will give you the student's question, and their previous questions to the LLM. \
    Then I will give you a list of course documents, where the first line is the file path of the document, and the next line is a short description of the document along with some keywords. \
    You should reply with the file path of the course document that would be most useful to the LLM in answering the student's question. \
    If you do not select a document, reply with \""+no_selection_text+"\"." 
    for document_description in document_descriptions: 
        if document_description['file_path'][-3:] != "txt":
            helper_query_system_string += document_description['file_path'] + '\n' + document_description['description'] + '\n\n'

    user_messages = [{"role":"user","content":"Here is an earlier question the student asked: " + message['content']} for message in chat_history_messages if message['role']=="user"]

    helper_query_messages = [{"role":"system","content":helper_query_system_string}] + user_messages + [{"role":"user","content":"Here is the most recent question the student asked. Remember, do not try to answer the question, but instead reply with the name of a document that would be useful to an LLM trying to answer the question. \n" + query}]
    helper_query_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=helper_query_messages,
        max_tokens=2000,
        stream=False
    ).choices[0].message.content
    document_choice = helper_query_response.strip().rstrip('.')
    print(document_choice)

    context_string = ""
    # Import and stuff the first relevant document
    if document_choice != no_selection_text and document_choice != no_selection_text.rstrip('.'):
        document_text = get_document_text(document_choice)
        context_string += "Here is the course document that you already selected as being most useful to answer the student's question:\n"+document_choice+'\n'+document_text+'\n'

    helper_query_string_2 = "I am teaching a college finance course. \
    A student has asked a question. \
    You cannot speak to the student, so you will not try to answer the question. \
    Instead, you are helping me build a prompt to a different LLM that will answer the question. \
    This LLM does not know anything about my course except what I provide in the prompt. \
    Below I will give you the question, your conversation history with the student, and possibly a course document that you have already selected as being useful to answer the question. \
    Now I want you to reply with a list of four or fewer keywords that I should use in a semantic search through my course documents to provide the LLM with additional context for the question. \
    Start your answer with the first keyword, not with any kind of label, and separate each keyword in the list with a semicolon."
    helper_query_messages = [{"role":"system","content":helper_query_string_2+context_string}] + user_messages + [{"role":"user","content":"Here is the most recent question the student asked. Remember, do not try to answer the question, but instead reply with keywords that could be used for a semantic search to give useful information to an LLM trying to answer the question. \n" + query}]
    helper_query_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=helper_query_messages,
        max_tokens=2000,
        stream=False
    ).choices[0].message.content
    keywords = [keyword.strip() for keyword in helper_query_response.split(';')]
    
    # Retrieve other context from course materials using the function defined above.
    other_context = retrieve_context(keywords)
    context_string += "\nHere is some other content from the course materials that is related to the student's question:\n"+other_context

    ####
    
    # Base system prompt.
    ai_answer_query_system_string = "You are a helpful TA answering student questions in a college finance course. \
    You refuse to answer questions about people or topics that are not mentioned in the course materials provided below. \
    You also refuse to give any information about other students in the class. \
    Whenever a dollar sign, percentage sign, or ampersand should appear in the output, you escape it with a backslash. \
    Below is some information from the course documents that will be useful in answering the student's question. \
    You can use outside information as well, but the information I provide is always more reliable. \
    \n\n"

    # Build dictionary of messages in the format expected by the OpenAI API
    ai_answer_query_messages = [{"role":"system","content":ai_answer_query_system_string+context_string}] + chat_history_messages + [{"role":"user","content":query}]
    
    # Send request off to the API and retrieve response.
    ai_response_stream = client.chat.completions.create(
        model="gpt-4o",
        messages=ai_answer_query_messages,
        max_tokens=2000,
        stream=True
    )

    # Retrieve just the text from each chunk in the response stream, and return it serialized with JSON
    for chunk in ai_response_stream:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content:
            yield json.dumps({"token": chunk_content})+'\n'

# 3. Flask code

# Set up Flask server
app = Flask(__name__)

# Route decorator to attach get_response() to localhost:5000/get_response
@app.route('/get_response',methods=['POST'])
def get_response():
    # Flask automatically attaches the JSON payload to flask.request.json where it can be unpacked with get(). 
    query = request.json.get('query')
    chat_history_messages = request.json.get('chat_history_messages')
    # Get LLM response as a generator
    LLM_response = query_LLM(query,chat_history_messages)
    # Wrap the generator in Flask objects to stream content back to the API
    return Response( stream_with_context( LLM_response ), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000)
