import re, json, sqlite3, openai, faiss, numpy as np
from flask import Flask, request, jsonify, Response, stream_with_context
from pypdf import PdfReader

# 1. Define a class to encapsulate all interactions with the database that was build in build_embeddings.py.
class DB_Search:

    def __init__(self, db_file):
        self.db_file = db_file
        self.document_descriptions = self.load_documents()
        self.embeddings_dict = self.load_embeddings()

    # Load the filenames and LLM-descriptions of all training documents from the database into a dictionary
    def load_documents(self):
        conn = sqlite3.connect('my_db.db')
        cursor = conn.cursor()
        cursor.execute('SELECT file_path, description FROM documents')
        data = cursor.fetchall()
        conn.close()
        document_descriptions = [ { "file_path":row[0] , "description":row[1] } for row in data ]
        return document_descriptions

    # Retrieve the text of a given document based on file path.
    # This is roughly identical to a function defined in build_embeddings.py and doesn't even really need to be within this class.
    # Eventually I would like to define it in some kind of shared code space to avoid duplication.
    def get_document_text(self,file_path):
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
            return
        if document_text and document_text != "": return document_text
        else:
            return

    # Load the ids and vector embddings of the text chunks that were built from the training data, but NOT the text itself to avoid overloading memory.
    def load_embeddings(self):
        conn = sqlite3.connect('my_db.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, embedding FROM chunks')
        data = cursor.fetchall()
        conn.close()
        embeddings_dict = {row[0]: np.array( np.frombuffer(row[1]) ) for row in data}
        return embeddings_dict

    # Perform semantic search on the embeddings, and read the text of the top k results directly from the database.
    def retrieve_context(self, keywords, k=10):
        # To do: I would like to loop over the keywords and do a semantic search on each separately, 
        # and then rerank to prioritize those matching earlier keywords, those from lecture materials as opposed to outside materials, and those with higher match quality,
        # but need to think about the exact way to encode all of that.
        keywords_list = ', '.join(keywords)
        print("Keywords:",keywords_list)
        query_embedding = np.array([ openai.embeddings.create(model="text-embedding-ada-002",input=keywords_list).data[0].embedding ])
        query_embedding = query_embedding.flatten()
        # using FAISS index: if I want to do this, create the index in the __init__ method above.
        # distances_array, indices_array = index.search(query_embedding, k)
        # alternatively, just do it by hand
        similarities = {}
        for chunk_id, embedding in self.embeddings_dict.items():
            # Uncomment the end of the line to switch from inner product similarity to cosine similarity. 
            similarities[chunk_id] = np.dot( embedding, query_embedding ) # / ( np.linalg.norm( embedding ) * np.linalg.norm( query_embedding ) )
        top_k_ids = sorted(similarities, key=similarities.get, reverse=True)[:k]
        # retrieve the actual text chunks from the database
        conn = sqlite3.connect('my_db.db')
        cursor = conn.cursor()
        placeholders = ','.join('?' for _ in top_k_ids)
        query = f'SELECT id, chunk_text FROM chunks WHERE id IN ({placeholders})'
        cursor.execute(query,top_k_ids)
        data = cursor.fetchall()
        conn.close()
        # flatten text chunks to a single line
        context = '\n\n'.join([row[1] for row in data])
        print("Context:",context)
        return context

# Initialize the class used for database search.
db_search = DB_Search("my_db.db")

# 2. Initialize LLM client and set up function to build prompt and query LLM
client = openai.OpenAI()

def query_LLM(query,chat_history_messages):
    # Each call performs three queries. 
    # The first two use the lightweight LLM 4o-mini for query enhancement and context augmentation on the original question.
    # The final call uses the heavy LLM 4o to build the answer based on all retrieved context.

    # Prompt 1: Have the lightweight LLM request a course document that would be useful.
    no_selection_text = "No selection."

    # Instructions
    helper_query_system_string = "I am teaching a college finance course. \
    A student has asked a question. \
    You cannot speak to the student, so you will not try to answer the question. \
    Instead, you are helping me build a prompt to a different LLM that will answer the question. \
    This LLM does not know anything about my course except what I provide in the prompt. \
    Below I will give you the student's question, and their previous questions to the LLM. \
    Then I will give you a list of course documents, where the first line is the filename of the document, and the next line is a short description of the document along with some keywords. \
    You should select a filename from the list that would be most useful to the LLM in answering the student's question, and reply with only that filename. \
    If you do not select a document, reply with \""+no_selection_text+"\"." 
    
    # Flatten all document filenames and descriptions into text that I can feed to the LLM.
    document_descriptions_string = '\n\n'.join( [ document_description['file_path'] + '\n' + document_description['description'] for document_description in db_search.document_descriptions ] )

    annotated_messages = [
        {
            'role' : message['role'],
            'content' : (
                "Here is an earlier question the student asked: " + message['content'] if message['role'] == "user"
                else "Here was the LLM's answer: " + message['content'] if message['role'] == "assistant"
                else message['content']
            )
        }
        for message in chat_history_messages
    ]

    helper_query_messages = [{"role":"system","content":helper_query_system_string + document_descriptions_string}] \
        + annotated_messages + [{"role":"user","content":"Here is the most recent question the student asked: " + query}]
    helper_query_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=helper_query_messages,
        max_tokens=2000,
        stream=False
    ).choices[0].message.content
    document_choice = helper_query_response.strip().rstrip('.')
    print(document_choice)

    # Prompt 2: Have the lightweight LLM request keywords that would be useful in semantic search, given the document it already chose.

    context_string = ""

    # Import and stuff the relevant document that was already chosen.
    if document_choice != no_selection_text and document_choice != no_selection_text.rstrip('.'):
        document_text = db_search.get_document_text(document_choice)
        context_string += "Here is the course document that you already selected as being most useful to answer the student's question:\n"+document_choice+'\n'+document_text+'\n'

    helper_query_string_2 = "I am teaching a college finance course. \
    A student has asked a question. \
    You cannot speak to the student, so you will not try to answer the question. \
    Instead, you are helping me build a prompt to a different LLM that will answer the question. \
    This LLM does not know anything about my course except what I provide in the prompt. \
    Below I will give you the question, your conversation history with the student, and possibly a course document that you have already selected as being useful to answer the question. \
    Now I want you to reply with a list of four or fewer keywords that I should use in a semantic search through my course documents to provide the LLM with additional context for the question. \
    Start your answer with the first keyword, not with any kind of label, and separate each keyword in the list with a semicolon."
    helper_query_messages = [{"role":"system","content":helper_query_string_2+context_string}] + annotated_messages + [{"role":"user","content":"Here is the most recent question the student asked. Remember, do not try to answer the question, but instead reply with keywords that could be used for a semantic search to give useful information to an LLM trying to answer the question. \n" + query}]
    helper_query_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=helper_query_messages,
        max_tokens=2000,
        stream=False
    ).choices[0].message.content
    keywords = [keyword.strip() for keyword in helper_query_response.split(';')]
    
    # Retrieve other context from course materials using the function defined above.
    other_context = db_search.retrieve_context(keywords)
    context_string += "\nHere is some other content from the course materials that is related to the student's question:\n"+other_context

    # Prompt 3: Have the heavy LLM build an answer based on the document and keywords.
    
    ai_answer_query_system_string = "You are a helpful TA answering student questions in a college finance course. \
    You refuse to answer questions about people or topics that are not mentioned in the course materials provided below. \
    You also refuse to give any information about other students in the class. \
    Whenever a dollar sign, percentage sign, or ampersand should appear in the output, you escape it with a backslash. \
    Below is some information from the course documents that will be useful in answering the student's question. \
    You can use outside information as well, but the information I provide is always more reliable. \
    \n\n"

    ai_answer_query_messages = [{"role":"system","content":ai_answer_query_system_string+context_string}] + chat_history_messages + [{"role":"user","content":query}]
    
    ai_response_stream = client.chat.completions.create(
        model="gpt-4o",
        messages=ai_answer_query_messages,
        max_tokens=2000,
        stream=True
    )

    # Retrieve just the text from each chunk in the response stream, serialize with JSON, and yield it as output
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
