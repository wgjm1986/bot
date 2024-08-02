import sys, json, sqlite3, openai, tiktoken, numpy as np
from flask import Flask, request, Response, stream_with_context
from text_tools import get_document_paragraphs

# Import course settings based on parameter passed
with open('courses.json','r') as courses_file: 
    course_data = json.load(courses_file).get( sys.argv[1] ,{})
if not course_data: raise ValueError(f"No course data found for {sys.argv[1]}")
db_path = f"{course_data['db_file']}"

# 1. Define a class to encapsulate all interactions with the database that was build in build_embeddings.py.
class DB_Search:

    def __init__(self, db_file):
        self.db_file = db_file
        self.documents = self.load_documents()
        self.embeddings_dict = self.load_embeddings()

    # Load the filenames and LLM-descriptions of all training documents from the database into a dictionary
    def load_documents(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('SELECT file_path, description FROM documents')
        data = cursor.fetchall()
        conn.close()
        documents = [ { "file_path":row[0] , "description":row[1] } for row in data ]
        return documents

    # Load the ids and vector embddings of the text chunks that were built from the training data, but NOT the text itself to limit memory usage.
    def load_embeddings(self):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('SELECT id, embedding FROM chunks')
        data = cursor.fetchall()
        conn.close()
        embeddings_dict = {row[0]: np.array( np.frombuffer(row[1]) ) for row in data}
        return embeddings_dict

    # Perform semantic search on the embeddings, and read the text of the top k results directly from the database.
    def retrieve_context(self, keywords, k=5):
        # To do: I would like to loop over the keywords and do a semantic search on each separately, 
        # and then rerank to prioritize those matching earlier keywords, those from lecture materials as opposed to outside materials, and those with higher match quality,
        # but need to think about the exact way to encode all of that.
        keywords_list = ', '.join(keywords)
        # get embeddings of the query
        query_embedding = np.array( openai.embeddings.create(model="text-embedding-ada-002",input=keywords_list).data[0].embedding )
        # using FAISS index: if I want to do this, create the index in the __init__ method above.
        # distances_array, indices_array = index.search(query_embedding, k)
        # alternatively, just do it by hand
        similarities = {}
        for chunk_id, embedding in self.embeddings_dict.items():
            # Uncomment the end of the line to switch from inner product similarity to cosine similarity. 
            similarities[chunk_id] = np.dot( embedding, query_embedding ) # / ( np.linalg.norm( embedding ) * np.linalg.norm( query_embedding ) )
        top_k_ids = sorted(similarities, key=similarities.get, reverse=True)[:k]
        # retrieve the actual text chunks from the database
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        placeholders = ','.join('?' for _ in top_k_ids)
        query = f'SELECT id, chunk_text FROM chunks WHERE id IN ({placeholders})'
        cursor.execute(query,top_k_ids)
        data = cursor.fetchall()
        conn.close()
        # flatten text chunks to a single line
        context = '\n\n'.join([row[1] for row in data])
        # print("Context:",context)
        return context
# Initialize the class used for database search.
db_search = DB_Search(db_path)

# 2. Set up functions to build prompt and query LLM
client = openai.OpenAI()
no_selection_text = "No selection."

def document_prompt(query,chat_history_messages):
    helper_query_system_string = (
    f"I am teaching a college course on {course_data['topic']}.  "
    "A student has been having a conversation with a teaching assistant, and has just asked a question.  "
    "Now I would like to use an LLM to provide the answer that the TA would give. "
    "\n\nYou cannot speak to the student, so you will not try to answer the question. "
    "Instead, you are helping me build a prompt to a different LLM that will answer the question. "
    "This LLM will not know anything about my course except the information that you choose to provide! "
    "\n\nBelow I will give you the student's conversation with the TA so far, and the question they just asked. "
    "Then I will give you a list of course documents, where the first line is the filename, and the second line is a short description of the document along with some keywords. "
    "You should select the filename of the document that would be most useful to the LLM in providing the next message in the conversation.  "
    "Reply with only that filename, and do not enclose it in quotes of any kind. "
    "\n\nPlease remember that the LLM is not the TA, and will not know ANYTHING about the course except the document you select!  "
    "Therefore, if the student and the TA have been discussing a specific course document and the student has asked another question about it,  "
    "then you should choose that document from the list so that the LLM will understand the conversation so far.  ")
    
    # Flatten all document filenames and descriptions into text that I can feed to the LLM.
    document_descriptions_string = '\n\n'.join( 
            [ document['file_path'] + '\n' + document['description'] for document in db_search.documents if document['description'] != "Irrelevant" ] 
        )
    # print(document_descriptions_string)

    # Tag prior messages and query to remind the LLM that it is not part of this conversation, otherwise it tries to reply itself.
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
    annotated_query = ("Here is the student's current question. Remember, do not try to answer the question yourself. "
        "Instead reply with a filename of a document from the list I provided that would help an LLM to answer this question: "
        + query )

    helper_query_messages = ( [{"role":"system","content":helper_query_system_string + document_descriptions_string}] 
                                + annotated_messages 
                                + [{"role":"user","content":annotated_query}] )
    print("############## SENDING FIRST PROMPT")
    helper_query_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=helper_query_messages,
        max_tokens=100,
        stream=False
    )
    print("############## RESPONSE RECEIVED")

    input_tokens = helper_query_response.usage.prompt_tokens
    print(f'          {input_tokens} input tokens used,')
    output_tokens = helper_query_response.usage.completion_tokens
    print(f'          {output_tokens} output tokens used.')
    total_tokens = helper_query_response.usage.total_tokens
    print(f'          {total_tokens} total tokens used.')

    helper_query_response_string = helper_query_response.choices[0].message.content
    document_choice = helper_query_response_string.strip().strip('.`\"\'')

    return document_choice

def keyword_prompt(query,chat_history_messages,document_choice):
    helper_query_string = (
    f"I am teaching a college course on {course_data['topic']}.  "
    "A student has been having a conversation with a teaching assistant, and has just asked a question.  "
    "Now I would like to use an LLM to provide the answer that the TA would give. "
    "\n\nYou cannot speak to the student, so you will not try to answer the question. "
    "Instead, you are helping me build the prompt to a different LLM that will answer the question. "
    "This LLM will not know anything about my course except the information that you choose to provide! "
    "\n\nBelow I will give you the student's conversation with the TA so far, and the question they just asked. "
    "I may also provide a course document that you have already selected as being useful to answer the question. "
    "I want you to reply with a list of one to four keywords that I should use in a search through my course documents to provide the LLM with additional context for the question. "
    "Start your answer with the first keyword, not with any kind of label, and separate each keyword in the list with a semicolon."
    )

    context_string = ""

    if document_choice and document_choice.rstrip('.') != no_selection_text.rstrip('.'):
        try:
            document_paragraphs = get_document_paragraphs(document_choice)
            document_text = '\n\n'.join(document_paragraphs)
            # print("#### RETRIEVED DOCUMENT:\n\n"+document_text)
            if document_text: context_string += "Here is the course document that you already selected as being most useful to answer the student's question:\n"+document_choice+'\n'+document_text+'\n'
        except FileNotFoundError:
            pass

    # Tag prior messages and query to remind the LLM that it is not part of this conversation, otherwise it tries to reply itself.
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
    annotated_query = ("Here is the most recent question the student asked. Remember, do not try to answer the question yourself. "
        "Instead reply with keywords for a semantic search to give useful information to an LLM trying to answer the question: "
        + query)

    helper_query_messages = [{"role":"system","content":helper_query_string+context_string}] + annotated_messages + [{"role":"user","content":annotated_query}]
    print("############## SENDING SECOND PROMPT")
    helper_query_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=helper_query_messages,
        max_tokens=100,
        stream=False
    )
    print("############## RESPONSE RECEIVED")

    input_tokens = helper_query_response.usage.prompt_tokens
    print(f'          {input_tokens} input tokens used,')
    output_tokens = helper_query_response.usage.completion_tokens
    print(f'          {output_tokens} output tokens used.')
    total_tokens = helper_query_response.usage.total_tokens
    print(f'          {total_tokens} total tokens used.')

    helper_query_response_string = helper_query_response.choices[0].message.content
    keywords = [keyword.strip() for keyword in helper_query_response_string.split(';')]
    return keywords

def query_LLM(query,chat_history_messages):

    context_string = ""

    # Prompt 1: Have the LLM request a course document that would be useful.
    document_choice = document_prompt(query,chat_history_messages)
    print(document_choice)
    if document_choice != no_selection_text and document_choice != no_selection_text.rstrip('.'):
        try:
            document_paragraphs = get_document_paragraphs(document_choice)
            document_text = '\n\n'.join(document_paragraphs)
            context_string += "Here is the course document that you already selected as being most useful to answer the student's question:\n"+document_choice+'\n'+document_text+'\n'
        except FileNotFoundError:
            pass
    print(context_string)

    # Prompt 2: Have the LLM request keywords that would be useful in semantic search, given the document it already chose.
    keywords = keyword_prompt(query,chat_history_messages,document_choice)
    other_context = db_search.retrieve_context(keywords)
    context_string += "\nHere is some other content from the course materials that is related to the student's question:\n"+other_context
    
    # Prompt 3: Have the LLM build an answer based on the document and keywords.
    ai_answer_query_system_string = (
    f"You are a helpful TA answering student questions in a college course on {course_data['topic']}. "
    "You refuse to answer questions about people or topics that are not mentioned in the course materials provided below. "
    "You also refuse to give any information about other students in the class. "
    "Below is some information from the course documents that will be useful in answering the student's question. "
    "You can use outside information as well, but the information I provide is always more reliable. "
    "\n\n"
    )

    ai_answer_query_messages = [{"role":"system","content":ai_answer_query_system_string+context_string}] + chat_history_messages + [{"role":"user","content":query}]
    print("############ SENDING THIRD PROMPT")
    ai_response_stream = client.chat.completions.create(
        model="gpt-4o",
        messages=ai_answer_query_messages,
        max_tokens=2000,
        stream=True,
        stream_options={"include_usage":True}
    )
    print("############ RESPONSE RECEIVED")

    # Retrieve just the text from each chunk in the response stream, serialize with JSON, and yield it as output
    for chunk in ai_response_stream:
        # With include_usage set to true above, an extra token is added to the end of the stream to give usage statistics
        if chunk.usage:
            print(f"          Prompt tokens used: {chunk.usage.prompt_tokens}") 
            print(f"          Completion tokens used: {chunk.usage.completion_tokens}") 
            print(f"          Total tokens used: {chunk.usage.total_tokens}") 
            return
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
    app.run(host='127.0.0.1',port=course_data['api_port'])
