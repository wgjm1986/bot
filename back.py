from flask import Flask, request, jsonify, Response, stream_with_context
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from pypdf import PdfReader
import re
import json

# 0. Create dictionary of documents that the students might ask about.
# This is identical to the dictionary created in the front-end code to populate the dropdown menu.
document_file_paths = {
    "Fall 2022 midterm 1"    : '../Exam 1/Fall 2022 midterm 1 - with answers.pdf',
    "Fall 2022 midterm 2"    : '../Exam 2/Fall 2022 midterm 2 - with answers.pdf',
    "Fall 2022 final exam"    : '../Exam 3/Fall 2022 final exam - with answers.pdf',
    "Fall 2023 midterm 1"    : '../Exam 1/Fall 2023 midterm 1 - with answers.pdf',
    "Fall 2023 midterm 2"    : '../Exam 2/Fall 2023 midterm 2 - with answers.pdf',
    "Fall 2023 final exam"    : '../Exam 3/Fall 2023 final exam - with answers.pdf',
    "Midterm 2 past questions" : '../Exam 2/past exam questions.pdf',
    "Final exam past questions" : '../Exam 3/Other past exam questions.pdf',
    "Module 1: Background"    : '../Module 1/Week 0 - Background information/Background.pdf',
    "Module 1: Investment returns, portfolios, and indexes" : '../Module 1/Week 1 - Aug 29 - Investment returns, portfolios, indexes/Investment returns, portfolios, and indexes.pdf',
    "Module 1: Fund structures and performance measures" : '../Module 1/Week 2 - Sep 5 - Fund structures and performance measures/Funds.pdf',
    "Module 1: Valuation theory" : '../Module 1/Week 3 - Sep 12 - Valuation theory/Valuation theory.pdf',
    "Module 1: Evidence on returns to active strategies" : '../Module 1/Week 4 - Sep 19 - Evidence on returns to active strategies/Evidence on returns to active strategies.pdf',
    "Module 2: Diversification and portfolio optimization" : '../Module 2/Week 6 - Oct 3 - Diversification and portfolio optimization/Diversification and portfolio optimization.pdf',
    "Module 2: Portfolio statistics and the CAPM" : '../Module 2/Week 8 - Oct 17 - Portfolio statistics and CAPM/Portfolio statistics and CAPM.pdf',
    "Module 2: Investment styles and the CAPM" : '../Module 2/Week 9 - Oct 24 - Investment styles and the CAPM/Investment styles and the CAPM.pdf',
    "Module 3: Short sales and dollar-neutral strategies" : '../Module 3/Week 11 - Nov 7 - Short sales and dollar-neutral strategies/Short sales and dollar-neutral strategies.pdf',
    "Module 3: Factor models" : '../Module 3/Week 12 - Nov 14 - Factor models/Factor models.pdf',
    "Module 3: The profitability factor" : '../Module 3/Week 13 - Nov 21 - The profitability factor/Profitability.pdf',
    "Module 3: Buffet's Alpha" : '../Module 3/Week 13 - Nov 21 - The profitability factor/Buffetts Alpha.pdf'
    }
no_selection_text = "No selection."

# 1. Load prebuilt vector database of embeddings of chunks from course content.
doc_db = FAISS.load_local("langchain_embeddings", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
# 2. Set up function to retrieve context from the vector store.
def retrieve_context(keywords):
    keywords_list = ', '.join(keywords)
    # Retrieve a broad set of results
    semantic_retriever = doc_db.as_retriever(search_kwargs={'k':20})
    semantic_search_results = semantic_retriever.invoke(keywords_list)
    # Narrow it down with a reranker
    rerank_retriever = BM25Retriever.from_documents( semantic_search_results , search_kwargs={'k':10} )
    rerank_results = rerank_retriever.invoke(keywords_list)
    # Flatten everything to a string that can be stuffed into the prompt
    context = '\n\n'.join([document.page_content for document in rerank_results])
    return context

# 3. Initialize LLM client
client = OpenAI()
# 4. Set up function to build prompt and query LLM
def query_LLM(query,chat_history_messages):

    # Gather context.
    helper_query_system_string = "I am teaching a college finance course. \
    A student has asked a question. \
    You cannot speak to the student, so you will not try to answer the question. \
    Instead, you are helping me build a prompt to a different LLM that will answer the question. \
    This LLM does not know anything about my course except what I provide in the prompt. \
    Below I will give you the student's question, and their previous questions to the LLM. \
    Then I will give you a list of course documents. \
    You should reply with the name of the course document that would be most useful to the LLM in answering the student's question. \
    If you do not select a document, reply with \""+no_selection_text+"\"." 
    helper_query_system_string += "\n\nHere is a list of the course documents that you can request: \n"
    for key in document_file_paths.keys():
        helper_query_system_string += key+'\n'

    user_messages = [{"role":"user","content":"Here is an earlier question the student asked: " + message['content']} for message in chat_history_messages if message['role']=="user"]

    helper_query_messages = [{"role":"system","content":helper_query_system_string}] + user_messages + [{"role":"user","content":"Here is the most recent question the student asked. Remember, do not try to answer the question, but instead reply with the name of a document that would be useful to an LLM trying to answer the question. \n" + query}]
    helper_query_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=helper_query_messages,
        max_tokens=2000,
        stream=False
    ).choices[0].message.content
    document_choice = helper_query_response.strip().rstrip('.')

    context_string = ""
    # Import and stuff the first relevant document
    if document_choice != no_selection_text:
        try:
            file_path = document_file_paths[document_choice]
            pdf = PdfReader(file_path)
            document = '\n\n'.join([page.extract_text() for page in pdf.pages])
            context_string += "Here is the course document that you already selected as being most useful to answer the student's question:\n"+document_choice+'\n'+document+'\n'
        except KeyError:
            pass

    helper_query_string_2 = "I am teaching a college finance course. \
    A student has asked a question. \
    You are helping me build a prompt that I can give you, which you will then use to answer the question. \
    Below I will give you the question, your conversation history with the student, and possibly a course document that you have already selected as being useful to answer the question. \
    Now I want you to reply with a list of keywords, separated by semicolons, that I should use in a semantic search through my course documents to provide you with additional context for the question."
    helper_query_messages = [{"role":"system","content":helper_query_string_2+context_string}] + chat_history_messages + [{"role":"user","content":query}]
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

# 5. Flask

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
