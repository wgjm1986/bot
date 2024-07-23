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
	"Fall 2022 midterm 1"	: '../Exam 1/Fall 2022 midterm 1 - with answers.pdf',
	"Fall 2022 midterm 2"	: '../Exam 2/Fall 2022 midterm 2 - with answers.pdf',
	"Fall 2022 final exam"	: '../Exam 3/Fall 2022 final exam - with answers.pdf',
	"Fall 2023 midterm 1"	: '../Exam 1/Fall 2023 midterm 1 - with answers.pdf',
	"Fall 2023 midterm 2"	: '../Exam 2/Fall 2023 midterm 2 - with answers.pdf',
	"Fall 2023 final exam"	: '../Exam 3/Fall 2023 final exam - with answers.pdf',
	"Midterm 2 past questions" : '../Exam 2/past exam questions.pdf',
	"Final exam past questions" : '../Exam 3/Other past exam questions.pdf',
	"Module 1: Background"	: '../Module 1/Week 0 - Background information/Background.pdf',
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
no_selection_text = "No selection"

# 1. Load prebuilt vector database of embeddings of chunks from course content.
doc_db = FAISS.load_local("langchain_embeddings", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
# 2. Set up function to retrieve context from the vector store.
def retrieve_context(query,document=""):
	# The function accepts the user query, and an optional document that the user was asking about.
	# We first retrieve a broad set of relevant content using semantic search and including the document (if any).
	semantic_retriever = doc_db.as_retriever(search_kwargs={'k':20})
	semantic_search_results = semantic_retriever.invoke(query+'\n\n'+document)
	# We then narrow it down to the user's specific question with a reranker that focuses only on the query.
	rerank_retriever = BM25Retriever.from_documents( semantic_search_results , search_kwargs={'k':10} )
	rerank_results = rerank_retriever.invoke(query)
	# Flatten everything to a string that can be stuffed into the prompt
	context = '\n\n'.join([document.page_content for document in rerank_results])
	return context

# 3. Initialize LLM client
client = OpenAI()
# 4. Set up function to build prompt and query LLM
def query_LLM(query,chat_history_messages,document_choice):
	
	# Base system prompt.
	ai_answer_query_system_string = "You are a helpful TA answering student questions in a college finance course. \
	You refuse to answer questions about people or topics that are not mentioned in the course materials provided below. \
	You also refuse to give any information about other students in the class. \
	Whenever a dollar sign, percentage sign, or ampersand should appear in the output, you escape it with a backslash. \n\n"

	# Import the document that the user was asking about, if any, and stuff the whole document into the prompt.
	if document_choice != no_selection_text:
		file_path = document_file_paths[document_choice]
		pdf = PdfReader(file_path)
		document = '\n\n'.join([page.extract_text() for page in pdf.pages])
		ai_answer_query_system_string += "Here is a specific document from class that the student is asking about:"+document+'\n'

	# Retrieve other context from course materials using the function defined above.
	if document_choice == no_selection_text:
		context = retrieve_context(query)
	else:
		context = retrieve_context(query,document)
	ai_answer_query_system_string += "Here is some content from the course materials that is related to the student's question. \
	You can use outside information as well, but the information below is more reliable, so use it whenever possible. \
	If you rely on outside information, mention that in your answer, since it might be inconsistent with what Professor Mann said in class.\n"+context

	# Build dictionary of messages in the format expected by the OpenAI API
	ai_answer_query_messages = [{"role":"system","content":ai_answer_query_system_string}] + chat_history_messages + [{"role":"user","content":query}]
	
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
	document_choice = request.json.get('document_choice')
	chat_history_messages = request.json.get('chat_history_messages')
	# Get LLM response as a generator
	LLM_response = query_LLM(query,chat_history_messages,document_choice)
	# Wrap the generator in Flask objects to stream content back to the API
	return Response( stream_with_context( LLM_response ), content_type='text/event-stream')

if __name__ == '__main__':
	app.run(host='127.0.0.1',port=5000)
