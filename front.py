import streamlit as st
import requests
import json
import re

# Build a dictionary of documents that the students might ask about.
# The identical dictionary is created in back.py.
# To do: build it automatically or save it in an external file so I don't have duplicate code between the two files.
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

# Set up the welcome message:
welcome_message = """
	Hello, human! I am the virtual TA for FIN 323, taught by Professor Mann.
	
	Currently, I am still in development, so you should carefully check my answers against what you have learned in class.
	However, I will do my very best to help answer your questions, so ask away!
	
	At the left you can see a dropdown with a list of specific documents from the course webpage, including all our slides from class as well as past exams.
	If you know that your question is specific to one of these, then select it and my answer will be much more helpful!
	For example, you can select a past exam and then ask for help understanding a specific question.
	
	Other ideas would be to ask for step-by-step help with any calculations that we did in class, or reviewing background terms that may not be familiar.
	
	You can also speak to me in any language, and ask for translations of any course material into different languages!
	"""

def format_latex(text):
	text = re.sub(r'\\\(',r'$$',text)
	text = re.sub(r'\\\)',r'$$',text)
	text = re.sub(r'\\\[',r'$$',text)
	text = re.sub(r'\\\]',r'$$',text)
	text = re.sub(r'\\\\\$',r'\\\$',text)
	return text

def generate_tokens(json_payload):
	response = requests.post("http://localhost:5000/get_response", json=json_payload, stream=True)
	for line in response.iter_lines():
		if line:
			line_str = line.decode('utf-8')
			token_data = json.loads( line_str )
			token = token_data['token']
			yield token
	
# Then start the Streamlit interface.
# Wrap this within main() so it does not execute when this file is imported to other code, just in case someone ever wants to do that.
def main():

	# Use the full width of the page
	st.set_page_config(page_title="FIN 323 Virtual TA",layout="wide",page_icon="Emory.ico")
	
	# Remove various Streamlit default page decorations:
	st.markdown("""
		<style>
		div[data-testid="stToolbar"] {
		visibility: hidden;
		height: 0%;
		position: fixed;
		}
		div[data-testid="stDecoration"] {
		visibility: hidden;
		height: 0%;
		position: fixed;
		}
		div[data-testid="stStatusWidget"] {
		visibility: hidden;
		height: 0%;
		position: fixed;
		}
		#MainMenu {
		visibility: hidden;
		height: 0%;
		}
		header {
		visibility: hidden;
		height: 0%;
		}
		footer {
		visibility: hidden;
		height: 0%;
		}
		</style>
		""",unsafe_allow_html=True)
	
	# Set up the side bar with Emory icon and dropdown menu
	st.sidebar.image("Emory.png")
	st.sidebar.text("")
	st.sidebar.text("")
	st.sidebar.text("")
	# Times New Roman is the closest websafe font I can find to match the Emory logo
	st.sidebar.markdown('<p style="color:#0033a0; font-family:Times New Roman, serif; font-size: 24px">FIN 323 Virtual TA</p>', unsafe_allow_html=True)
	st.sidebar.text("")
	options = [no_selection_text] + list(document_file_paths.keys())
	document_choice = st.sidebar.selectbox("Select a specific course topic or document? (you can type to search)", options )
	
	# Initialize chat history in session state if not already present:
	if "chat_history" not in st.session_state:
		st.session_state.chat_history = [{"role":"assistant","content":welcome_message}]
	# Truncate chat history to last 3 messages.
	if len(st.session_state.chat_history) > 3:
		st.session_state.chat_history = st.session_state.chat_history[-3:]
	# Print chat history to screen.
	for message in st.session_state.chat_history:
		if message['role'] == "user":
			with st.chat_message("user",avatar="💬"):
				st.markdown(message['content'])
		elif message['role'] == "assistant":
			with st.chat_message("bot",avatar="✨"):
				st.markdown(message['content'])
	
	# Accept and print user query, retrieve and print LLM response via API
	if query := st.chat_input("Enter your question for the virtual TA"):
		# display user query
		query_edited = format_latex( query )
		with st.chat_message("user",avatar="💬"): st.markdown(query_edited)
		# get response from AI. To do: Better error handling here.
		json_payload = {'query':query,'document_choice':document_choice,'chat_history_messages':st.session_state.chat_history}
		api_response = generate_tokens(json_payload)
		response_message = st.chat_message("bot",avatar="✨") 
		response_placeholder = response_message.empty()
		response_text_raw = ""
		for token in api_response:
			response_text_raw += token
			response_text_edited = format_latex(response_text_raw)
			response_placeholder.markdown(response_text_edited)
		st.session_state.chat_history.append({"role":"user","content":query_edited})
		st.session_state.chat_history.append({"role":"assistant","content":response_text_edited})

if __name__ == '__main__':
	main()

