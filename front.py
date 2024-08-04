import streamlit as st
import requests
import json
import re
import sys

# Function to clean up markdown delimiters that OpenAI uses but Streamlit doesn't understand.
def format_latex(text):
    text = re.sub(r'\\\(',r'$$',text)
    text = re.sub(r'\\\)',r'$$',text)
    text = re.sub(r'\\\[',r'$$',text)
    text = re.sub(r'\\\]',r'$$',text)
    text = re.sub(r'\\\\\$',r'\\\$',text)
    return text

# Generator to receive individual tokens from a stream as JSON, unpack them and yield just the text
def generate_tokens(json_payload,port_number):
    response = requests.post(f"http://localhost:{port_number}/get_response", json=json_payload, stream=True)
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
    st.set_page_config(page_title="Virtual TA",layout="wide",page_icon="Emory.ico")
    
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
    st.sidebar.text("")
    st.sidebar.text("")
    st.sidebar.text("")
    st.sidebar.image("Emory.png",width=190)
    st.sidebar.text("")
    st.sidebar.text("")
    # Times New Roman is the closest websafe font I can find to match the Emory logo
    # course_number = course_data['number']
    # st.sidebar.markdown(f'<p style="text-align: center; color:#0033a0; font-family:Times New Roman, serif; font-size: 24px">{course_number} Virtual TA</p>', unsafe_allow_html=True)

    # Load dictionary of course info
    with open('courses.json','r') as courses_file: course_dict = json.load(courses_file)
    # Create reverse lookup dictionary
    course_dict_reverse = { course['number']+": "+course['title'] : key for key,course in course_dict.items() }

    def reset_chat_history():
        if st.session_state.course_selection_description == "No selection made":
            welcome_message = "Welcome! To get started, please select a course from the dropdown menu to the left."
            st.session_state.chat_history = [{"role":"assistant","content":welcome_message}]
        else:
            course_selection_key = course_dict_reverse[st.session_state.course_selection_description]
            course_selection = course_dict[course_selection_key]
            welcome_message = f"""
            Hello, human! I am the virtual TA for the course {course_selection['number']}: {course_selection['title']}, taught by {course_selection['instructor']}.
            Currently, I am still in development, so you should carefully check my answers against what you have learned in class.
            However, I will do my very best to help answer your questions, so ask away!
            
            For example, you can ask for an explanation of questions from past exams, material from class, or reviewing terms that may not be familiar.
            You can also speak to me in any language, and ask for translations of any course material into different languages!
            """
            st.session_state.chat_history = [{"role":"assistant","content":welcome_message}]

    with st.sidebar:
        st.selectbox(
            "Choose your course:",
            ["No selection made"] + list( course_dict_reverse.keys()),
            key='course_selection_description',
            on_change=reset_chat_history
        )

    # Initialize course selection if it exists:
    if 'course_selection_description' not in st.session_state:
        st.session_state.course_selection_description = "No selection made"
        welcome_message = "Welcome! To get started, please select a course from the dropdown menu to the left."
        st.session_state.chat_history = [{"role":"assistant","content":welcome_message}]
    elif st.session_state.course_selection_description == "No selection made":
        welcome_message = "Welcome! To get started, please select a course from the dropdown menu to the left."
        st.session_state.chat_history = [{"role":"assistant","content":welcome_message}]
    else:
        course_selection_key = course_dict_reverse[st.session_state.course_selection_description]
        course_selection = course_dict[course_selection_key]


    # Truncate chat history to last 6 messages (three each from the student and the assistant).
    if len(st.session_state.chat_history) > 6:
        st.session_state.chat_history = st.session_state.chat_history[-6:]

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
        # Display user query
        query_edited = format_latex( query )
        with st.chat_message("user",avatar="💬"): st.markdown(query_edited)
        # Retrieve and stream the LLM response from the API
        # Only hand off the last four messages in the chat history (two for the user, two for the bot).
        # The use of placeholder and empty() are tricks to be able to render markup while streaming:
        # as each new token arrives, we replace and re-render the entire message up to this point,
        # so that any closing delimiters are correctly paired with opening delimiters when they arrive,
        # and the raw text printed up to this point is replaced with correctly rendered markdown.
        json_payload = {'query':query,'chat_history_messages':st.session_state.chat_history[-2:]}
        response_message = st.chat_message("bot",avatar="✨") 
        response_placeholder = response_message.empty()
        response_text_raw = ""
        api_response = generate_tokens(json_payload,course_selection['api_port'])
        with response_placeholder, st.spinner("Thinking..."):
            for token in api_response:
                response_text_raw += token
                response_text_edited = format_latex(response_text_raw)
                response_placeholder.markdown(response_text_edited)
        response_placeholder.markdown(response_text_edited)
        # Add both messages to conversation history
        st.session_state.chat_history.append({"role":"user","content":query_edited})
        st.session_state.chat_history.append({"role":"assistant","content":response_text_edited})

if __name__ == '__main__':
    main()

