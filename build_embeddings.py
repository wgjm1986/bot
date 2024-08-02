import sys, os, openai, sqlite3, json

import numpy as np

from glob import glob
from multiprocessing import Pool
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from text_tools import get_document_paragraphs, chunk_paragraphs

client = openai.OpenAI()

# Import course settings based on parameter passed
with open('courses.json','r') as courses_file: 
    course_data = json.load(courses_file).get( sys.argv[1] ,{})
if not course_data: raise ValueError(f"No course data found for {sys.argv[1]}")

db_path = course_data['db_file']
db_temp_path = db_path + '.tmp'
db_folder = course_data['db_folder']

if os.path.exists(db_temp_path):
        os.remove(db_temp_path)
        print(f"Deleted existing database file: {db_temp_path}")

extensions = ["pdf","tex","docx","pptx","ipynb"]
filenames = [filename for ext in extensions for filename in glob(f"{db_folder}/**/*.{ext}",recursive=True)]

print("Num files: " + str(len(filenames)))

def get_slide_text(slide):
    slide_text_chunks = []
    for shape in slide.shapes:
        if hasattr(shape,"text"):
            slide_text_chunks.append(shape.text+' ')
    return '\n'.join(slide_text_chunks)

conn = sqlite3.connect(db_temp_path)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        doc_id INTEGER PRIMARY KEY,
        file_path TEXT,
        description TEXT
    )
''')
conn.commit()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        doc_id INTEGER,
        chunk_text TEXT,
        embedding BLOB,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
    )
''')
conn.commit()
conn.close()


print(f"Importing documents: {datetime.now():%H:%M:%S}")

def process_file(filename):
    # This is not only to save time, but also to cut down on token usage
    if os.path.getsize(filename) > 1e6:
        print(filename+": above 1M, skipping")
        return
    document_paragraphs = get_document_paragraphs(filename,10) # second argument is the max number of pages to read from each PDF
    document_text = '\n\n'.join(document_paragraphs)
    if not document_text:
        print(filename+"get_document_paragraphs returned null, skipping")
        return
    if type(document_text) is not str:
        print(filename+"get_document_paragraphs returned non-string, skipping")
        return
    chunks = chunk_paragraphs(document_paragraphs)
    # Get document description
    description_prompt = f"I am indexing documents for a college course on {course_data['topic']}."
    description_prompt += """
    If the document below is irrelevant to that topic, or does not have enough content to be worth using in the course, reply with "Irrelevant".
    Otherwise, please reply with a short description of the document (30 words or fewer). 
    Your description does not need to be a complete sentence. 
    It should consist only of ASCII characters with no tabs, newlines, or form feeds. 
    Be sure to mention any authors, and the year of publication, is you can find them. 
    If the document is an exam, specify the semester, and which exam it was (first midterm, second midterm, final exam, etc.). 
    Finally, list 5 or fewer keywords for the document's content.
    Your entire response should be one line.
    """
    client = openai.OpenAI()
    if len(document_text) > 5000: document_text = document_text[0:5000]
    description_messages = [{"role":"system","content":description_prompt} , {"role":"user","content":document_text}]
    description = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=description_messages,
        max_tokens=200,
        stream=False
    ).choices[0].message.content
    # print(filename)
    # print(description)
    # Add this file to the table of files
    embedding_bytes_list = []
    for chunk in chunks:
        # Add each chunk and embeddings into the table of chunks
        embedding = openai.embeddings.create(model="text-embedding-ada-002",input=chunk).data[0].embedding
        embedding_bytes = np.array( embedding ).tobytes()
        embedding_bytes_list.append( embedding_bytes )
    # Write to database: save this for the end to keep the lock as brief as possible
    conn = sqlite3.connect(db_temp_path,timeout=10)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO documents (file_path, description)
        VALUES (?, ?)
    ''', (filename, description))
    doc_id = cursor.lastrowid
    for chunk, embedding_bytes in zip(chunks, embedding_bytes_list):
        cursor.execute('''
            INSERT INTO chunks (doc_id, chunk_text, embedding) VALUES (?, ?, ?)
        ''', (doc_id, chunk, embedding_bytes) )
    conn.commit()
    conn.close()
    print("Complete: " + filename)

# Note that exceptions thrown within this pool will NOT propagate back to the main process.
# The filename that caused the error will just be missing from the final output.
# To catch and display these errors requires much more complicated syntax.
# What I should really do is handle and log exceptions within process_file().
with ThreadPoolExecutor(max_workers=16) as executor:
    executor.map(process_file, filenames)

print(f"Finished importing documents: {datetime.now():%H:%M:%S}")

## Now that the database construction has ended successfully, overwrite the existing one (if present)
os.rename(db_temp_path,db_path)
