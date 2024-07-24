import sys

import docx
import pptx
import re
import csv
import os

from glob import glob
from multiprocessing import Pool
from datetime import datetime
from pypdf import PdfReader
import openai
import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

client = openai.OpenAI()

extensions = ["pdf","tex","txt","docx","pptx"]
filenames = ['../syllabus.pdf'] \
	+ [filename for ext in extensions for filename in glob(f"../Module */**/*.{ext}",recursive=True)] \
	+ [filename for ext in extensions for filename in glob(f"../Discussion/**/*.{ext}",recursive=True)] \
	+ [filename for ext in extensions for filename in glob(f"../Transcripts/**/*.{ext}",recursive=True)] \
	+ [filename for ext in extensions for filename in glob(f"../Corporate finance slides/**/*.{ext}",recursive=True)]
	+ [filename for ext in extensions for filename in glob(f"../Textbook/**/*.{ext}",recursive=True)]

# extensions = ["pdf"]
# filenames = [filename for ext in extensions for filename in glob(f"../Module 1/**/*.{ext}",recursive=True)]

# extensions = ["tex"]
# filenames = [filename for ext in extensions for filename in glob(f"../Module 1/Week 4 - Sep 19 - Evidence on returns to active strategies/*.{ext}",recursive=True)]

# extensions = ["pdf"]
# filenames = [filename for ext in extensions for filename in glob(f"../Exam */**/*.{ext}",recursive=True)]

def get_slide_text(slide):
	slide_text_chunks = []
	for shape in slide.shapes:
		if hasattr(shape,"text"):
			slide_text_chunks.append(shape.text+' ')
	return '\n'.join(slide_text_chunks)

def get_document_text(file_path):
	print(file_path)
	extension = file_path[-3:]
	match extension:
		case "txt" | "tex":
			with open(file_path, 'r') as file:
				document_text = file.read()
		case "pdf":
			document_text = ""
			pdf = PdfReader(file_path)
			for page in pdf.pages:
				try:
					document_text += page.extract_text() + "\n\n"
					# document_text += page.extract_text(extraction_mode="layout") + "\n\n"
				except:
					print("PDF read failure: " + file_path)
					continue
		case "docx":
			doc = docx.Document(file_path)
			document_text = '\n\n'.join([p.text for p in doc.paragraphs])
		case "pptx":
			pres = pptx.Presentation(file_path)
			document_text = '\n\n'.join([get_slide_text(slide) for slide in pres.slides])
		case _:
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
		try:
			description = os.getxattr( file_path, "user.description" ).decode()
		except OSError as e:
			description = "none"
		metadata = { 'file_path' : file_path , 'source_type' : source_type, 'description': description }
		return { 'document_text' : document_text, 'metadata' : metadata }
	else:
		print("failure: " + file_path)
		return

print(f"Importing documents: {datetime.now():%H:%M:%S}")
with Pool(processes=7) as pool:
 	# documents,filenames = zip(*pool.map(get_document_text,filenames))
 	doc_dicts = pool.map(get_document_text,filenames)
# documents = []
# for filename in filenames:
# 	documents.append( get_document_text(filename) )
doc_dicts = [doc_dict for doc_dict in doc_dicts if doc_dict is not None]
doc_dicts = [doc_dict for doc_dict in doc_dicts if type(doc_dict['document_text']) == str]

documents = [doc_dict['document_text'] for doc_dict in doc_dicts]
metadatas = [doc_dict['metadata'] for doc_dict in doc_dicts]
print(f"Num total documents: {len(documents)}")
documents = [d for d in documents if type(d) == str]
print(f"Num good documents: {len(documents)}")

# print(f"Serializing documents: {datetime.now():%H:%M:%S}")
# with open("./serializations/documents.json",'w') as documents_output:
# 	json.dump(documents, documents_output)

print(f"Chunking text: {datetime.now():%H:%M:%S}")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, add_start_index=True)
chunks = text_splitter.split_documents( text_splitter.create_documents( texts=documents, metadatas=metadatas ) )

chunks = [chunk for chunk in chunks if re.search('[A-Za-z]',chunk.page_content)]
for chunk in chunks: chunk.page_content = re.sub('\t',' ',chunk.page_content)
for chunk in chunks: chunk.page_content = re.sub(' +',' ',chunk.page_content)
for chunk in chunks: chunk.page_content = "From file " + chunk.metadata['file_path'] + ": " + chunk.page_content

print(f"Building and serializing vector database: {datetime.now():%H:%M:%S}")
db = FAISS.from_documents(chunks, OpenAIEmbeddings() )
db.save_local("langchain_embeddings")
