
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import time
import google.generativeai as genai
from IPython.display import Markdown
import time
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)


load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_DIR = os.getenv("FILES_DIR", "/tmp/files")
os.makedirs(BASE_DIR, exist_ok=True)


report_text_path = os.path.join(BASE_DIR, "final_report.txt")
pdf_path    = os.path.join(BASE_DIR, "report.pdf")


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

genai.configure(api_key=GOOGLE_API_KEY)



@app.route("/",methods=["GET"])
def home():
    return "Welcome to Suthradhar!"

@app.route("/get-report/", methods=["POST"])
def get_report():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        file.save(tmp_file)
        temp_video_path = tmp_file.name

    try:
        gemini_file = upload_video(temp_video_path)
        report_text = process_video_with_gemini(gemini_file)

        if report_text:
            with open(report_text_path, "w") as f:
                f.write(report_text)
            print(f"Final report saved to {report_text_path}")
            generate_pdf_from_text_file()
            
            return send_file(
                   pdf_path,
                as_attachment=True,
                download_name="report.pdf",
                mimetype="application/pdf"
            )
        else:
            return jsonify({"error": "Failed to generate report"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to generate report: {str(e)}"}), 500



@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_data = request.json
        user_query = user_data.get('query')

        if not user_query:
            return jsonify({"error": "Query is required"}), 400

        response_text = get_conversational_answer(user_question=user_query)

        print(response_text)
        return jsonify({ "response": response_text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def upload_video(file_path):
    print(f"Uploading file: {file_path}")
    video_file = genai.upload_file(path=file_path)
    print(f"Completed upload: {video_file.uri}")
    return video_file

def process_video_with_gemini(video_file):
    while video_file.state.name == "PROCESSING":
        print('.', end='')
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(f"File processing failed: {video_file.state.name}")

    prompt = """
    You are analyzing a full video. Provide the following details:

    1. **Incident Overview**: Describe what happened, environmental conditions (e.g., lighting, weather).
    2. **Suspects**: Provide details on their appearance, actions, and any distinguishing features.
    3. **Victims/Witnesses**: If visible, describe them, including their actions.
    4. **Vehicles**: Describe any vehicles, license plates, and their interaction with the crime scene.
    5. **Affected Items**: Describe items impacted during the incident (e.g., ATM, its condition before and after).
    6. **Suspicious Activities**: Identify illegal activities and summarize the sequence of events.
    7. **Conversations/Sounds**: Analyze any audible conversations or sounds relevant to the crime scene.
    8. **Additional Details**: Mention any visible landmarks, signboards, or other context about the scene.

    Summarize all information and generate a comprehensive report for the police investigation.
    """

    model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    print("Making LLM inference request...")
    
    try:
        response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
    except TypeError as e:
        print(f"TypeError encountered: {e}")
        return None

    return response.text

def generate_pdf_from_text_file():
    txt_file_path = report_text_path
    pdf_output_path = pdf_path
    try:
        with open(txt_file_path, 'r') as file:
            text_content = file.read()
        
        document = SimpleDocTemplate(pdf_output_path, pagesize=letter)
        story = []

        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading1']
        normal_style = styles['BodyText']

        lines = text_content.split("\n")
        for line in lines:
            line = line.strip()
            if not line: 
                story.append(Spacer(1, 12))
                continue

            if line.startswith("## "):  
                story.append(Paragraph(line.replace("## ", ""), heading_style))
                story.append(Spacer(1, 12))
            else:
                story.append(Paragraph(line, normal_style))
                story.append(Spacer(1, 12))

        document.build(story)
        print(f"PDF successfully saved at {pdf_output_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")




def get_conversational_answer(
    user_question: str,
    session_id: str = "default_session",
    groq_api_key: str = GROQ_API_KEY
) -> str:
    if not groq_api_key:
        raise ValueError("Groq API key is required.")

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

    pdf_paths=[ pdf_path]
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question "
                   "which might reference context in the chat history, "
                   "formulate a standalone question. Do NOT answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. "
                   "Use the following pieces of retrieved context to answer the question. "
                   "If unknown, say you don't know. Answer concisely.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    history_store = {}

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in history_store:
            history_store[session] = ChatMessageHistory()
        return history_store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    response = conversational_rag_chain.invoke(
        {"input": user_question},
        config={"configurable": {"session_id": session_id}}
    )
    return response["answer"]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port,debug=True ,use_reloader=False)
