import os
import re
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pyttsx3
import tempfile

# For PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# =====================
# CONFIG
# =====================
load_dotenv()


def init_llm(api_key: str):
    if not api_key:
        raise ValueError("Google API Key is required")
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)


def init_embeddings(hf_api_token: str):
    if not hf_api_token:
        raise ValueError("HuggingFace API Token is required")
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=hf_api_token
    )

# =====================
# TEXT TO SPEECH
# =====================

def text_to_audio(text):

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        filename = f.name
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename

# ============================
# YouTube video ID Extractor
# ============================

def extract_youtube_id(url):
    """
    Extract YouTube video ID from URL.
    Returns the video ID if found, else None.
    """
    # Regex pattern to match YouTube video IDs in different URL formats
    pattern = (
        r'(?:https?://)?'               
        r'(?:www\.)?'                   
        r'(?:youtube\.com/watch\?v=|'   
        r'youtu\.be/|'                  
        r'youtube\.com/embed/)'         
        r'([A-Za-z0-9_-]{11})'          
    )
    
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

# =====================
# GET YOUTUBE TRANSCRIPT
# =====================
def get_transcript(video_id):
    try:
        YouTube_api = YouTubeTranscriptApi()
        transcript_list = YouTube_api.fetch(video_id)
        transcript = " ".join(chunk.text for chunk in transcript_list)
        return transcript
    except Exception as e:
        print(f"❌ Failed to fetch transcript: {e}")
        return None

# =====================
# BUILD VECTOR DB
# =====================
def build_vector_db(text, hf_api_token, persist=False):

    embeddings = init_embeddings(hf_api_token)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([text])

    if persist:
        vectordb = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory="db"
        )
        vectordb.persist()
    else:
        vectordb = FAISS.from_documents(chunks, embeddings)

    return vectordb

# =====================
# LLM Summarization
# =====================
def summarize(text, google_api_key):

    llm = init_llm(google_api_key)
    prompt_summary = f"Summarize the following YouTube transcript clearly:\n\n{text}"
    result = llm.invoke(prompt_summary)
    return result.content

# ==============================
# Youtube Transcript Q&A Setup
# ==============================
def qa_agent(vectordb, google_api_key):

    llm = init_llm(google_api_key)
    prompt_transcript = ChatPromptTemplate.from_messages([
        ("system",
        "You are a helpful AI assistant. You have access to the transcript of a YouTube video. "
        "Answer the user's questions **only if they are related to the transcript**. "
        "If the question is unrelated, respond with: "
        "'❌ I'm sorry, that information is not available in the transcript.'"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt_transcript | llm
    histories = {}

    def get_history(session_id: str):
        if session_id not in histories:
            histories[session_id] = ChatMessageHistory()
        return histories[session_id]

    conversation = RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    return conversation, retriever

# =====================
# PDF GENERATOR
# =====================
def generate_pdf(summary, history, general_history=None, filename="youtube_notes.pdf"):

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []

    # Title
    elements.append(Paragraph("<b>YouTube Video Summary & Conversation Notes</b>", styles['Title']))
    elements.append(Spacer(1, 20))

    # Function to clean bold and backticks
    def clean_text(text):
        text = re.sub(r"[`]+", "", text)
        text = re.sub(r"\*\*", "", text)
        text = re.sub(r":\s*\*+$", ":", text)
        text = re.sub(r"\*$", "", text)
        return text.strip()

    added_lines = set()

    def process_lines(lines, base_indent=0):
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if not line.strip():
                i += 1
                continue

            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            bullet_symbols = ["*", "-", "•", "+", "o", "▪", "■"]

            is_bullet = stripped[0] in bullet_symbols and indent <= base_indent
            text = stripped

            if is_bullet:
                bullet_symbol = "•"
                text = stripped[1:].strip()
                text = clean_text(text)

                sub_bullets = re.split(r"\s*\*\s+", text)
                for sb in sub_bullets:
                    sb_clean = sb.strip()
                    if sb_clean and sb_clean not in added_lines:
                        elements.append(Paragraph("&nbsp;" * (base_indent + indent) + f"{bullet_symbol} {sb_clean}", styles['Normal']))
                        elements.append(Spacer(1, 5))
                        added_lines.add(sb_clean)
                i += 1
            else:
                text_clean = clean_text(stripped)
                if text_clean and text_clean not in added_lines:
                    elements.append(Paragraph("&nbsp;" * (base_indent + indent) + text_clean, styles['Normal']))
                    elements.append(Spacer(1, 5))
                    added_lines.add(text_clean)
                i += 1

    # Process summary
    elements.append(Paragraph("<b>Summary:</b>", styles['Heading2']))
    summary_lines = summary.split("\n")
    process_lines(summary_lines)
    elements.append(Spacer(1, 20))

    # Process transcript-based conversation history
    elements.append(Paragraph("<b>YouTube Transcript AI Q/A Conversation</b>", styles['Heading2']))
    for msg in history.messages:
        if msg.type == "human":
            match = re.search(r"Question:(.*)", msg.content, re.DOTALL)
            user_query = match.group(1).strip() if match else msg.content.strip()
            elements.append(Paragraph(f"<b>USER:</b> {user_query}", styles['Normal']))
            elements.append(Spacer(1, 10))
        else:
            bot_answer = clean_text(msg.content)
            bot_lines = bot_answer.split("\n")
            if bot_lines:
                elements.append(Paragraph(f"<b>BOT:</b> {bot_lines[0].strip()}", styles['Normal']))
                if len(bot_lines) > 1:
                    process_lines(bot_lines[1:], base_indent=4)
            elements.append(Spacer(1, 10))

    # ✅ Process general chat history
    if general_history:
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("<b>General AI Chat Conversation</b>", styles['Heading2']))
        for msg in general_history.messages:
            if msg.type == "human":
                elements.append(Paragraph(f"<b>USER:</b> {msg.content.strip()}", styles['Normal']))
                elements.append(Spacer(1, 10))
            else:
                bot_answer = clean_text(msg.content)
                bot_lines = bot_answer.split("\n")
                if bot_lines:
                    elements.append(Paragraph(f"<b>BOT:</b> {bot_lines[0].strip()}", styles['Normal']))
                    if len(bot_lines) > 1:
                        process_lines(bot_lines[1:], base_indent=4)
                elements.append(Spacer(1, 10))

    doc.build(elements)
    return filename

# =====================
# SEND EMAIL
# =====================
def send_email(receiver_email, pdf_file):
    sender_email = os.getenv("EMAIL_USER")
    sender_pass = os.getenv("EMAIL_PASS")

    if not sender_email or not sender_pass:
        print("❌ Email credentials not set in .env")
        return False

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "📄 YouTube Video Summary & Conversation Notes"

    body = "Attached is the summary and conversation notes for your requested YouTube video."
    msg.attach(MIMEText(body, 'plain'))

    # Attach PDF
    with open(pdf_file, "rb") as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(pdf_file)}')
        msg.attach(part)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_pass)
        server.send_message(msg)
        server.quit()
        print(f"✅ Email sent successfully to {receiver_email}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

# =====================
# General Q&A Setup
# =====================

def general_chat_agent(google_api_key: str):
    llm = init_llm(google_api_key)
    prompt_general = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful, concise general-purpose assistant. "
         "Answer user questions clearly and directly."),
        MessagesPlaceholder(variable_name="general_history"),
        ("human", "{input}")
    ])

    chain = prompt_general | llm
    histories = {}

    def get_history(session_id: str):
        if session_id not in histories:
            histories[session_id] = ChatMessageHistory()
        return histories[session_id]

    conversation = RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="general_history"
    )
    return conversation