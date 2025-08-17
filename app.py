import streamlit as st
import tempfile
from main import (
    get_transcript, summarize, build_vector_db, qa_agent, generate_pdf,
    send_email, extract_youtube_id, text_to_audio, general_chat_agent
)
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError

st.set_page_config(page_title="YouTube Chat + General Chat", layout="wide")

# ----------------- SIDEBAR: Keys -----------------
st.sidebar.title("API Keys Setup")
google_api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
huggingface_api_token = st.sidebar.text_input("HuggingFace API Token", type="password")

st.sidebar.markdown(
    """
    <div style="background-color:#d4edda; padding:8px; border-radius:6px; font-size:12px; color:#155724; margin-top:4px;">
        Press <b>Enter</b> after typing your API keys to confirm.
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------- Session State -----------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "conversation_obj" not in st.session_state:
    st.session_state.conversation_obj = None
if "session_id" not in st.session_state:
    st.session_state.session_id = "yt_session"

if "gen_conversation" not in st.session_state:
    st.session_state.gen_conversation = []
if "gen_agent" not in st.session_state:
    st.session_state.gen_agent = None
if "gen_session_id" not in st.session_state:
    st.session_state.gen_session_id = "general_session"

# ----------------- Prepare agents -----------------
if google_api_key and huggingface_api_token:
    if st.session_state.gen_agent is None:
        try:
            st.session_state.gen_agent = general_chat_agent(google_api_key)
        except (ResourceExhausted, GoogleAPICallError) as e:
            st.error(f"❌ Gemini error while starting general agent: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

    # ---- YouTube Setup Section (always before page selection) ----
    st.sidebar.title("YouTube ChatBot")
    video_link = st.sidebar.text_input("YouTube Video Link")
    st.sidebar.markdown(
        """
        <div style="background-color:#d4edda; padding:8px; border-radius:6px; font-size:12px; color:#155724; margin-top:4px;">
            ✅ Only provide YouTube videos with an <b>English transcript</b>.
        </div>
        <div style="height:20px;"></div>  <!-- Spacer between disclaimer and button -->
        """,
        unsafe_allow_html=True
    )

    video_id = extract_youtube_id(video_link)

    if st.sidebar.button("Submit Video ID"):
        if video_id:
            transcript_text = get_transcript(video_id)

            if transcript_text:  # ✅ Only proceed if transcript fetched
                try:
                    summary_content = summarize(transcript_text, google_api_key)
                    st.session_state.summary = summary_content

                    vectordb = build_vector_db(transcript_text, huggingface_api_token)
                    conversation, retriever = qa_agent(vectordb, google_api_key)

                    st.session_state.vectordb = vectordb
                    st.session_state.conversation_obj = conversation
                    st.session_state.retriever = retriever
                    st.session_state.conversation = []
                    st.success("✅ Video loaded successfully! Summary generated.")
                except (ResourceExhausted, GoogleAPICallError) as e:
                    st.error(f"❌ Gemini quota/rate-limit error: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error while summarizing or creating QA agent: {e}")
            else:
                st.error("❌ Could not fetch transcript. Please try another video.")
        else:
            st.error("❌ Please enter a valid YouTube Video link/ID.")

    # ---- Sidebar menu (AFTER video link section) ----
    page = st.sidebar.radio("📂 Choose Chat Page:", ["YouTube ChatBot", "General ChatBot"], key="main_page")
else:
    st.sidebar.info("🔑 Please enter both API keys to continue.")

# ----------------- PDF + Logout -----------------
if st.session_state.summary:
    email = st.sidebar.text_input("Email for PDF")
    if st.sidebar.button("Send Summary & Chat as PDF"):
        if email:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                pdf_file = generate_pdf(
                    st.session_state.summary,
                    st.session_state.conversation_obj.get_session_history(st.session_state.session_id),
                    general_history=st.session_state.gen_agent.get_session_history(st.session_state.gen_session_id),
                    filename=tmp.name
                )
                success = send_email(email, pdf_file)
                if success:
                    st.success(f"✅ PDF sent to {email}")
                else:
                    st.error(f"❌ Failed to send PDF to {email}")
        else:
            st.error("❌ Please enter an email to send PDF.")

    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ Logged out successfully!")
        st.rerun()
        
    st.sidebar.markdown(
        """
        <div style="background-color:#fff3cd; padding:8px; border-radius:6px; font-size:12px; color:#856404; margin-top:8px;">
            ℹ️ To start a new chat with a different video, please use the <b>Logout</b> button first.
        </div>
        """,
        unsafe_allow_html=True
    )
    # ----------------- PAGE RENDER -----------------
if google_api_key and huggingface_api_token:
    st.markdown(
        "<h1 style='text-align:center; color:#1e3a8a; font-size:48px; font-weight:bold;'>🎥 YouTube AI Chat & 💬 General AI Assistant</h1>",
        unsafe_allow_html=True
    )

    if page == "YouTube ChatBot":
        st.subheader("🎬 AI Video Summary & Contextual Q/A")
        if st.session_state.summary:
            st.markdown(
                f"""
                <div style='background-color:#ffffff; color:#000; border:1px solid #000; padding:15px; border-radius:8px; margin:10px 0;'>
                    {st.session_state.summary}
                </div>
                """,
                unsafe_allow_html=True
            )

            chat_container = st.container()
            with chat_container:
                for chat in st.session_state.conversation:
                    if chat["role"] == "user":
                        st.markdown(
                            f"""
                            <div style='background-color:#f0f0f0; color:#000; padding:10px; border-radius:8px; margin:5px 0;'>
                                <b>You:</b> {chat['content']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        col1, col2 = st.columns([0.9, 0.1])
                        with col1:
                            st.markdown(
                                f"""
                                <div style='background-color:#ffffff; color:#000; border:1px solid #000; padding:10px; border-radius:8px; margin:5px 0;'>
                                    <b>AI:</b> {chat['content']}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        with col2:
                            if st.button("🔊", key=f"tts_right_{chat['id']}"):
                                audio_file = text_to_audio(chat['content'])
                                st.audio(audio_file)

            yt_input = st.chat_input("Ask about the video only…", key="yt_chat")
            if yt_input:
                st.session_state.conversation.append({
                    "role": "user",
                    "content": yt_input,
                    "id": len(st.session_state.conversation)
                })
                try:
                    docs = st.session_state.retriever.invoke(yt_input)
                    context = "\n".join([d.page_content for d in docs])
                    result = st.session_state.conversation_obj.invoke(
                        {"input": f"Context: {context}\nQuestion: {yt_input}"},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                    answer = result.content
                except (ResourceExhausted, GoogleAPICallError) as e:
                    answer = f"❌ Gemini quota/rate-limit error: {e}"
                except Exception as e:
                    answer = f"❌ Unexpected error: {e}"

                st.session_state.conversation.append({
                    "role": "ai",
                    "content": answer,
                    "id": len(st.session_state.conversation)
                })
                st.rerun()
        else:
            st.info("Upload a YouTube link in the sidebar to load transcript summary & Q/A.")

    elif page == "General ChatBot":
        st.subheader("💬 Smart Assistant")
        if st.session_state.gen_agent:
            gen_chat_container = st.container()
            with gen_chat_container:
                for chat in st.session_state.gen_conversation:
                    if chat["role"] == "user":
                        st.markdown(
                            f"""
                            <div style='background-color:#f0f0f0; color:#000; padding:10px; border-radius:8px; margin:5px 0;'>
                                <b>You:</b> {chat['content']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        col1, col2 = st.columns([0.95, 0.05])
                        with col1:
                            st.markdown(
                                f"""
                                <div style='background-color:#ffffff; color:#000; border:1px solid #000; padding:10px; border-radius:8px; margin:5px 0;'>
                                    <b>AI:</b> {chat['content']}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        with col2:
                            if st.button("🔊", key=f"tts_right_{chat['id']}"):
                                audio_file = text_to_audio(chat['content'])
                                st.audio(audio_file)

            gen_input = st.chat_input("Ask me anything…", key="general_chat")
            if gen_input:
                st.session_state.gen_conversation.append({
                    "role": "user",
                    "content": gen_input,
                    "id": len(st.session_state.gen_conversation)
                })
                try:
                    result = st.session_state.gen_agent.invoke(
                        {"input": gen_input},
                        config={"configurable": {"session_id": st.session_state.gen_session_id}}
                    )
                    gen_answer = result.content
                except (ResourceExhausted, GoogleAPICallError) as e:
                    gen_answer = f"❌ Gemini quota/rate-limit error: {e}"
                except Exception as e:
                    gen_answer = f"❌ Unexpected error: {e}"

                st.session_state.gen_conversation.append({
                    "role": "ai",
                    "content": gen_answer,
                    "id": len(st.session_state.gen_conversation)
                })
                st.rerun()
        else:
            st.info("Enter your Hugging Face token in the sidebar to enable General Chat.")