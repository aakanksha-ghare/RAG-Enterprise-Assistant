
import streamlit as st

from src.rag_pipeline import ask_question
from src.retriever import retrieve_documents


st.set_page_config(
    page_title="RAG Enterprise Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Enterprise Assistant")
st.caption(
    "Ask questions about your documents. "
    "Answers are generated locally using Ollama."
)

# Keep the conversation available across Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Start a fresh conversation when requested.
with st.sidebar:
    st.subheader("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.info(
            f"{len(uploaded_files)} PDF file(s) selected. "
            "Upload processing will be added next."
        )

    st.divider()
    st.subheader("Conversation")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show saved retrieved passages for assistant responses.
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Inspect retrieved passages"):
                for index, source in enumerate(
                    message["sources"], start=1
                ):
                    st.markdown(
                        f'**Passage {index}** — '
                        f'`{source["filename"]}`, '
                        f'page {source["page"]}'
                    )
                    st.caption(
                        f'Vector distance: {source["distance"]:.4f}'
                    )
                    st.write(source["content"])

# Accept a new question.
question = st.chat_input("Ask a question about your documents...")

if question:
    # Display and save the user's message.
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate and save the assistant's answer.
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                answer = ask_question(question)
                results = retrieve_documents(question, k=4)

                sources = []

                for document, distance in results:
                    page = document.metadata.get("page")
                    page_number = (
                        page + 1 if isinstance(page, int) else "Unknown"
                    )

                    sources.append(
                        {
                            "filename": document.metadata.get(
                                "source", "Unknown"
                            ),
                            "page": page_number,
                            "distance": float(distance),
                            "content": document.page_content,
                        }
                    )

                st.markdown(answer)

                with st.expander("Inspect retrieved passages"):
                    for index, source in enumerate(sources, start=1):
                        st.markdown(
                            f'**Passage {index}** — '
                            f'`{source["filename"]}`, '
                            f'page {source["page"]}'
                        )
                        st.caption(
                            f'Vector distance: {source["distance"]:.4f}'
                        )
                        st.write(source["content"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as error:
                error_message = f"Something went wrong: {error}"
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
