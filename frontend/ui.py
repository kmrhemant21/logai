import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.title("Chat with Your Logs")

query = st.text_input("Enter your log query:")
top_k = st.slider("Number of results", 1, 20, 5)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            response = requests.post(API_URL, json={"query": query, "top_k": top_k})
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    for res in results:
                        st.markdown(f"**Timestamp:** {res['timestamp']}")
                        st.code(res['log'])
                        st.write(f"Similarity Score: {res['score']:.2f}")
                        st.markdown("---")
                else:
                    st.info("No similar logs found.")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")