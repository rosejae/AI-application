import streamlit as st
import numpy as np

with st.chat_message('user'):
    st.write('hello')

# with st.chat_message('assistant'):
#     st.write('hello human')
#     st.bar_chart(np.random.randn(30, 3))

message = st.chat_message('assistant')
message.write('hello human')
message.bar_chart(np.random.randn(30, 3))

prompt = st.chat_input('say something')
if prompt:
    st.write(f'user has sent the following prompt: {prompt}')