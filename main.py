import pickle
import streamlit as st
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing import preprocessing


class FakeNewsDetection:
    def __init__(self):
        st.title("Fake news detection")
        st.caption("In this day and age, fake news is having a very serious impact on our human life, through this research, I would like everyone to join hands to build a better life and make a small contribution to it. Fight against fake news.")
        st.caption("Trong thời đại ngày nay, tin giả đang gây ảnh hưởng rất nghiêm trọng đến đời sống con người chúng ta, thông qua nghiên cứu này, tôi muốn mọi người hãy chung tay xây dựng một cuộc sống tốt đẹp hơn và góp phần nhỏ trong cuộc chiến chống tin giả.")
        self.input = st.text_area(label="Your input")

    def run_check(self):
        def load_model():
            self.model = tensorflow.keras.models.load_model("model.h5")

        def load_tokenizer():
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)

        if 'input' not in st.session_state:
            st.session_state.input = preprocessing(self.input)

        def get_data():
            st.session_state.input = preprocessing(self.input)

        click = st.button(label="Check", on_click=get_data)
        if click:
            load_model()
            load_tokenizer()
            st.write(st.session_state.input)
            st.session_state.input = [st.session_state.input]

            st.session_state.input = self.tokenizer.texts_to_sequences(
                st.session_state.input)

            st.session_state.input = pad_sequences(
                st.session_state.input, maxlen=256, padding='post', truncating='post')

            pred = self.model.predict(st.session_state.input)

            st.write(pred.item())


object = FakeNewsDetection()
object.run_check()
