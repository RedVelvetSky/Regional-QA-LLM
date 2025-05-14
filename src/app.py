import streamlit as st

from registry import answering_methods, evaluation_methods, retrieval_methods

ss = st.session_state

# --- Streamlit primitives ---


def get_question():
    return st.text_input(
        "Question",
        value=ss.get("question", ""),
    )


def get_context():
    return st.text_area(
        label="Context",
        height=250,
        value=ss.get("context", ""),
    )


def get_pred_answer():
    return st.text_area(
        label="Generated Answer",
        value=ss.get("pred_answer", ""),
    )


def get_true_answer(optional=False):
    return st.text_area(
        label="True Answer (optional)" if optional else "True Answer",
        value=ss.get("true_answer", ""),
    )


# --- Streamlit pages ---


def page_intro():
    with open("README.md", "r") as f:
        readme = f.read()
        st.markdown(readme)
    st.sidebar.success("Select a pipeline above.")


def page_retrieval():
    st.write("# Context Retrieval")
    method = st.sidebar.selectbox("Select retrieval method:", retrieval_methods.keys())
    retrieve_fn = retrieval_methods[method]

    # get input question
    ss["question"] = get_question()
    button = st.button("Submit")
    if not button:
        st.stop()

    # retrieve context
    st.write("**Retrieved context**")
    ss["context"] = retrieve_fn(ss["question"])
    with st.container(border=True):
        st.write(ss["context"])


def page_answering():
    st.write("# Question Answering")
    method = st.sidebar.selectbox("Select answering method:", answering_methods.keys())
    answering_fn = answering_methods[method]

    # get input question
    ss["question"] = get_question()
    ss["context"] = get_context()
    button = st.button("Submit")
    if not button:
        st.stop()

    # answer question
    st.write("**Answer**")
    ss["pred_answer"] = answering_fn(ss["question"], ss["context"])
    with st.container(border=True):
        st.write(ss["pred_answer"])


def page_evaluation():
    st.write("# Evaluation")
    method = st.sidebar.selectbox("Select evaluation method:", evaluation_methods.keys())
    eval_fn = evaluation_methods[method]

    # get input question
    ss["question"] = get_question()
    ss["true_answer"] = get_true_answer()
    ss["pred_answer"] = get_pred_answer()
    button = st.button("Submit")
    if not button:
        st.stop()

    # answer question
    st.write("**Evaluation**")
    ss["eval"] = eval_fn(ss["question"], ss["true_answer"], ss["pred_answer"])
    with st.container(border=True):
        st.write(ss["eval"])


def page_e2e():
    st.write("# End-to-End")
    retrieve_fn = retrieval_methods[st.sidebar.selectbox("Select retrieval method:", retrieval_methods.keys())]
    answering_fn = answering_methods[st.sidebar.selectbox("Select answering method:", answering_methods.keys())]
    eval_fn = evaluation_methods[st.sidebar.selectbox("Select evaluation method:", evaluation_methods.keys())]

    # get input question
    question = get_question()
    true_answer = get_true_answer(optional=True)
    button = st.button("Submit")
    if not button:
        st.stop()

    context = retrieve_fn(question)
    with st.expander("Retrieved context", expanded=False):
        st.write(context)

    pred_answer = answering_fn(question, context)
    with st.expander("Answer", expanded=True):
        st.write(pred_answer)

    if true_answer:
        evaluation = eval_fn(question, true_answer, pred_answer)
        with st.expander("Evaluation", expanded=False):
            st.write(evaluation)


page_names_to_funcs = {
    "README": page_intro,
    "**1.** Context Retrieval": page_retrieval,
    "**2.** Question Answering": page_answering,
    "**3.** Answer Evaluation": page_evaluation,
    "End-To-End": page_e2e,
}

page_name = st.sidebar.radio("Choose a pipeline", page_names_to_funcs.keys())
st.sidebar.divider()
page_names_to_funcs[page_name]()
