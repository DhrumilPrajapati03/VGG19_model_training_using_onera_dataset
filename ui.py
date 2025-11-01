import streamlit as st

st.title("Hello & Welcome to your task management app")
st.header("this app will help you to manage your everyday tasks")
st.caption("please enter your tasks")
st.markdown("available options")
st.subheader("1.add")
st.subheader("2.update")
st.subheader("3.delete")
st.subheader("4.view")
st.sidebar.title("Sidebar Title")
st.sidebar.markdown("This is the sidebar content")

st.set_page_comfig(page_title="Task Manager",page_icon="",layout="wide")

if "tasks"not in st.session_state:
    st.session_state.tasks=[]

st.title("Simple Task Manager")
st.markdown("---")

menu = st.sidebar.radio("Choose An Action:",["View Tasks","Add Tasks","Update Task","Delete tasks"])

if menu == "View Tasks":
    st.header("View Tasks")
    if st.session_state.tasks:
        for i, task in enumerate(st.session_state.tasks,1):
            st.write(f"**{i}.{task}**")

        if st.button("Clear All Tasks"):
            st.session_state.tasks.clear()
            st.success("All Tasks Cleared!")
            st.rerun()
    else:
        st.info("No tasks yet.Add one!")


elif menu == "Add Task":
    st.header("Add Task")
    task = st.text_input("Enter Task:",placeholder="e.g.,Complete assignment")
    if st.button("Add"):
        if task.strip():
            st.session_state.tasks.append(task.strip())
            st.success(f"Added:  {task}")
            st.rerun()
        else:
            st.warning("Please Enter A Valid Task!")


elif menu == "Update Task":
    st.header("Update Task")
    if st.session_state.tasks:
        selected = st.selectbox("Select a task to update:",st.session_state.tasks)
        new_text = st.text_input("New task name:",value=selected)
        if st.button("Update"):
            idx = st.session state.tasks.index(selected)
            st.session_state.tasks[idx] = new_text.strip()
            st.success("Task Updated!")
            st.rerun()

    else:
        st.info("No Tasks To Update!")

