import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Task Manager", page_icon="📋", layout="wide")

# --- INITIALIZE STATE ---
if "tasks" not in st.session_state:
    st.session_state.tasks = []

# --- MAIN APP ---
st.title("Task Management APP")
st.markdown("---")

# Sidebar for navigation
menu = st.sidebar.radio("Choose an action:", ["View Tasks", "Add Task", "Update Task", "Delete Task"])

# --- VIEW TASKS ---
if menu == "View Tasks":
    st.header("View Tasks")
    if st.session_state.tasks:
        for i, task in enumerate(st.session_state.tasks, 1):
            st.write(f"**{i}. {task}**")
            
    else:
        st.info("No tasks yet. Add one!")

# --- ADD TASK ---
elif menu == "Add Task":
    st.header("Add Task")
    task = st.text_input("Enter task:", placeholder="e.g., Complete assignment")
    if st.button("Add"):
        if task.strip():
            st.session_state.tasks.append(task.strip())
            st.success(f"Added: {task}")
            st.rerun()
        else:
            st.warning("Please enter a valid task!")

# --- UPDATE TASK ---
elif menu == "Update Task":
    st.header("Update Task")
    if st.session_state.tasks:
        selected = st.selectbox("Select a task to update:", st.session_state.tasks)
        new_text = st.text_input("New task name:", value=selected)
        if st.button("Update"):
            idx = st.session_state.tasks.index(selected)
            st.session_state.tasks[idx] = new_text.strip()
            st.success("Task updated!")
            st.rerun()
    else:
        st.info("No tasks to update!")

# --- DELETE TASK ---
elif menu == "Delete Task":
    st.header("Delete Task")
    if st.session_state.tasks:
        selected = st.selectbox("Select a task to delete:", st.session_state.tasks)
        if st.button("Delete"):
            st.session_state.tasks.remove(selected)
            st.success("Task deleted!")
            st.rerun()
    else:
        st.info("No tasks to delete!")

# --- QUICK STATS ---
st.sidebar.markdown("---")
st.sidebar.metric("Total Tasks", len(st.session_state.tasks))