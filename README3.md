## 🧩 What This Code Does

This program creates a **Task Management Web App** using **Streamlit** — a Python library that makes it easy to build web apps for data and productivity tools.

You can:

* ➕ **Add** new tasks
* ✏️ **Update** existing tasks
* ❌ **Delete** tasks
* 👀 **View** all tasks

---

## 💡 Step-by-Step Explanation

### 1. **Import Streamlit**

```python
import streamlit as st
```

This imports the Streamlit library so we can use its features (like buttons, text boxes, and sidebars).

---

### 2. **Set Page Configuration**

```python
st.set_page_config(page_title="Task Manager", page_icon="📋", layout="wide")
```

This sets up how your web app looks:

* `page_title`: The name shown in the browser tab.
* `page_icon`: The emoji 📋 shown next to the title.
* `layout="wide"`: Makes the app use the full width of the screen.

---

### 3. **Initialize Session State**

```python
if "tasks" not in st.session_state:
    st.session_state.tasks = []
```

Streamlit **resets** variables every time you interact (like clicking a button).
So we use `st.session_state` to **remember data** across interactions.

Here, we’re checking:

* If there’s no “tasks” list yet, we create one (`[]`).

So your list of tasks will stay stored while you use the app.

---

### 4. **Main Title and Divider**

```python
st.title("Task Management APP")
st.markdown("---")
```

* `st.title()` → Displays a big title at the top.
* `st.markdown("---")` → Adds a horizontal line for separation.

---

### 5. **Sidebar Menu**

```python
menu = st.sidebar.radio("Choose an action:", ["View Tasks", "Add Task", "Update Task", "Delete Task"])
```

This creates a **sidebar menu** with radio buttons (so you can pick one option at a time).

You can choose what you want to do:

* View Tasks
* Add Task
* Update Task
* Delete Task

The selected option is stored in the variable `menu`.

---

### 6. **View Tasks Section**

```python
if menu == "View Tasks":
    st.header("View Tasks")
    if st.session_state.tasks:
        for i, task in enumerate(st.session_state.tasks, 1):
            st.write(f"**{i}. {task}**")
    else:
        st.info("No tasks yet. Add one!")
```

If the user selects **“View Tasks”**, this part runs:

* Shows a section header: "View Tasks".
* Checks if there are any tasks saved.

  * If yes → displays them in a numbered list (using `enumerate`).
  * If no → shows an info message: “No tasks yet. Add one!”

---

### 7. **Add Task Section**

```python
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
```

If the user chooses **“Add Task”**:

* It shows a text box to type a new task.
* When the “Add” button is clicked:

  * It checks that the input isn’t empty.
  * Adds the new task to the list.
  * Shows a success message.
  * Then `st.rerun()` reloads the page to show updated tasks.

---

### 8. **Update Task Section**

```python
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
```

If the user chooses **“Update Task”**:

* A dropdown (`selectbox`) shows all existing tasks.
* You can pick one, type a new name for it, and click “Update”.
* The old task is replaced with the new one.
* If there are no tasks, it tells you there’s nothing to update.

---

### 9. **Delete Task Section**

```python
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
```

If the user chooses **“Delete Task”**:

* Shows a dropdown with all tasks.
* Lets you select one and click “Delete”.
* Removes the selected task from the list.
* Shows a success message.
* Refreshes the page to update the view.

---

### 10. **Show Quick Stats**

```python
st.sidebar.markdown("---")
st.sidebar.metric("Total Tasks", len(st.session_state.tasks))
```

At the bottom of the sidebar:

* Adds a separator line.
* Shows how many tasks you currently have using a **metric widget**.

---

## 🎯 Summary of What’s Happening

| Action             | What It Does                         |
| ------------------ | ------------------------------------ |
| **View Tasks**     | Shows all saved tasks                |
| **Add Task**       | Adds a new task to the list          |
| **Update Task**    | Changes the name of an existing task |
| **Delete Task**    | Removes a selected task              |
| **Sidebar Metric** | Displays total number of tasks       |

---

## 🧠 Key Streamlit Concepts Used

* **`st.session_state`** → Keeps data even after user interactions
* **`st.radio()` / `st.selectbox()` / `st.text_input()` / `st.button()`** → For user input
* **`st.success()` / `st.warning()` / `st.info()`** → For user feedback messages
* **`st.rerun()`** → Refreshes the app to show updates immediately

---
