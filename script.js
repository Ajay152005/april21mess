// Get references to the input field and task list

const taskInput = document.getElementById('taskInput');
const taskList = document.getElementById('taskList');

// Function to add a new task
function addTask() {
    // Get the task text from the input field
    const taskText = taskInput.value.trim();

    // If the input field is empty, do nothing

    if (taskText === '') {
        return;
    }

    // Creating a new list item element
    const listItem = document.createElement('li');

    // Create a checkbox for task completion
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.onchange = function() {
        toggleTaskCompletion(this);

    }

    // Create a span element for the task text

    const taskSpan = document.createElement('span');
    taskSpan.className = 'task-text';
    taskSpan.textContent = taskText;

    // Create a delete button
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = 'Delete';
    deleteBtn.onclick = function() {
        deleteTask(this);
    };

    //Append elements to the list item
    listItem.appendChild(checkbox);
    listItem.appendChild(taskSpan);
    listItem.appendChild(deleteBtn);
    

    // Set the text content of the list item
    listItem.textContent = taskText;

    // Append the list item to the task list
    taskList.appendChild(listItem);

    // Clear the input field
    taskInput.value = '';
}

// Function to toggle task completion status

function toggleTaskCompletion(checkbox) {
    const listItem = checkbox.parentNode;
    listItem.classList.toggle('completed');
}

// function to delete a task
function deleteTask(button) {
    const listItem = button.parentNode;
    listItem.remove();
}