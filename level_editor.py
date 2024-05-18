import tkinter as tk
from tkinter import messagebox
import json

class LevelEditor:
    def __init__(self, master, rows=10, columns=10):
        self.master = master
        self.rows = rows
        self.columns = columns
        self.cell_size = 40
        self.grid = [[0] * columns for _ in range(rows)]
        self.agents = {}  # Store agent positions

        self.canvas = tk.Canvas(master, width=self.columns * self.cell_size, height=self.rows * self.cell_size)
        self.canvas.grid(row=0, column=0, columnspan=columns)

        self.rectangles = [[None for _ in range(columns)] for _ in range(rows)]
        for row in range(rows):
            for col in range(columns):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='black')
                self.canvas.tag_bind(rect, '<Button-1>', lambda event, r=row, c=col: self.toggle_cell(r, c))
                self.canvas.tag_bind(rect, '<B1-Motion>', lambda event, r=row, c=col: self.toggle_cell_drag(event))
                self.rectangles[row][col] = rect

        # Object type currently selected to place on the grid
        self.current_object = 0
        self.current_color = 'white'  # Default color
        self.agent_colors = {5: 'red', 6: 'green', 7: 'yellow'}  # Agent types mapped to colors
        self.placed_agents = {5: None, 6: None, 7: None}  # Track placed agents

        # Last cell that was updated
        self.last_row = -1
        self.last_col = -1

        # Create a frame for control buttons
        control_frame = tk.Frame(master)
        control_frame.grid(row=rows+1, column=0, columnspan=columns, pady=10)

        tk.Button(control_frame, text='Clear', command=lambda: self.set_current_object(0), width=12, font=("Arial", 10)).pack(side=tk.LEFT)
        tk.Button(control_frame, text='Wall', bg='black', fg='white', command=lambda: self.set_current_object(1), width=12, font=("Arial", 10)).pack(side=tk.LEFT)
        tk.Button(control_frame, text='Yellow Button', bg='yellow', fg='black', command=lambda: self.set_current_object(2), width=12, font=("Arial", 10)).pack(side=tk.LEFT)
        tk.Button(control_frame, text='Green Button', bg='green', fg='white', command=lambda: self.set_current_object(3), width=12, font=("Arial", 10)).pack(side=tk.LEFT)
        tk.Button(control_frame, text='Red Button', bg='red', fg='white', command=lambda: self.set_current_object(4), width=12, font=("Arial", 10)).pack(side=tk.LEFT)
        tk.Button(control_frame, text='Red Agent', bg='red', fg='white', command=lambda: self.set_current_object(5), width=12, font=("Arial", 10)).pack(side=tk.LEFT)
        tk.Button(control_frame, text='Green Agent', bg='green', fg='white', command=lambda: self.set_current_object(6), width=12, font=("Arial", 10)).pack(side=tk.LEFT)
        tk.Button(control_frame, text='Yellow Agent', bg='yellow', fg='black', command=lambda: self.set_current_object(7), width=12, font=("Arial", 10)).pack(side=tk.LEFT)

        # Create a frame for saving and loading the layout
        save_load_frame = tk.Frame(master)
        save_load_frame.grid(row=rows+2, column=0, columnspan=columns, pady=10)

        tk.Label(save_load_frame, text="Filename:").pack(side=tk.LEFT)
        self.filename_entry = tk.Entry(save_load_frame)
        self.filename_entry.pack(side=tk.LEFT)
        tk.Button(save_load_frame, text='Save', command=self.save_layout).pack(side=tk.LEFT)
        tk.Button(save_load_frame, text='Load', command=self.load_layout).pack(side=tk.LEFT)

    def set_current_object(self, obj_type):
        self.current_object = obj_type
        self.current_color = self.agent_colors.get(obj_type, 'white')
        print(f"Current object set to: {self.current_object} with color {self.current_color}")

    def toggle_cell(self, row, col):
        if self.current_object in self.agent_colors:
            # Place an agent
            if self.placed_agents[self.current_object] is not None:
                self.canvas.delete(self.placed_agents[self.current_object])  # Remove existing agent
            x1 = col * self.cell_size + self.cell_size // 4
            y1 = row * self.cell_size + self.cell_size // 4
            x2 = x1 + self.cell_size // 2
            y2 = y1 + self.cell_size // 2
            agent = self.canvas.create_oval(x1, y1, x2, y2, fill=self.current_color, outline='black')
            self.placed_agents[self.current_object] = agent
            self.agents[(row, col)] = self.current_object
        else:
            # Place a regular object
            self.grid[row][col] = self.current_object
            if self.current_object == 1:
                self.current_color = 'black'
            elif self.current_object == 2:
                self.current_color = 'yellow'
            elif self.current_object == 3:
                self.current_color = 'green'
            elif self.current_object == 4:
                self.current_color = 'red'
            self.canvas.itemconfig(self.rectangles[row][col], fill=self.current_color)
        print(f"Toggled cell at ({row}, {col}) to object {self.current_object} with color {self.current_color}")

    def toggle_cell_drag(self, event):
        row = event.y // self.cell_size
        col = event.x // self.cell_size
        if 0 <= row < self.rows and 0 <= col < self.columns:
            if row != self.last_row or col != self.last_col:
                self.toggle_cell(row, col)
                self.last_row = row
                self.last_col = col

    def save_layout(self):
        filename = self.filename_entry.get()
        if not filename:
            messagebox.showerror("Error", "Please enter a filename")
            return
        data = {
            "grid": self.grid,
            "agents": {f"{row},{col}": self.agent_colors[agent_type] for (row, col), agent_type in self.agents.items()}
        }
        with open(f"{filename}.json", "w") as f:
            json.dump(data, f)
        messagebox.showinfo("Success", f"Layout saved to {filename}.json")

    def load_layout(self):
        filename = self.filename_entry.get()
        if not filename:
            messagebox.showerror("Error", "Please enter a filename")
            return
        try:
            with open(f"{filename}.json", "r") as f:
                data = json.load(f)
                self.grid = data["grid"]
                self.agents = {}
                self.placed_agents = {5: None, 6: None, 7: None}  # Reset placed agents
                for key, color in data["agents"].items():
                    row, col = map(int, key.split(','))
                    agent_type = [k for k, v in self.agent_colors.items() if v == color][0]
                    x1 = col * self.cell_size + self.cell_size // 4
                    y1 = row * self.cell_size + self.cell_size // 4
                    x2 = x1 + self.cell_size // 2
                    y2 = y1 + self.cell_size // 2
                    agent = self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline='black')
                    self.agents[(row, col)] = agent_type
                    self.placed_agents[agent_type] = agent
            for row in range(self.rows):
                for col in range(self.columns):
                    self.update_cell_color(row, col)
            messagebox.showinfo("Success", f"Layout loaded from {filename}.json")
        except FileNotFoundError:
            messagebox.showerror("Error", f"File {filename}.json not found")

    def update_cell_color(self, row, col):
        color = 'white'
        if self.grid[row][col] == 1:
            color = 'black'
        elif self.grid[row][col] == 2:
            color = 'yellow'
        elif self.grid[row][col] == 3:
            color = 'green'
        elif self.grid[row][col] == 4:
            color = 'red'

        self.canvas.itemconfig(self.rectangles[row][col], fill=color)

if __name__ == "__main__":
    root = tk.Tk()
    editor = LevelEditor(root)
    root.mainloop()