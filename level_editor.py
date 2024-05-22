import sys
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QLabel, QLineEdit, QMessageBox
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtCore import Qt, QPointF

class LevelEditor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.rows = 10
        self.columns = 10
        self.grid = [[0] * self.columns for _ in range(self.rows)]
        self.cell_size = 40
        self.current_object = 0
        self.current_color = 'white'
        self.agent_colors = {5: 'red', 6: 'green', 7: 'yellow'}
        self.placed_agents = {5: None, 6: None, 7: None}
        self.agent_items = {5: None, 6: None, 7: None}  # Store agent QGraphicsEllipseItems
        self.dragging = False

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setFixedSize(self.columns * self.cell_size + 2, self.rows * self.cell_size + 2)
        layout.addWidget(self.view)

        self.rectangles = []
        for row in range(self.rows):
            row_items = []
            for col in range(self.columns):
                rect = QGraphicsRectItem(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                rect.setBrush(QBrush(QColor('white')))
                rect.setPen(QColor('black'))
                rect.setData(0, (row, col))
                self.scene.addItem(rect)
                row_items.append(rect)
            self.rectangles.append(row_items)

        self.view.setMouseTracking(True)
        self.view.mousePressEvent = self.on_mouse_press
        self.view.mouseMoveEvent = self.on_mouse_move
        self.view.mouseReleaseEvent = self.on_mouse_release

        control_frame = QWidget()
        control_layout = QHBoxLayout(control_frame)
        layout.addWidget(control_frame)

        self.add_button(control_layout, 'Clear', 'white', 0)
        self.add_button(control_layout, 'Wall', 'black', 1, 'white')
        self.add_button(control_layout, 'Yellow Button', 'yellow', 2)
        self.add_button(control_layout, 'Green Button', 'green', 3)
        self.add_button(control_layout, 'Red Button', 'red', 4)
        self.add_button(control_layout, 'Red Agent', 'red', 5)
        self.add_button(control_layout, 'Green Agent', 'green', 6)
        self.add_button(control_layout, 'Yellow Agent', 'yellow', 7)

        save_load_frame = QWidget()
        save_load_layout = QHBoxLayout(save_load_frame)
        layout.addWidget(save_load_frame)

        self.filename_entry = QLineEdit()
        save_load_layout.addWidget(QLabel("Filename:"))
        save_load_layout.addWidget(self.filename_entry)

        save_btn = QPushButton('Save')
        save_btn.clicked.connect(self.save_layout)
        save_load_layout.addWidget(save_btn)

        load_btn = QPushButton('Load')
        load_btn.clicked.connect(self.load_layout)
        save_load_layout.addWidget(load_btn)

        self.setWindowTitle('Level Editor')
        self.show()

    def add_button(self, layout, text, bg_color, obj_type, text_color='black'):
        button = QPushButton(text)
        button.setStyleSheet(f"background-color: {bg_color}; color: {text_color};")
        button.clicked.connect(lambda: self.set_current_object(obj_type, bg_color))
        layout.addWidget(button)

    def set_current_object(self, obj_type, color):
        self.current_object = obj_type
        self.current_color = color

    def on_mouse_press(self, event):
        self.dragging = True
        self.toggle_cell(event)

    def on_mouse_move(self, event):
        if self.dragging:
            self.toggle_cell(event)

    def on_mouse_release(self, event):
        self.dragging = False

    def toggle_cell(self, event):
        pos = self.view.mapToScene(event.pos())
        items = self.scene.items(QPointF(pos.x(), pos.y()))
        if items:
            item = items[0]
            if isinstance(item, QGraphicsRectItem):
                row, col = item.data(0)
                if self.current_object in self.agent_colors:
                    if self.placed_agents[self.current_object] is not None:
                        old_row, old_col = self.placed_agents[self.current_object]
                        self.rectangles[old_row][old_col].setBrush(QBrush(QColor('white')))
                        self.scene.removeItem(self.agent_items[self.current_object])
                    agent_size = self.cell_size * 0.5
                    agent = QGraphicsEllipseItem(col * self.cell_size + self.cell_size * 0.25, row * self.cell_size + self.cell_size * 0.25, agent_size, agent_size)
                    agent.setBrush(QBrush(QColor(self.current_color)))
                    agent.setPen(QColor('black'))
                    self.scene.addItem(agent)
                    self.agent_items[self.current_object] = agent
                    self.rectangles[row][col].setBrush(QBrush(QColor('white')))
                    self.placed_agents[self.current_object] = (row, col)
                else:
                    item.setBrush(QBrush(QColor(self.current_color)))
                    self.grid[row][col] = self.current_object

    def save_layout(self):
        filename = self.filename_entry.text()
        if not filename:
            QMessageBox.warning(self, "Error", "Please enter a filename")
            return
        data = {
            "grid": self.grid,
            "agents": {f"{row},{col}": agent for agent, (row, col) in self.placed_agents.items() if self.placed_agents[agent] is not None}
        }
        with open(f"{filename}.json", "w") as f:
            json.dump(data, f)
        QMessageBox.information(self, "Success", f"Layout saved to {filename}.json")

    def load_layout(self):
        filename = self.filename_entry.text()
        if not filename:
            QMessageBox.warning(self, "Error", "Please enter a filename")
            return
        try:
            with open(f"{filename}.json", "r") as f:
                data = json.load(f)
                self.grid = data["grid"]
                self.placed_agents = {agent: None for agent in self.agent_colors}
                self.agent_items = {agent: None for agent in self.agent_colors}
                for key, agent in data["agents"].items():
                    row, col = map(int, key.split(','))
                    self.placed_agents[agent] = (row, col)
                self.update_grid()
            QMessageBox.information(self, "Success", f"Layout loaded from {filename}.json")
        except FileNotFoundError:
            QMessageBox.warning(self, "Error", f"File {filename}.json not found")

    def update_grid(self):
        for row in range(self.rows):
            for col in range(self.columns):
                obj_type = self.grid[row][col]
                color = 'white'
                if obj_type == 1:
                    color = 'black'
                elif obj_type == 2:
                    color = 'yellow'
                elif obj_type == 3:
                    color = 'green'
                elif obj_type == 4:
                    color = 'red'
                self.rectangles[row][col].setBrush(QBrush(QColor(color)))

        for agent, position in self.placed_agents.items():
            if position is not None:
                row, col = position
                color = self.agent_colors[agent]
                agent_size = self.cell_size * 0.5
                agent_item = QGraphicsEllipseItem(col * self.cell_size + self.cell_size * 0.25, row * self.cell_size + self.cell_size * 0.25, agent_size, agent_size)
                agent_item.setBrush(QBrush(QColor(color)))
                agent_item.setPen(QColor('black'))
                self.scene.addItem(agent_item)
                self.agent_items[agent] = agent_item
                self.rectangles[row][col].setBrush(QBrush(QColor('white')))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = LevelEditor()
    sys.exit(app.exec_())