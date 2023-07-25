from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PySide6.QtGui import QGuiApplication
from PySide6.QtCore import Qt

import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # You will need to replace this with your own video display widget
        # The QPushButton is just used as a placeholder
        self.video_display = QPushButton("Video Stream Placeholder")

        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_display)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
