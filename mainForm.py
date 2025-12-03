from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation
from PyQt5.QtGui import QPixmap, QPalette, QColor, QPainter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QLabel, QPushButton,
    QAction, QComboBox, QLineEdit, QCheckBox, QRadioButton, QSlider,
    QSpinBox, QTableWidget, QTableWidgetItem, QTextEdit,
    QScrollArea, QGridLayout, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QWidget, QGroupBox, QGraphicsDropShadowEffect, QGraphicsOpacityEffect
)

import os

# 自定义菜单栏类
class CustomMenuBar(QtWidgets.QMenuBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.background_color = QColor(0, 5, 90)  # 菜单栏背景色（深蓝色）
        self.text_color = Qt.white                     # 文字颜色
        self.hover_color = QColor(0, 48, 144)          # 悬停背景色

        self.setFont(QtGui.QFont("Microsoft YaHei", 10))

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = event.rect()

        # 填充背景
        painter.fillRect(rect, self.background_color)

        # 绘制每个菜单项
        for action in self.actions():
            opt = QtWidgets.QStyleOptionButton()
            opt.rect = self.actionGeometry(action)
            opt.text = action.text()
            opt.state = QtWidgets.QStyle.State_Enabled

            if self.activeAction() == action:
                opt.state |= QtWidgets.QStyle.State_MouseOver

            # 设置文本颜色
            painter.setPen(self.text_color)

            # 填充背景
            if opt.state & QtWidgets.QStyle.State_MouseOver:
                painter.fillRect(opt.rect, self.hover_color)
            else:
                painter.fillRect(opt.rect, self.background_color)

            painter.drawText(opt.rect, Qt.AlignCenter, opt.text)

    def mouseMoveEvent(self, event):
        for action in self.actions():
            if self.actionGeometry(action).contains(event.pos()):
                self.update(self.actionGeometry(action))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        for action in self.actions():
            self.update(self.actionGeometry(action))
        super().leaveEvent(event)


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(1050, 900)

        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")

        bg_size = QtCore.QSize(4096, 2160)

        self.backgroundLabel = QLabel(self.centralwidget)
        self.backgroundLabel.setObjectName("backgroundLabel")
        self.backgroundLabel.setScaledContents(False)
        pixmap = QPixmap("img/bg.png").scaled(
            bg_size,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation
        )
        self.backgroundLabel.setPixmap(pixmap)
        self.backgroundLabel.setGeometry(
            (mainWindow.width() - bg_size.width()) // 2,
            (mainWindow.height() - bg_size.height()) // 2,
            bg_size.width(),
            bg_size.height()
        )

        # == 主布局：网格布局 ==
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout.setContentsMargins(30, 30, 30, 30)
        self.gridLayout.setSpacing(25)

        # == 左上角：图片预览区域 ==
        self.imageView = QLabel()
        self.imageView.setObjectName("imageView")
        self.imageView.setAlignment(Qt.AlignCenter)
        self.imageView.setFixedHeight(360)
        self.imageView.setStyleSheet("""
            background-color: rgba(255, 255, 255, 150);
            border-radius: 16px;
            border: 2px solid rgba(200, 200, 200, 50);
            padding: 10px;
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QtGui.QColor(0, 0, 0, 50))
        shadow.setOffset(0, 4)
        self.imageView.setGraphicsEffect(shadow)
        # 设置imageView的大小策略为水平方向可扩展，垂直方向固定
        self.imageView.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addWidget(self.imageView, 0, 0)

        # == 右上角：GIF 显示区域 ==
        self.gifView = QtWidgets.QLabel()
        self.gifView.setObjectName("gifView")
        self.gifView.setAlignment(Qt.AlignCenter)
        self.gifView.setFixedSize(480, 360)
        self.gifView.setStyleSheet("""
            background-color: rgba(255, 255, 255, 150);
            border-radius: 16px;
            border: 2px solid rgba(200, 200, 200, 50);
            padding: 10px;
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QtGui.QColor(0, 0, 0, 50))
        shadow.setOffset(0, 4)
        self.gifView.setGraphicsEffect(shadow)
        self.gridLayout.addWidget(self.gifView, 0, 1)

        # == 左下角：日志显示区域 ==
        self.logContainer = QtWidgets.QWidget()
        self.logContainer.setObjectName("logContainer")
        self.logContainer.setStyleSheet("""
            QWidget {
                background-color: rgba(240, 240, 240, 150);  /* 灰色半透明背景 */
                border-radius: 16px;                         /* 圆角 */
                padding: 12px;
            }
        """)

        shadow_log_container = QGraphicsDropShadowEffect(self.logContainer)
        shadow_log_container.setBlurRadius(20)
        shadow_log_container.setColor(QtGui.QColor(0, 0, 0, 50))  # 阴影颜色（黑色半透明）
        shadow_log_container.setOffset(0, 4)                      # 偏移量
        self.logContainer.setGraphicsEffect(shadow_log_container)

        self.logTextBox = QtWidgets.QTextEdit()
        self.logTextBox.setObjectName("logTextBox")
        self.logTextBox.setMinimumHeight(250)
        self.logTextBox.setReadOnly(True)
        self.logTextBox.setStyleSheet("""
            QTextEdit {
                background-color: #B0B0B0;
                border-radius: 16px;                         /* 圆角 */
                border: 2px solid rgba(200, 200, 200, 50);   /* 边框 */
                padding: 12px;                               /* 内边距 */
                font-family: "Consolas", "Microsoft YaHei", monospace;
                font-size: 14px;
                color: #003366;
                selection-background-color: #4a86e8;
                selection-color: white;
            }
            QScrollBar:vertical {
                width: 12px;
                margin: 16px 0 16px 0;
                border: none;
                background-color: transparent;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical {
                background-color: rgba(200, 200, 200, 150);
                min-height: 30px;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: rgba(150, 150, 150, 150);
            }

            QScrollBar::add-line:vertical {
                height: 16px;
                subcontrol-origin: margin;
                subcontrol-position: bottom right;
                border-image: ▼;
                border-width: 1px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
                background-color: rgba(200, 200, 200, 150);
            }

            QScrollBar::sub-line:vertical {
                height: 16px;
                subcontrol-origin: margin;
                subcontrol-position: top right;
                border-image: ▲;
                border-width: 1px;
                border-top-left-radius: 6px;
                border-bottom-left-radius: 6px;
                background-color: rgba(200, 200, 200, 150);
            }

            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                width: 12px;
                height: 12px;
                background: none;
            }

            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background-color: transparent;
            }
        """)

        log_layout = QtWidgets.QVBoxLayout(self.logContainer)
        log_layout.addWidget(self.logTextBox)

        # 添加日志区加入布局
        self.gridLayout.addWidget(self.logContainer, 1, 0)

        # == 右下角：操作面板（按钮组）==
        self.operationPanel = QtWidgets.QWidget()
        self.operationPanel.setObjectName("operationPanel")
        self.operationPanel.setStyleSheet("""
            background-color: rgba(255, 255, 255, 150);
            border-radius: 16px;
            border: 2px solid rgba(200, 200, 200, 50);
            padding: 10px;
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QtGui.QColor(0, 0, 0, 50))
        shadow.setOffset(0, 4)
        self.operationPanel.setGraphicsEffect(shadow)

        self.operationLayout = QtWidgets.QVBoxLayout(self.operationPanel)
        self.operationLayout.setSpacing(16)

        # 控制按钮组
        self.controlButtons = QtWidgets.QHBoxLayout()
        self.controlButtons.setSpacing(16)

        self.btnStartCheck = QtWidgets.QPushButton("▶️ 启动复判")
        self.btnStartCheck.setObjectName("btnStartCheck")
        self.btnStartCheck.setStyleSheet("""
            QPushButton {
                background-color: #B0B0B0;  /* 灰色 */
                color: white;
                border-radius: 14px;
                font-family: "Microsoft YaHei";
                font-size: 18px;
                font-weight: 500;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #3631CE;
            }
            QPushButton:pressed {
                background-color: #3631CE;
            }
        """)

        self.btnStopCheck = QtWidgets.QPushButton("⏸️ 停止复判")
        self.btnStopCheck.setObjectName("btnStopCheck")
        self.btnStopCheck.setStyleSheet("""
            QPushButton {
                background-color: #B0B0B0;  /* 灰色 */
                color: white;
                border-radius: 14px;
                font-family: "Microsoft YaHei";
                font-size: 18px;
                font-weight: 500;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #DD2442;
            }
            QPushButton:pressed {
                background-color: #DD2442;
            }
        """)
        self.controlButtons.addWidget(self.btnStartCheck)
        self.controlButtons.addWidget(self.btnStopCheck)
        self.operationLayout.addLayout(self.controlButtons)

        # 状态信息组
        self.statusGroup = QtWidgets.QWidget()
        self.statusGroup.setObjectName("statusGroup")
        self.statusGroup.setStyleSheet("""
            background-color: rgba(240, 240, 240, 150);
            border-radius: 12px;
            padding: 8px;
        """)
        self.statusLayout = QtWidgets.QVBoxLayout(self.statusGroup)

        self.state = QtWidgets.QLabel("欢迎使用AOI智能复判系统")
        self.state.setObjectName("state")
        self.state.setAlignment(Qt.AlignCenter)
        self.state.setStyleSheet("font-size: 20px; font-weight: 600; color: black;")
        self.state.setMinimumHeight(100)
        self.statusLayout.addWidget(self.state)

        # 添加实时数据统计文本框
        self.dataStats = QtWidgets.QTextEdit()
        self.dataStats.setObjectName("dataStats")
        self.dataStats.setReadOnly(True)
        self.dataStats.setStyleSheet("font-size: 15px; font-weight: 600; color: black;")
        self.statusLayout.addWidget(self.dataStats)

        self.operationLayout.addWidget(self.statusGroup)

        self.gridLayout.addWidget(self.operationPanel, 1, 1)

        # 设置行权重
        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 0)

        # 设置列权重
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 0)

        # 设置菜单栏和状态栏
        mainWindow.setCentralWidget(self.centralwidget)

        # 使用自定义菜单栏
        self.menubar = CustomMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 30))
        self.menubar.setObjectName("menubar")
        self.menuSystem = self.menubar.addMenu("系统设置")
        self.menuDataQuery = self.menubar.addMenu("数据查询")
        
        mainWindow.setMenuBar(self.menubar)

        # 信号槽连接
        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "AOI智能复判系统V2.0"))