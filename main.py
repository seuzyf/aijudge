import sys
import os
import pandas as pd
import cv2
import logging
from logging.handlers import RotatingFileHandler
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QDate
from PyQt5.QtGui import QIcon, QImage, QPixmap, QPainter, QColor 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QLabel, QPushButton,
    QLineEdit, QComboBox, QCheckBox, QRadioButton, QSlider,
    QSpinBox, QTableWidget, QTableWidgetItem, QTextEdit,
    QScrollArea, QGridLayout, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QWidget, QGroupBox, QAction, QHeaderView, QGraphicsDropShadowEffect, QDateEdit,
    QFormLayout
)
from mainForm import Ui_mainWindow
import time
import multiprocessing
import re
from collections import Counter
from PyQt5.QtGui import QMovie
import numpy as np
from utils import get_device_module
import matplotlib.backends.backend_qt5agg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import shutil
from pathlib import Path

# 日志配置
logger = logging.getLogger()
logger.setLevel(logging.INFO) # 设置日志级别

# 创建一个RotatingFileHandler
# maxBytes=5*1024*1024 表示每个日志文件最大为 5MB
# backupCount=5 表示最多保留5个备份文件 (check.log.1, check.log.2, ...)
# encoding='gbk' 推荐使用gbk以避免编码问题
handler = RotatingFileHandler(
    'check.log', 
    maxBytes=512*1024, 
    backupCount=5,
    encoding='gbk'
)

# 创建一个日志格式器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(handler)

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, text=""):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)  # 显示手型光标

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统设置")
        self.resize(900, 600)

        # 设置整体样式：黑色背景 + 白色字体
        self.setStyleSheet("""
            QDialog {
                background-color: #1E1E1E;
                color: white;
            }
            QLabel {
                color: white;
                padding: 4px;
            }
            QComboBox {
                background-color: #2D2D2D;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px;
                color: white;
            }
            QPushButton {
                background-color: #3A3A3A;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                color: white;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QTableWidget {
                background-color: #1E1E1E;
                color: white;
                gridline-color: #444;
                border: none;
            }
            QHeaderView::section {
                background-color: #1E1E1E;
                color: white;
                padding: 6px;
                border: 1px solid #444;
                font-weight: bold;
            }
            QTableWidget::item {
                background-color: #1E1E1E;
                color: white;
            }
        """)

        self.layout = QVBoxLayout()

        # 创建表格
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setRowCount(5)
        self.table.setHorizontalHeaderLabels(["设置项", "值"])
        self.table.verticalHeader().setVisible(False)

        # 设置第二列自适应宽度
        self.table.horizontalHeader().setStretchLastSection(True)

        # 行 0：图片路径
        self.image_path_label = ClickableLabel("当前路径未设置")
        self.image_path_label.setStyleSheet("""
            background-color: #1E1E1E;
            color: white;
            padding: 6px;
            border-radius: 4px;
        """)
        self.image_path_label.clicked.connect(self.select_image_path)

        image_item = QTableWidgetItem("图片路径")
        image_item.setFlags(image_item.flags() & ~Qt.ItemIsEditable)  # 不可编辑
        self.table.setItem(0, 0, image_item)
        self.table.setCellWidget(0, 1, self.image_path_label)

        # 行 1：结果路径
        self.result_path_label = ClickableLabel("当前路径未设置")
        self.result_path_label.setStyleSheet("""
            background-color: #1E1E1E;
            color: white;
            padding: 6px;
            border-radius: 4px;
        """)
        self.result_path_label.clicked.connect(self.select_result_path)

        result_item = QTableWidgetItem("结果路径")
        result_item.setFlags(result_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(1, 0, result_item)
        self.table.setCellWidget(1, 1, self.result_path_label)

        # 行 2：OK范围
        self.ok_range_combo = QComboBox()
        self.ok_range_combo.addItems([f"{i * 0.1:.1f}" for i in range(1, 11)])
        ok_item = QTableWidgetItem("OK范围")
        ok_item.setFlags(ok_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(2, 0, ok_item)
        self.table.setCellWidget(2, 1, self.ok_range_combo)

        # 行 3：图片收集
        self.collect_combo = QComboBox()
        self.collect_combo.addItems(["开启", "关闭"])
        collect_item = QTableWidgetItem("图片收集")
        collect_item.setFlags(collect_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(3, 0, collect_item)
        self.table.setCellWidget(3, 1, self.collect_combo)

        # 行 4：设备类型
        self.device_combo = QComboBox()
        self.device_combo.addItems(["神州", "神州SMT", "奔创", "Saki", "KY"])
        device_item = QTableWidgetItem("设备类型")
        device_item.setFlags(device_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(4, 0, device_item)
        self.table.setCellWidget(4, 1, self.device_combo)

        self.layout.addWidget(self.table)
        self.setLayout(self.layout)

        # 初始化数据
        self.load_settings()

    def load_settings(self):
        config_path = 'config.txt'
        if not os.path.exists(config_path):
            return

        config = {}
        with open(config_path, 'r', encoding='gbk') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key] = value

        imagePath = config.get('ImagePath', '')
        resultPath = config.get('ResultPath', '')
        okRange = config.get('okRange', '0.9')
        collect = config.get('collect', '1')
        device = config.get('Device', '神州')

        self.image_path_label.setText(imagePath)
        self.result_path_label.setText(resultPath)
        self.ok_range_combo.setCurrentText(okRange)
        self.collect_combo.setCurrentText('开启' if collect == '1' else '关闭')
        self.device_combo.setCurrentText(device)

    def select_image_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片路径")
        if path:
            self.image_path_label.setText(path)

    def select_result_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择结果路径")
        if path:
            self.result_path_label.setText(path)

    def get_settings(self):
        return {
            'ImagePath': self.image_path_label.text(),
            'ResultPath': self.result_path_label.text(),
            'okRange': self.ok_range_combo.currentText(),
            'collect': '1' if self.collect_combo.currentText() == '开启' else '0',
            'Device': self.device_combo.currentText()
        }

    def closeEvent(self, event):
        settings = self.get_settings()
        self.parent().imagePath = settings['ImagePath']
        self.parent().resPath = settings['ResultPath']
        self.parent().okRange = float(settings['okRange'])
        self.parent().collect = int(settings['collect']) if settings['collect'].isdigit() else 1
        self.parent().device = settings['Device']
        self.parent().saveConfig()
        event.accept()

class DataQueryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据查询")
        self.resize(1600, 900)
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"设置字体失败: {e}")
        # 主布局
        main_layout = QHBoxLayout(self)
        # 左侧：图表 + 直通率文字
        left_widget = QWidget()
        self.left_layout = QVBoxLayout(left_widget)
        self.left_layout.setAlignment(Qt.AlignTop)
        self.left_layout.setSpacing(12)
        # 卡片风格 - 直通率统计
        pass_rate_group = QGroupBox()
        pass_rate_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d1d5db;
                border-radius: 16px;
                margin-top: 8px;
                padding: 12px;
                background-color: white;
            }
            QGroupBox::title {
                text-align: center;
                padding: 4px 10px;
                color: #4b5563;
                background-color: #f3f4f6;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                top: -12px;
            }
        """)
        shadow_effect_pass = QGraphicsDropShadowEffect()
        shadow_effect_pass.setBlurRadius(20)
        shadow_effect_pass.setColor(QColor(0, 0, 0, 50))
        shadow_effect_pass.setOffset(0, 4)
        pass_rate_group.setGraphicsEffect(shadow_effect_pass)
        self.pass_rate_label = QLabel()
        self.pass_rate_label.setTextFormat(Qt.RichText)
        self.pass_rate_label.setStyleSheet("background-color: #f9fafb; font-size: 14px; padding: 8px;")
        pass_rate_layout = QVBoxLayout()
        pass_rate_layout.addWidget(self.pass_rate_label)
        pass_rate_group.setLayout(pass_rate_layout)
        self.left_layout.addWidget(pass_rate_group)
        # 卡片风格 - 图表区域容器
        self.chart_container = QWidget()
        self.chart_container.setObjectName("chart_container")
        self.chart_container.setStyleSheet("")
        shadow_effect_chart = QGraphicsDropShadowEffect()
        shadow_effect_chart.setBlurRadius(20)
        shadow_effect_chart.setColor(QColor(0, 0, 0, 50))
        shadow_effect_chart.setOffset(0, 4)
        self.chart_container.setGraphicsEffect(shadow_effect_chart)
        self.chart_layout = QVBoxLayout(self.chart_container)
        self.chart_layout.setContentsMargins(0, 0, 0, 0)
        self.chart_layout.setSpacing(10)
        self.left_layout.addWidget(self.chart_container)
        # 初始化图像标签列表
        self.image_labels = []
        # 右侧：表格 + 筛选面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(16)
        # 筛选面板（右上角）
        filter_group = QGroupBox()
        filter_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d1d5db;
                border-radius: 16px;
                margin-top: 8px;
                padding: 12px;
                background-color: white;
            }
        """)
        shadow_effect_filter = QGraphicsDropShadowEffect()
        shadow_effect_filter.setBlurRadius(20)
        shadow_effect_filter.setColor(QColor(0, 0, 0, 50))
        shadow_effect_filter.setOffset(0, 4)
        filter_group.setGraphicsEffect(shadow_effect_filter)
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setSpacing(12)  # 更紧凑
        filter_layout.setContentsMargins(12, 12, 12, 12)
        # 第一行：日期选择
        date_layout = QHBoxLayout()
        self.start_date_edit = QDateEdit(calendarPopup=True)
        self.end_date_edit = QDateEdit(calendarPopup=True)
        self.start_date_edit.setDate(QDate.currentDate().addDays(-7))
        self.end_date_edit.setDate(QDate.currentDate())
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        for date_edit in [self.start_date_edit, self.end_date_edit]:
            date_edit.setStyleSheet("""
                QDateEdit {
                    padding: 8px 12px;
                    border: 1px solid #d1d5db;
                    border-radius: 12px;
                    font-size: 14px;
                    background-color: #f9fafb;
                }
                QDateEdit:focus {
                    border-color: #3B82F6;
                    outline: none;
                }
            """)
        date_layout.addWidget(QLabel("起始日期(00:00):"), stretch=0)
        date_layout.addWidget(self.start_date_edit, stretch=1)
        date_layout.addWidget(QLabel("结束日期(00:00):"), stretch=0)
        date_layout.addWidget(self.end_date_edit, stretch=1)
        # 第二行：下拉框 + 输入框 + 查询按钮
        input_layout = QHBoxLayout()
        # 下拉框美化（含下拉箭头）
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["任务令", "单板条码", "缺陷类型", "程序名"])
        self.filter_combo.setStyleSheet("""
            QComboBox {
                padding: 8px 12px;
                border: 1px solid #d1d5db;
                border-radius: 12px;
                font-size: 14px;
                background-color: #f9fafb;
                text-align: center; /* 文字居中 */
            }
            QComboBox::down-arrow {
                image: url(:/icons/down_arrow.png); /* 如果有图标资源 */
                width: 16px;
                height: 16px;
            }
            QComboBox::drop-down {
                border: none;
                background-color: transparent;
            }
            QComboBox:focus {
                border-color: #3B82F6;
                outline: none;
            }
        """)
        # 输入框美化
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("输入筛选值")
        self.filter_input.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #d1d5db;
                border-radius: 12px;
                font-size: 14px;
                background-color: #f9fafb;
            }
            QLineEdit:focus {
                border-color: #3B82F6;
                outline: none;
            }
        """)
        # 查询按钮美化
        self.query_button = QPushButton("查询数据")
        self.query_button.setFixedWidth(100)
        self.query_button.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                padding: 8px 16px;
                border-radius: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        self.query_button.clicked.connect(self.on_query)
        input_layout.addWidget(self.filter_combo, stretch=0)
        input_layout.addWidget(self.filter_input, stretch=1)
        input_layout.addWidget(self.query_button, stretch=0)
        filter_layout.addLayout(date_layout)
        filter_layout.addLayout(input_layout)
        right_layout.addWidget(filter_group)
        # 表格区域
        table_group = QGroupBox()
        table_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d1d5db;
                border-radius: 16px;
                margin-top: 8px;
                padding: 12px;
                background-color: white;
            }
        """)
        shadow_effect_table = QGraphicsDropShadowEffect()
        shadow_effect_table.setBlurRadius(20)
        shadow_effect_table.setColor(QColor(0, 0, 0, 50))
        shadow_effect_table.setOffset(0, 4)
        table_group.setGraphicsEffect(shadow_effect_table)
        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_widget.setStyleSheet("""
            QTableWidget {
                border: 1px solid #d1d5db;
                background-color: white;
            }
            QHeaderView::section {
                background-color: #f9fafb;
                padding: 6px;
                border-bottom: 1px solid #e5e7eb;
                font-weight: bold;
                color: #374151;
            }
            QTableWidget::item {
                padding: 6px;
            }
        """)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.table_widget)
        table_layout = QVBoxLayout(table_group)
        table_layout.addWidget(self.scroll_area)
        right_layout.addWidget(table_group)
        # 设置主布局
        main_layout.addWidget(left_widget, stretch=0)
        main_layout.addWidget(right_widget, stretch=1)
        # 初始加载数据
        self.update_data()

    def update_data(self, df=None):
        if df is None:
            csv_path = 'history.csv'
            if not os.path.exists(csv_path):
                self.table_widget.setRowCount(0)
                self.table_widget.setColumnCount(0)
                return
            df = pd.read_csv(csv_path)
        self.load_table_data(df)
        (review_ok_boards, pre_ok_boards, ng_boards, ok_images, ng_images, total_boards,
         current_task_order_review_ok_boards, current_task_order_pre_ok_boards,
         current_task_order_ng_boards, current_task_order_ok_images,
         current_task_order_ng_images, current_task_order_total_boards) = self.analyze_data(df)
        self.plot_statistics(ok_images, ng_images,
                             current_task_order_review_ok_boards, current_task_order_pre_ok_boards,
                             current_task_order_ng_boards, review_ok_boards, pre_ok_boards, ng_boards,
                             current_task_order_total_boards, total_boards)

    def on_query(self):
        print("【开始】进入 on_query 函数")
        filter_type = self.filter_combo.currentText()
        value = self.filter_input.text().strip()
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")
        print(f"【筛选条件】filter_type: {filter_type}, value: '{value}', 日期范围: {start_date} ~ {end_date}")
        # 校验筛选条件
        if (filter_type == "请选择筛选条件" or not value) and start_date >= end_date:
            print("【提示】未输入任何筛选条件")
            QMessageBox.warning(self, "提示", "请至少输入一个筛选条件！")
            return
        csv_path = 'history.csv'
        print(f"【检查文件】CSV 路径是否存在: {os.path.exists(csv_path)}")
        if not os.path.exists(csv_path):
            print("【错误】CSV 文件不存在")
            return
        print("【开始读取 CSV】")
        self.query_button.setEnabled(False)
        df = pd.read_csv(csv_path)
        print(f"【读取完成】共 {len(df)} 行数据")
        print("【筛选开始】")
        filtered_df = df.copy()
        # 时间戳筛选
        if "日期" in filtered_df.columns:
            try:
                filtered_df["日期"] = pd.to_datetime(filtered_df["日期"], errors='coerce')
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                filtered_df = filtered_df[(filtered_df["日期"] >= start) & (filtered_df["日期"] < end)]
                print(f"【日期筛选完成】剩余 {len(filtered_df)} 行数据")
            except Exception as e:
                print(f"【日期筛选失败】错误信息: {e}")
        # 字段筛选
        if filter_type != "请选择筛选条件" and value:
            if filter_type in filtered_df.columns:
                print(f"【字段筛选】按 '{filter_type}' 列进行模糊匹配，关键词: '{value}'")
                try:
                    filtered_df = filtered_df[
                        filtered_df[filter_type].astype(str).str.contains(value, case=False, na=False, regex=False)
                    ]
                    print(f"【字段筛选】完成，剩余 {len(filtered_df)} 行数据")
                except Exception as e:
                    print(f"【字段筛选失败】错误信息: {e}")
            else:
                print(f"【字段筛选跳过】'{filter_type}' 列不存在于数据中")
        print("【筛选结束】准备更新界面")
        # ✅ 新增：无数据时只弹出提示框，不更新任何界面
        if len(filtered_df) == 0:
            QMessageBox.warning(self, "提示", "无有效数据，请检查筛选条件！")
            self.query_button.setEnabled(True)
            return
        self.update_statistics_only(filtered_df)
        self.query_button.setEnabled(True)
        print("【on_query 完成】")

    def update_statistics_only(self, df=None):
        df = df if df is not None else pd.DataFrame()
        (review_ok_boards, pre_ok_boards, ng_boards, ok_images, ng_images, total_boards,
        current_task_order_review_ok_boards, current_task_order_pre_ok_boards,
        current_task_order_ng_boards, current_task_order_ok_images,
        current_task_order_ng_images, current_task_order_total_boards) = self.analyze_data(df)
        self.plot_statistics(ok_images, ng_images,
                            current_task_order_review_ok_boards, current_task_order_pre_ok_boards,
                            current_task_order_ng_boards, review_ok_boards, pre_ok_boards, ng_boards,
                            current_task_order_total_boards, total_boards)
        print("【刷新完成】")

    def load_table_data(self, df=None):
        df = df if df is not None else pd.DataFrame()
        print(f"【load_table_data】开始加载 {len(df)} 行数据")
        self.table_widget.setRowCount(df.shape[0])
        self.table_widget.setColumnCount(df.shape[1])
        self.table_widget.setHorizontalHeaderLabels(df.columns.tolist() if not df.empty else [])
        # 批量设置数据
        for i in range(df.shape[0]):
            row = df.iloc[i].astype(str)
            for j in range(df.shape[1]):
                item = QTableWidgetItem(row.iloc[j])
                self.table_widget.setItem(i, j, item)
        self.adjust_column_widths()
        print("【load_table_data】加载完成")

    def adjust_column_widths(self):
        header = self.table_widget.horizontalHeader()
        for col in range(self.table_widget.columnCount()):
            max_length = 0
            for row in range(self.table_widget.rowCount()):
                item = self.table_widget.item(row, col)
                if item:
                    max_length = max(max_length, len(item.text()))
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
            width = header.sectionSize(col)
            header.setDefaultSectionSize(width)

    def clear_charts(self):
        # 清空图表区域
        for i in reversed(range(self.chart_layout.count())):
            widget = self.chart_layout.takeAt(i).widget()
            if widget:
                widget.deleteLater()
        self.image_labels.clear()

    def plot_statistics(self, ok_images, ng_images,
                        current_task_order_review_ok_boards, current_task_order_pre_ok_boards,
                        current_task_order_ng_boards, review_ok_boards, pre_ok_boards, ng_boards,
                        current_task_order_total_boards, total_boards):
        if not os.path.exists('img'):
            os.makedirs('img')
        # 清空旧图表
        for i in reversed(range(self.chart_layout.count())):
            widget = self.chart_layout.takeAt(i).widget()
            if widget:
                widget.deleteLater()
        self.image_labels.clear()
        # 如果所有数据都为 0，则不绘制图表（由 on_query 弹出提示）
        has_data = any([
            ok_images > 0, ng_images > 0,
            current_task_order_review_ok_boards > 0, current_task_order_pre_ok_boards > 0, current_task_order_ng_boards > 0,
            review_ok_boards > 0, pre_ok_boards > 0, ng_boards > 0
        ])
        if not has_data:
            return  # 不显示“暂无数据”，由 on_query 统一处理
        # 正常绘制图表
        self.draw_pie_chart(['OK', 'NG'], [ok_images, ng_images], ['#38BDF8', '#F87171'], '复判图片')
        self.draw_pie_chart(['复判后OK板', 'OK板', 'NG板'],
                            [current_task_order_review_ok_boards, current_task_order_pre_ok_boards,
                            current_task_order_ng_boards],
                            ['#60A5FA', '#93C5FD', '#F87171'], '当前任务令单板')
        self.draw_pie_chart(['复判后OK板', 'OK板', 'NG板'],
                            [review_ok_boards, pre_ok_boards, ng_boards],
                            ['#60A5FA', '#7DDA58', '#F87171'], '总计单板')
        self.show_pass_rate_text(
            current_task_order_pre_ok_boards, current_task_order_review_ok_boards,
            current_task_order_total_boards, pre_ok_boards, review_ok_boards, total_boards
        )

    def draw_pie_chart(self, labels, values, colors, title):
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # 过滤掉值为 0 的项
        filtered_indices = [i for i, v in enumerate(values) if v > 0]
        if not filtered_indices:
            ax.axis('off')
            ax.text(0.5, 0.6, '无数据', ha='center', va='center', fontsize=10, fontweight='bold', color='#9ca3af')
            ax.set_title(title, fontsize=10, color='#1F2937', pad=5, fontweight='bold')
            plt.close(fig)
            chart_path = f'img/chart_{len(self.image_labels)}.png'
            plt.savefig(chart_path, dpi=150, bbox_inches='tight', transparent=False)
            return

        filtered_labels = [labels[i] for i in filtered_indices]
        filtered_values = [values[i] for i in filtered_indices]
        filtered_colors = [colors[i] for i in filtered_indices]

        # 绘制饼图，不显示 autopct
        wedges, texts = ax.pie(
            filtered_values,
            labels=filtered_labels,
            colors=filtered_colors,
            startangle=90,
            labeldistance=1.1,
            pctdistance=0.75,
            wedgeprops=dict(width=0.4),
            textprops={'fontsize': 8, 'fontweight': 'bold'}
        )

        # 手动添加带换行的 autopct 文本
        autotexts = []
        for i, (wedge, value) in enumerate(zip(wedges, filtered_values)):
            angle = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
            x = wedge.r * 0.5 * np.cos(np.radians(angle))
            y = wedge.r * 0.5 * np.sin(np.radians(angle))

            total = sum(filtered_values)
            percent = 100. * value / total
            number = value
            percent_str = f"{percent:.1f}%"
            number_str = str(number)

            # 第一行：数量
            t1 = ax.text(x, y, number_str, ha='center', va='center',
                        fontsize=8, fontweight='bold', color='#120CA8')
            # 第二行：百分比（略低一点）
            t2 = ax.text(x, y - 0.2, percent_str, ha='center', va='center',
                        fontsize=8, fontweight='bold', color='#FE9900')
            autotexts.append((t1, t2))

        # 设置标签字体样式
        for text in texts:
            text.set_color('#374151')
            text.set_fontsize(9)
            text.set_fontweight('bold')

        ax.set_title(title, fontsize=10, color='#1F2937', pad=5, fontweight='bold')
        ax.axis('equal')

        # 去掉坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, bottom=False)

        plt.tight_layout()
        chart_path = f'img/chart_{len(self.image_labels)}.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', transparent=False)
        plt.close(fig)

        # 创建带卡片样式的 QLabel 容器
        label_container = QWidget()
        label_container.setStyleSheet("""
            background-color: white;
            border-radius: 12px;
            padding: 10px;
            border: 1px solid #e5e7eb;
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QtGui.QColor(0, 0, 0, 30))
        shadow.setOffset(0, 2)
        label_container.setGraphicsEffect(shadow)

        layout = QVBoxLayout(label_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        pixmap = QPixmap(chart_path).scaled(400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(image_label)

        self.chart_layout.addWidget(label_container)
        self.image_labels.append(label_container)


    def show_pass_rate_text(self, task_pre, task_review, task_total, all_pre, all_review, all_total):
        print(f"Params: {task_pre}, {task_review}, {task_total}, {all_pre}, {all_review}, {all_total}")
        def calc_rate(a, b, t):
            return ((a + b) / t * 100) if t > 0 else 0
        def get_fraction(a, b, t):
            return f"{int(a + b)}/{int(t)}"
        task_before = (task_pre / task_total * 100) if task_total > 0 else 0
        task_after = calc_rate(task_pre, task_review, task_total)
        task_fraction_before = f"{int(task_pre)}/{int(task_total)}"
        task_fraction_after = get_fraction(task_pre, task_review, task_total)
        all_before = (all_pre / all_total * 100) if all_total > 0 else 0
        all_after = calc_rate(all_pre, all_review, all_total)
        all_fraction_before = f"{int(all_pre)}/{int(all_total)}"
        all_fraction_after = get_fraction(all_pre, all_review, all_total)
        def get_improvement(before, after):
            if before == 0 or before is None:
                return "0%"
            improvement = ((after - before) / before * 100)
            return f"{improvement:.1f}%"
        html = """
        <div style="border:1px solid #d1d5db; padding:1.25rem; border-radius:0.5rem; background:#f9fafb; box-shadow:0 2px 4px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin-top:0; font-size:1.1rem; color:#1e3a8a; font-weight:600;">📊 直通率分析</h3>
                <small style="color:#94a3b8;">数据统计至：{}</small>
            </div>
            <ul style="list-style:none; padding-left:0; margin-bottom:0;">
                <li style="margin-bottom:0.75rem;">
                    <span style="color:#64748b; font-weight:500;">任务令复判前直通率：</span>
                    <span style="color:#ea580c; font-weight:600;">{:.1f}%</span>
                    <small style="color:#94a3b8; font-size:0.75rem;">({})</small>
                </li>
                <li style="margin-bottom:0.75rem;">
                    <span style="color:#64748b; font-weight:500;">任务令复判后直通率：</span>
                    <span style="color:#060270; font-weight:600;">{:.1f}%</span>
                    <small style="color:#94a3b8; font-size:0.75rem;">({})</small>
                    <span style="color:#16a34a; font-weight:600; font-size: 0.75rem;">↑ {}%</span>
                </li>
                <li style="margin-bottom:0.75rem;">
                    <span style="color:#64748b; font-weight:500;">总复判前直通率：&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                    <span style="color:#ea580c; font-weight:600;">{:.1f}%</span>
                    <small style="color:#94a3b8; font-size:0.75rem;">({})</small>
                </li>
                <li style="margin-bottom:0;">
                    <span style="color:#64748b; font-weight:500;">总复判后直通率：&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                    <span style="color:#060270; font-weight:600;">{:.1f}%</span>
                    <small style="color:#94a3b8; font-size:0.75rem;">({})</small>
                    <span style="color:#16a34a; font-weight:600; font-size: 0.75rem;">↑ {}%</span>
                </li>
            </ul>
        </div>
        """.format(
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            task_before,
            task_fraction_before,
            task_after,
            task_fraction_after,
            get_improvement(task_before, task_after),
            all_before,
            all_fraction_before,
            all_after,
            all_fraction_after,
            get_improvement(all_before, all_after)
        )
        print("show_pass_rate_text")
        self.pass_rate_label.setText(html)

    def analyze_data(self, df=None):
        print("【analyze_data】开始分析数据")
        df = df if df is not None else pd.DataFrame()
        print(f"【analyze_data】传入数据行数: {len(df)}")
        # 获取当前任务令（默认最后一个）
        current_task_order = ""
        if not df.empty and '任务令' in df.columns:
            try:
                current_task_order = str(df['任务令'].iloc[-1]).strip()
                print(f"【analyze_data】当前任务令: {current_task_order}")
            except Exception as e:
                print(f"【analyze_data】获取任务令失败: {e}")
        else:
            print("【analyze_data】任务令列不存在或数据为空")
        # 统计图片 OK/NG 数量
        ok_images = ng_images = 0
        if '结果' in df.columns:
            ok_images = df[df['结果'] == 'OK'].shape[0]
            ng_images = df[df['结果'] == 'NG'].shape[0]
            print(f"【analyze_data】图片统计 - OK: {ok_images}, NG: {ng_images}")
        else:
            print("【analyze_data】缺少‘结果’列")
        # 按“单板条码”分组统计每块板的结果
        board_grouped = pd.DataFrame()
        if '单板条码' in df.columns and '结果' in df.columns:
            try:
                board_grouped = df.groupby('单板条码')['结果'].apply(list).reset_index(name='results_list')
                print(f"【analyze_data】分组统计完成，共 {len(board_grouped)} 块单板")
            except Exception as e:
                print(f"【analyze_data】分组统计失败: {e}")
        else:
            print("【analyze_data】缺少‘单板条码’或‘结果’列")
        def classify_board(results):
            has_none = 'NONE' in results
            all_ok = all(r == 'OK' for r in results)
            return {'all_ok': all_ok, 'has_none': has_none}
        review_ok_boards = pre_ok_boards = ng_boards = 0
        total_boards = len(board_grouped)
        if not board_grouped.empty:
            try:
                board_grouped['status'] = board_grouped['results_list'].apply(classify_board)
                review_ok_boards = board_grouped[board_grouped['status'].apply(lambda x: x['all_ok'])].shape[0]
                pre_ok_boards = board_grouped[
                    board_grouped['status'].apply(lambda x: not x['all_ok'] and x['has_none'])
                ].shape[0]
                ng_boards = board_grouped[
                    board_grouped['status'].apply(lambda x: not x['all_ok'] and not x['has_none'])
                ].shape[0]
                print(f"【analyze_data】板级统计完成 - OK: {review_ok_boards}, Pre-OK: {pre_ok_boards}, NG: {ng_boards}")
            except Exception as e:
                print(f"【analyze_data】板级分类失败: {e}")
        # 当前任务令的数据
        current_review_ok_boards = current_pre_ok_boards = current_ng_boards = 0
        current_ok_images = current_ng_images = 0
        current_total_boards = 0
        if current_task_order and '任务令' in df.columns:
            try:
                current_df = df[df['任务令'] == current_task_order]
                current_total_boards = current_df['单板条码'].nunique()
                current_ok_images = current_df[current_df['结果'] == 'OK'].shape[0]
                current_ng_images = current_df[current_df['结果'] == 'NG'].shape[0]
                current_board_grouped = current_df.groupby('单板条码')['结果'].apply(list).reset_index(name='results_list')
                current_board_grouped['status'] = current_board_grouped['results_list'].apply(classify_board)
                current_review_ok_boards = current_board_grouped[
                    current_board_grouped['status'].apply(lambda x: x['all_ok'])
                ].shape[0]
                current_pre_ok_boards = current_board_grouped[
                    current_board_grouped['status'].apply(lambda x: not x['all_ok'] and x['has_none'])
                ].shape[0]
                current_ng_boards = current_board_grouped[
                    current_board_grouped['status'].apply(lambda x: not x['all_ok'] and not x['has_none'])
                ].shape[0]
                print(f"【analyze_data】当前任务令统计完成 - OK: {current_review_ok_boards}, Pre-OK: {current_pre_ok_boards}, NG: {current_ng_boards}")
            except Exception as e:
                print(f"【analyze_data】任务令子集分析失败: {e}")
        else:
            print("【analyze_data】跳过任务令统计")
        return (
            review_ok_boards, pre_ok_boards, ng_boards,
            ok_images, ng_images, total_boards,
            current_review_ok_boards, current_pre_ok_boards, current_ng_boards,
            current_ok_images, current_ng_images, current_total_boards
        )

class PyQtMainEntry(QMainWindow, Ui_mainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.error_dialog_is_visible = False
        self.last_error_message = None
        # 初始化路径参数
        self.imagePath, self.resPath = '', ''
        self.okRange = 0.5
        self.collect = 1
        self.device = "神州"
        # 日志系统初始化
        self.logTextBox.setReadOnly(True)
        self.max_log_lines = 300
        self.logFilePath = './check.log'
        self.last_position = 0
        self.is_scrolled_to_bottom = True
        self.logTextBox.verticalScrollBar().valueChanged.connect(self.on_scroll_value_changed)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_log)
        self.timer.start(200)
        # 加载配置
        self.configFilePath = 'config.txt'
        if not os.path.exists(self.configFilePath):
            self.saveConfig()
        self.imagePath, self.resPath, self.okRange, self.collect, self.device = self.readConfig(self.configFilePath)
        # 初始化状态提示
        self.state.setText("欢迎使用AOI智能复判系统")
        # GIF 动画显示
        self.movie = QMovie("img/working.gif")
        self.gifView.setMovie(self.movie)
        gif_size = self.gifView.size()
        self.movie.setScaledSize(gif_size)
        self.movie.start()
        self.movie.stop()
        # 菜单事件绑定
        self.actionSettings = QAction("系统参数设置", self)
        self.actionSettings.triggered.connect(self.open_settings_dialog)
        self.menuSystem.addAction(self.actionSettings)
        self.actionDataQuery = QAction("复判记录查询", self)
        self.actionDataQuery.triggered.connect(self.open_data_query_dialog)
        self.menuDataQuery.addAction(self.actionDataQuery)
        # 添加清理系统日志菜单项
        self.actionClearLogs = QAction("清理系统日志", self)
        self.actionClearLogs.triggered.connect(self.clear_log_files)
        self.menuSystem.addAction(self.actionClearLogs)
        # 按钮点击事件绑定
        self.btnStartCheck.clicked.connect(self.btnStartCheckClk)
        self.btnStopCheck.clicked.connect(self.btnStopCheckClk)
        # 图片显示相关
        self.displayFolderPath = 'display'
        if os.path.exists(self.displayFolderPath):
            [shutil.rmtree(p) if p.is_dir() else p.unlink() for p in Path(self.displayFolderPath).iterdir()]
        self.lastImageFile = ''
        self.imageTimer = QTimer(self)
        self.imageTimer.timeout.connect(self.update_display_image)
        self.imageTimer.start(200)
        self.dataTimer = QTimer(self)
        self.dataTimer.timeout.connect(self.ShowResult)
        self.show_default_image()
        # 对所有需要增强的 QLabel 应用该样式
        for widget in self.findChildren(QLabel):
            self.apply_label_style(widget)
        for widget in self.findChildren(QPushButton):
            self.apply_label_style(widget)
        for widget in self.findChildren(QTextEdit):
            self.apply_label_style(widget)
        logging.info("软件启动")


    def apply_label_style(self, label: QLabel):

        # 添加轻微阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(6)
        shadow.setColor(Qt.black)
        shadow.setOffset(0, 1)
        label.setGraphicsEffect(shadow)


    def custom_confirm_dialog(self, title, text):
        """
        自定义无背景的确认对话框，按钮为圆角矩形 + 浅蓝/浅红风格
        """
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)

        yes_button = msg.addButton("✅ 是", QMessageBox.YesRole)
        no_button = msg.addButton("❌ 否", QMessageBox.NoRole)

        for child in msg.findChildren(QWidget):
            child.setStyleSheet("""
                background-color: transparent;
                border: none;
                border-radius: 0px;
                padding: 0px;
                margin: 0px;
            """)

        button_style_yes = """
            QPushButton {
                min-width: 80px;
                min-height: 30px;
                font-size: 14px;
                color: white;
                background-color: #74A2DB;   /* 浅蓝色 */
                border: none;
                border-radius: 8px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #638BBD;
            }
            QPushButton:pressed {
                background-color: #4F719A;
            }
        """
        button_style_no = """
            QPushButton {
                min-width: 80px;
                min-height: 30px;
                font-size: 14px;
                color: white;
                background-color: #F0696A;   /* 浅红色 */
                border: none;
                border-radius: 8px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #CA5D5D;
            }
            QPushButton:pressed {
                background-color: #A94B4C;
            }
        """

        yes_button.setStyleSheet(button_style_yes)
        no_button.setStyleSheet(button_style_no)

        msg.exec_()
        return msg.clickedButton() == yes_button

    def show_info_dialog(self, title, message):
        """
        显示一个统一风格的信息提示对话框，只有一个“确定”按钮。
        :param title: 弹窗标题
        :param message: 提示内容
        """
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(message)

        ok_button = msg.addButton("✔ 确定", QMessageBox.AcceptRole)

        # 去除所有子控件默认背景和边框
        for child in msg.findChildren(QWidget):
            child.setStyleSheet("""
                background-color: transparent;
                border: none;
                border-radius: 0px;
                padding: 0px;
                margin: 0px;
            """)

        # 设置按钮样式
        button_style_ok = """
            QPushButton {
                min-width: 80px;
                min-height: 30px;
                font-size: 14px;
                color: white;
                background-color: #95CC5C;   /* 浅绿色 */
                border: none;
                border-radius: 8px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #81AF4F;
            }
            QPushButton:pressed {
                background-color: #73A042;
            }
        """

        ok_button.setStyleSheet(button_style_ok)
        msg.exec_()

    def clear_log_files(self):
        clicked = self.custom_confirm_dialog(
            "确认清理",
            "确定要删除所有归档的日志文件吗？此操作不可恢复！"
        )
        if clicked:
            current_path = os.getcwd()
            log_files = [f for f in os.listdir(current_path) if f.startswith("check_") and f.endswith(".log")]
            deleted_count = 0
            for file in log_files:
                try:
                    os.remove(os.path.join(current_path, file))
                    deleted_count += 1
                except Exception as e:
                    logging.warning(f"无法删除日志文件 {file}: {e}")
            if deleted_count > 0:
                self.show_info_dialog("清理完成", f"删除了{deleted_count}个日志文件")
            else:
                self.show_info_dialog("清理完成", "没有找到符合条件的日志文件。")


    def closeEvent(self, event):
        clicked = self.custom_confirm_dialog('确认退出', "退出复判程序前请找技术员进行确认！！！")
        if clicked:
            # 1. 停止后台进程（确保子进程释放文件句柄）
            if hasattr(self, 'pool'):
                self.pool.terminate()
                self.pool.join() # 等待子进程完全结束
            
            # 2. 日志归档逻辑
            try:
                # 显式关闭日志系统，释放 check.log 的文件占用
                logging.shutdown()
                
                log_file = 'check.log'
                if os.path.exists(log_file):
                    has_error = False
                    try:
                        # 读取日志内容检测是否存在 ERROR
                        with open(log_file, 'r', encoding='gbk', errors='ignore') as f:
                            content = f.read()
                            if "ERROR" in content:
                                has_error = True
                    except Exception:
                        pass # 读取失败则忽略

                    if has_error:
                        # 生成带时间戳的归档文件名
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        archive_name = f"check_{timestamp}_ERROR.log"
                        
                        # 重命名文件（实现归档）
                        try:
                            shutil.move(log_file, archive_name)
                            print(f"【系统】检测到错误日志，已归档为: {archive_name}")
                        except Exception as e:
                            print(f"【系统】日志归档失败: {e}")
                    else:
                        # 如果没有错误，可以选择保留或删除，这里选择保留（方便查看历史INFO），但因为没有ERROR，下次启动不会弹窗
                        pass

            except Exception as e:
                print(f"【系统】退出清理逻辑出错: {e}")
            
            event.accept()
        else:
            event.ignore()

    def select_image_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片路径")
        if path:
            self.ImagePath.setText(path)
            self.imagePath = path

    def select_result_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择结果路径")
        if path:
            self.ResultPath.setText(path)
            self.resPath = path

    def set_stop_button_active(self):
        self.btnStopCheck.setStyleSheet("""
            QPushButton {
                background-color: #DD2442;
                color: white;
                border-radius: 14px;
                font-family: "Microsoft YaHei";
                font-size: 18px;
                font-weight: 500;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #E64B65;
            }
            QPushButton:pressed {
                background-color: #E64B65;
            }
        """)

    def set_start_button_active(self):
        self.btnStartCheck.setStyleSheet("""
            QPushButton {
                background-color: #3631CE;
                color: white;
                border-radius: 14px;
                font-family: "Microsoft YaHei";
                font-size: 18px;
                font-weight: 500;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #4D7FFF;
            }
            QPushButton:pressed {
                background-color: #4D7FFF;
            }
        """)

    def btnStartCheckClk(self):
        if not (os.path.exists(self.imagePath) and os.path.exists(self.resPath)):
            self.state.setText("路径不存在，请检查路径")
            logging.error("路径不存在，请检查路径")
            return

        # 设置按钮状态
        self.set_start_button_active()
        grey_style = """
            QPushButton {
                background-color: #B0B0B0;
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
        """
        self.btnStopCheck.setStyleSheet(grey_style)
        
        # 启动任务
        self.pool = multiprocessing.Pool(processes=1)
        try:
            device_module = get_device_module(self.device)
            logging.info(f"当前对接设备类型:{self.device}")
            
            self.pool.apply_async(device_module.process_all_files, (1, self.imagePath, self.resPath))
        except ImportError as e:
            logging.error(str(e))
            QMessageBox.critical(self, "错误", str(e))
        self.state.setText("已开始复判")
        self.movie.start()
        self.dataTimer.start(1000)
        logging.info("已开始复判")


    def btnStopCheckClk(self):
        if not hasattr(self, 'pool'):
            self.state.setText("请先启动复判")
            logging.info("请先启动复判")
            return

        self.pool.terminate()

        # 设置按钮状态
        self.set_stop_button_active()
        grey_style = """
            QPushButton {
                background-color: #B0B0B0;
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
        """
        self.btnStartCheck.setStyleSheet(grey_style)

        self.state.setText("已停止复判")
        self.movie.stop()
        self.dataTimer.stop()
        self.show_default_image()
        logging.info("已停止复判")

    def ShowResult(self):
        csv_path = 'history.csv'
        total_ok = 0
        total_ng = 0
        
        # 默认显示内容
        filter_ratio = 0.0
        display_text = "等待数据..."

        try:
            if os.path.exists(csv_path):
                # 使用 pandas 读取 CSV
                try:
                    df = pd.read_csv(csv_path, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    df = pd.read_csv(csv_path, encoding='gbk', errors='ignore')

                if not df.empty and '结果' in df.columns:
                    # --- 直接统计所有数据 ---
                    
                    # 统计 OK 和 NG (不筛选日期)
                    total_ok = len(df[df['结果'] == 'OK'])
                    total_ng = len(df[df['结果'] == 'NG'])
                    
                    total_count = total_ok + total_ng
                    if total_count > 0:
                        filter_ratio = (total_ok / total_count) * 100
                    
                    display_text = "📅 历史总数据统计"
                else:
                    display_text = "历史记录为空或格式错误"
            else:
                display_text = "历史记录文件未找到"

            # 生成 HTML 显示内容
            html_content = f"""
            <div style="font-family: Microsoft YaHei, sans-serif; font-size: 14px; color: #003366;">
                <p style="margin: 6px 0; color: #555555; font-size: 12px;">{display_text}</p>
                <p style="margin: 6px 0; color: #003366;"><b>📊 累计已复判：</b> <span style="color: #000000;">{total_ok + total_ng}</span> 张</p>
                <p style="margin: 6px 0; color: #003366;"><b>✅ 累计复判 OK 数量： </b> <span style="color: #2E7D32; font-weight:bold; font-size: 16px;">{total_ok}</span></p>
                <p style="margin: 6px 0; color: #003366;"><b>❌ 累计复判 NG 数量： </b> <span style="color: #C62828; font-weight:bold; font-size: 16px;">{total_ng}</span></p>
                <p style="margin: 6px 0; color: #003366;"><b>📈 累计复判率： </b> <span style="color: #00838F; font-weight:bold;">{filter_ratio:.2f}%</span></p>
                <p style="margin: 6px 0; color: #888888; font-size: 10px;">🕒 更新时间：{datetime.now().strftime("%H:%M:%S")}</p>
            </div>
            """
            self.dataStats.setHtml(html_content)

        except Exception as e:
            logging.error(f"统计刷新失败: {str(e)}")
            self.state.setText(f"统计刷新出错: {str(e)}")

    def open_settings_dialog(self):
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            settings = dialog.get_settings()
            self.imagePath = settings['ImagePath']
            self.resPath = settings['ResultPath']
            self.okRange = float(settings['okRange'])
            self.collect = 1 if settings['collect'] == '开启' else 0
            self.device = settings['Device']
            self.saveConfig()

    def open_data_query_dialog(self):
        dialog = DataQueryDialog(self)
        dialog.exec_()

    def show_default_image(self):
        """显示默认的欢迎/等待图片"""
        try:
            img_path = "img/init.png"
            if not os.path.exists(img_path):
                # 如果没有图片，清空显示
                self.imageView.clear()
                self.imageView.setText("等待图片...")
                return

            # 读取图片 (保留透明通道)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return

            # 获取 QLabel 实际显示尺寸
            original_size = self.imageView.size()
            target_width = original_size.width() - 60
            target_height = original_size.height() - 60
            
            if target_width <= 0 or target_height <= 0:
                return

            # 等比缩放
            h, w = img.shape[:2]
            scale = min(target_width / w, target_height / h)
            new_size = (int(w * scale), int(h * scale))
            resized_img = cv2.resize(img, new_size)

            # 转换为 QImage
            if len(resized_img.shape) == 3 and resized_img.shape[2] == 4:
                q_img = QImage(resized_img.data, resized_img.shape[1], resized_img.shape[0],
                               resized_img.strides[0], QImage.Format_RGBA8888)
            elif len(resized_img.shape) == 3 and resized_img.shape[2] == 3:
                q_img = QImage(resized_img.data, resized_img.shape[1], resized_img.shape[0],
                               resized_img.strides[0], QImage.Format_BGR888)
            else:
                q_img = QImage(resized_img.data, resized_img.shape[1], resized_img.shape[0],
                               resized_img.strides[0], QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_img)
            self.imageView.setPixmap(pixmap.scaled(
                target_width, target_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.lastImageFile = "" # 重置最后图片记录
            
        except Exception as e:
            logging.error(f"显示默认图片失败: {e}")

    def update_display_image(self):
        # 检查 display 文件夹是否存在
        if not os.path.exists(self.displayFolderPath):
            return

        # 获取 display 文件夹中的所有图像文件
        files = [f for f in os.listdir(self.displayFolderPath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # === 核心修改点 ===
        # 如果文件夹为空，直接返回，保持当前 QLabel 上的图片不变
        if not files:
            return 
        # ================

        # 处理文件夹中的图片
        pattern = re.compile(r'(.+)_(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)
        valid_files = []
        for f in files:
            match = pattern.match(f)
            if match:
                base_name, count_str, ext = match.groups()
                count = int(count_str)
                valid_files.append((f, base_name.upper(), count))

        if not valid_files:
            return

        # 按计数排序，获取最早的文件
        valid_files.sort(key=lambda x: x[2])
        earliest_file = valid_files[0][0]

        # 如果文件与上次相同，则不更新
        if earliest_file == self.lastImageFile:
            return

        img_path = os.path.join(self.displayFolderPath, earliest_file)
        original_size = self.imageView.size()

        try:
            # 读取并显示图片逻辑 (与之前保持一致)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                logging.error(f"无法读取图片: {img_path}")
                return

            target_width = original_size.width() - 60
            target_height = original_size.height() - 60
            
            if target_width <= 0 or target_height <= 0:
                return

            h, w = img.shape[:2]
            scale = min(target_width / w, target_height / h)
            new_size = (int(w * scale), int(h * scale))
            resized_img = cv2.resize(img, new_size)

            # 获取基础名用于显示标签
            match = pattern.match(earliest_file)
            if match:
                base_name, count_str, ext = match.groups()
            else:
                base_name = "IMAGE"

            # 绘制文字
            if "OK" in base_name.upper():
                color = (0, 255, 0)
            elif "NG" in base_name.upper():
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)

            cv2.putText(resized_img, base_name.upper(), (resized_img.shape[1] - 80, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

            # 转 QImage
            if len(resized_img.shape) == 3 and resized_img.shape[2] == 4:
                q_img = QImage(resized_img.data, resized_img.shape[1], resized_img.shape[0],
                            resized_img.strides[0], QImage.Format_RGBA8888)
            elif len(resized_img.shape) == 3 and resized_img.shape[2] == 3:
                q_img = QImage(resized_img.data, resized_img.shape[1], resized_img.shape[0],
                            resized_img.strides[0], QImage.Format_BGR888)
            else:
                q_img = QImage(resized_img.data, resized_img.shape[1], resized_img.shape[0],
                            resized_img.strides[0], QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_img)
            self.imageView.setPixmap(pixmap.scaled(
                target_width, target_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

            self.lastImageFile = earliest_file

            # 删除旧图片
            current_count = [item[2] for item in valid_files if item[0] == earliest_file][0]
            for f, _, count in valid_files:
                if count <= current_count:
                    try:
                        os.remove(os.path.join(self.displayFolderPath, f))
                    except Exception as e:
                        logging.warning(f"无法删除旧图片 {f}: {e}")

            # 允许后续显示相同计数的新图片
            QTimer.singleShot(500, lambda: setattr(self, 'lastImageFile', ''))

        except Exception as e:
            logging.error(f"显示图片失败: {e}")

    def loadLogFile(self):
        try:
            with open(self.logFilePath, 'r', encoding='gbk') as file:
                file.seek(self.last_position)
                new_content = file.read()
                if new_content:
                    self.appendLog(new_content)
                    self.last_position = file.tell()
        except FileNotFoundError:
            self.appendLog("无法找到日志文件。\n")
    
    def show_custom_error_message(self, error_line):
        """
        显示自定义错误弹窗。
        - 增加并发锁，防止重复打开。
        - 增加内容判断，如果错误内容和上次相同，则不弹窗。
        - 在弹窗中显示具体的错误信息。
        """
        # 规则1：如果一个错误弹窗已经在显示，则不处理
        if self.error_dialog_is_visible:
            return

        # 规则2：如果本次错误内容和上一次弹窗的错误内容相同，则不处理
        if error_line == self.last_error_message:
            return

        try:
            # 上锁，防止并发
            self.error_dialog_is_visible = True
            
            # 记录下本次的错误信息，用于下一次比较
            self.last_error_message = error_line

            # --- 创建并显示一个内容更丰富的弹窗 ---
            msg_box = QMessageBox()
            msg_box.setWindowTitle("错误提示")
            
            # 对错误文本进行简单的HTML转义，防止内容里有特殊字符扰乱格式
            safe_error_text = error_line.replace("<", "&lt;").replace(">", "&gt;")

            # 设置带有具体错误内容的HTML文本
            msg_box.setText(f"""
                <h3 style='color: red; font-weight: bold; font-size: 16px;'>⚠️ 检测到新的ERROR</h3>
                <p style='font-size: 12px; color: #333;'>请立即检查日志中的以下错误：</p>
                <pre style='background-color:#F5F5F5; border-radius:4px; padding:8px; font-size:11px; color:black; white-space: pre-wrap;'>{safe_error_text}</pre>
            """)
            msg_box.setTextFormat(Qt.RichText)
            msg_box.exec_()
            
        finally:
            # 解锁，让下一次的弹窗可以正常显示
            self.error_dialog_is_visible = False

    def appendLog(self, content):
        lines = content.splitlines()
        doc = self.logTextBox.document()
        cursor = QtGui.QTextCursor(doc)
        cursor.movePosition(QtGui.QTextCursor.End)

        for line in lines:
            # 判断是否是 ERROR 行
            is_error = "ERROR" in line

            # 设置文本块格式（控制背景色）
            block_format = QtGui.QTextBlockFormat()
            if is_error:
                block_format.setBackground(QtGui.QColor("#D32F2F"))  # 红色背景
                QTimer.singleShot(0, lambda line=line: self.show_custom_error_message(line))

            # 设置字符格式（控制文字颜色、大小等）
            char_format = QtGui.QTextCharFormat()
            if is_error:
                char_format.setForeground(QtGui.QColor("#FFFFFF"))  # 白色文字
                char_format.setFontWeight(QtGui.QFont.Bold)
                char_format.setFontPointSize(12)  # 可选：加大字体
            else:
                char_format.setForeground(QtGui.QColor("#003366"))  # 深蓝色
                char_format.setFontWeight(QtGui.QFont.Bold)
                char_format.setFontPointSize(11)

            # 插入带格式的文本块
            cursor.insertBlock(block_format)
            cursor.setCharFormat(char_format)

            # 如果是普通行，再按 OK/NG 高亮关键词
            words = line.split(' ')
            if not is_error:
                for i, word in enumerate(words):
                    highlight_format = QtGui.QTextCharFormat(char_format)  # 复制基础格式

                    if word == 'OK':
                        highlight_format.setForeground(QtGui.QColor("#2E7D32"))  # 深绿色
                        highlight_format.setFontPointSize(14)
                        highlight_format.setFontWeight(QtGui.QFont.Bold)
                    elif word == 'NG':
                        highlight_format.setForeground(QtGui.QColor("#C62828"))  # 深红色
                        highlight_format.setFontPointSize(14)
                        highlight_format.setFontWeight(QtGui.QFont.Bold)

                    cursor.setCharFormat(highlight_format)
                    cursor.insertText(word + ' ')
            else:
                cursor.insertText(line)

            cursor.insertBlock()

        # 控制最大行数
        while doc.lineCount() > self.max_log_lines:
            cursor.movePosition(QtGui.QTextCursor.Start)
            cursor.select(QtGui.QTextCursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

        if self.is_scrolled_to_bottom:
            QTimer.singleShot(0, self.scrollToBottom)

    def scrollToBottom(self):
        scrollbar = self.logTextBox.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_scroll_value_changed(self, value):
        scrollbar = self.logTextBox.verticalScrollBar()
        self.is_scrolled_to_bottom = (value == scrollbar.maximum())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.is_scrolled_to_bottom:
            self.scrollToBottom()

    def update_log(self):
        self.loadLogFile()

    def readConfig(self, path):
        if not os.path.exists(path):
            QMessageBox.critical(self, "错误", f"找不到配置文件: {path}")
            return "", "", "0.9", "1", "神州"
        config = {}
        try:
            with open(path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        config[key] = value
        except Exception as e:
            logging.error(f"读取配置文件失败: {e}")
            return "", "", "0.9", "1", "神州"
        imagePath = config.get('ImagePath', '')
        resultPath = config.get('ResultPath', '')
        okRange = config.get('okRange', '0.9')
        collect = config.get('collect', '1')
        device = config.get('Device', '神州')
        return imagePath, resultPath, okRange, collect, device

    def saveConfig(self):
        fields_to_save = {
            "ImagePath": self.imagePath,
            "ResultPath": self.resPath,
            "okRange": str(self.okRange),
            "collect": str(self.collect),
            "Device": self.device
        }
        if not os.path.exists(self.configFilePath):
            logging.info(f"配置文件 {self.configFilePath} 不存在，正在创建新文件")
            with open(self.configFilePath, 'w') as file:
                for key, value in fields_to_save.items():
                    file.write(f"{key}={value}\n")
            return
        existing_lines = []
        updated_fields = set()
        try:
            with open(self.configFilePath, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    if '=' in line:
                        key, _ = line.split('=', 1)
                        if key in fields_to_save:
                            updated_fields.add(key)
                            existing_lines.append(f"{key}={fields_to_save[key]}\n")
                        else:
                            existing_lines.append(line + "\n")
            for key, value in fields_to_save.items():
                if key not in updated_fields:
                    existing_lines.append(f"{key}={value}\n")
            with open(self.configFilePath, 'w') as file:
                file.writelines(existing_lines)
            logging.info(f"配置文件 {self.configFilePath} 已更新")
        except Exception as e:
            logging.error(f"保存配置文件失败: {e}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon("img/icon.png"))
    window = PyQtMainEntry()
    window.setStyleSheet("""
        QMainWindow {
            background-color: #F8F8F8;
            font-family: "Microsoft YaHei", "Helvetica Neue", Helvetica, Arial;
            font-size: 16px;
        }
        QWidget {
            color: #1C1C1E;
        }
        QPushButton {
            background-color: #F1F1F1;  /* 浅灰色按钮背景 */
            border: none;
            border-radius: 14px;
            padding: 12px 24px;
            min-width: 160px;
            min-height: 48px;
            font-family: "Microsoft YaHei";
            font-size: 17px;
            font-weight: 500;
            margin: 6px;
        }
        QPushButton:hover {
            background-color: #E0E0E0;
        }
        QPushButton:pressed {
            background-color: #D3D3D3;
        }
        QLabel {
            font-family: "Microsoft YaHei";
            font-size: 16px;
            color: #1C1C1E;
        }
        QTextEdit {
            background-color: #FFFFFF;
            border: 1px solid #DFDFDF;
            border-radius: 14px;
            padding: 12px;
            font-family: Menlo, Consolas, monospace;
            font-size: 13px;
            color: #000000;
        }
        QLineEdit, QLabel {
            padding: 4px 8px;
            border-radius: 8px;
            background-color: #F1F1F1;
        }
        QMenuBar {
            background-color: #F8F8F8;
            border-bottom: 1px solid #CCCCCC;
        }
        QMenuBar::item:selected {
            background-color: #E0E0E0;
        }
        QMenu {
            background-color: #FFFFFF;
            border: 1px solid #CCCCCC;
            border-radius: 8px;
        }
        QMenu::item:selected {
            background-color: #E0E0E0;
            border-radius: 8px;
        }
        QStatusBar {
            background-color: #F8F8F8;
            border-top: 1px solid #CCCCCC;
        }
    """)
    window.show()
    sys.exit(app.exec_())
    