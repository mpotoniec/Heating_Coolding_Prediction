import joblib
import PyQt5.QtWidgets as qtw
import pandas as pd
import keras
import os
import platform
import numpy as np

def clear_console():
    if platform.system() == 'Linux':
        os.system('clear')
    elif platform.system() == 'Windows':
        os.system('cls')

class MainWindow(qtw.QWidget):
    def __init__(self) -> None:
        super().__init__()

        #self.setGeometry(0, 0, 300, 20)
        self.setLayout(qtw.QGridLayout())

        self.scaler = None

        self.Heating_Load_Liniar_Regressor = None
        self.Cooling_Load_Liniar_Regressor = None
        self.Heating_Load_Decision_Tree_Regressor = None
        self.Cooling_Load_Decision_Tree_Regressor = None
        self.Heating_Load_Random_Forest_Regressor = None
        self.Cooling_Load_Random_Forest_Regressor = None
        self.Heating_Load_Neural_Network = None 
        self.Cooling_Load_Neural_Network = None 

        self.models_names = ['Liniar Regressor', 'Decision Tree', 'Random Forest', 'Neural Network']

        self.label1 = qtw.QLabel('Relative Compactness:')
        self.layout().addWidget(self.label1, 0,0)

        self.label2 = qtw.QLabel('Surface Area:')
        self.layout().addWidget(self.label2, 1,0)

        self.label3 = qtw.QLabel('Wall Area:')
        self.layout().addWidget(self.label3, 2,0)

        self.label4 = qtw.QLabel('Roof Area:')
        self.layout().addWidget(self.label4, 3,0)

        self.label5 = qtw.QLabel('Overall Height:')
        self.layout().addWidget(self.label5, 4,0)

        self.label6 = qtw.QLabel('Orientation:')
        self.layout().addWidget(self.label6, 5,0)

        self.label7 = qtw.QLabel('Glazing Area:')
        self.layout().addWidget(self.label7, 6,0)

        self.label8 = qtw.QLabel('Glazing Area Distribution:')
        self.layout().addWidget(self.label8, 7,0)


        self.entry1 = qtw.QLineEdit()
        self.layout().addWidget(self.entry1, 0,1)
        self.entry1.setText(str(0.98))

        self.entry2 = qtw.QLineEdit()
        self.layout().addWidget(self.entry2, 1,1)
        self.entry2.setText(str(514.5))

        self.entry3 = qtw.QLineEdit()
        self.layout().addWidget(self.entry3, 2,1)
        self.entry3.setText(str(294.0))

        self.entry4 = qtw.QLineEdit()
        self.layout().addWidget(self.entry4, 3,1)
        self.entry4.setText(str(110.25))

        self.entry5 = qtw.QLineEdit()
        self.layout().addWidget(self.entry5, 4,1)
        self.entry5.setText(str(7.0))

        self.entry6 = qtw.QLineEdit()
        self.layout().addWidget(self.entry6, 5,1)
        self.entry6.setText(str(2.0))

        self.entry7 = qtw.QLineEdit()
        self.layout().addWidget(self.entry7, 6,1)
        self.entry7.setText(str(0.0))

        self.entry8 = qtw.QLineEdit()
        self.layout().addWidget(self.entry8, 7,1)
        self.entry8.setText(str(0.0))

        self.predict_button = qtw.QPushButton('Predict', clicked = lambda: self.predict())
        self.layout().addWidget(self.predict_button, 8,0)

        self.model_choose = qtw.QComboBox()
        self.model_choose.addItems(self.models_names)
        self.layout().addWidget(self.model_choose, 8,1)

        self.label9 = qtw.QLabel('Heating Load = ')
        self.layout().addWidget(self.label9, 9,0)

        self.label10 = qtw.QLabel('Cooling Load = ')
        self.layout().addWidget(self.label10, 9,1)

        self.load_scaler()
        self.load_models()

    def load_scaler(self):
        self.scaler = joblib.load('scaler\\scaler.joblib')

    def load_models(self):

        self.Heating_Load_Liniar_Regressor = joblib.load('models\\Heating_Load_Liniar_Regressor.joblib')
        self.Cooling_Load_Liniar_Regressor = joblib.load('models\\Cooling_Load_Liniar_Regressor.joblib')
        self.Heating_Load_Decision_Tree_Regressor = joblib.load('models\\Heating_Load_Decision_Tree_Regressor.joblib')
        self.Cooling_Load_Decision_Tree_Regressor = joblib.load('models\\Cooling_Load_Decision_Tree_Regressor.joblib')
        self.Heating_Load_Random_Forest_Regressor = joblib.load('models\\Heating_Load_Random_Forest_Regressor.joblib')
        self.Cooling_Load_Random_Forest_Regressor = joblib.load('models\\Cooling_Load_Random_Forest_Regressor.joblib')
        self.Heating_Load_Neural_Network = keras.models.load_model('models\\Heating_Load_Neural_Network.h5')
        self.Cooling_Load_Neural_Network = keras.models.load_model('models\\Cooling_Load_Neural_Network.h5')

    def scale_X(self, scaler, data):
        data_columns = data.columns
        return pd.DataFrame(scaler.transform(data), columns=data_columns)

    def inverse_scaling_X(self, scaler, data):
        data_columns = data.columns
        return pd.DataFrame(scaler.inverse_transform(data), columns=data_columns)

    def scale_y(self, y_Heating_Load, y_Cooling_Load):
        y_Heating_Load = y_Heating_Load / 100
        y_Cooling_Load = y_Cooling_Load / 100

        return y_Heating_Load, y_Cooling_Load

    def inverse_y_scalling(self, y_Heating_Load, y_Cooling_Load):
        y_Heating_Load = y_Heating_Load * 100
        y_Cooling_Load = y_Cooling_Load * 100

        return y_Heating_Load, y_Cooling_Load

    def predict(self):
        data = []
        data.append(float(self.entry1.text()))
        data.append(float(self.entry2.text()))
        data.append(float(self.entry3.text()))
        data.append(float(self.entry4.text()))
        data.append(float(self.entry5.text()))
        data.append(float(self.entry6.text()))
        data.append(float(self.entry7.text()))
        data.append(float(self.entry8.text()))
        data = np.array(data).reshape(1,8)

        sample = pd.DataFrame(data,
        index=[0], 
        columns=[[
        'Relative_Compactness', 
        'Surface_Area', 
        'Wall_Area',
        'Roof_Area', 
        'Overall_Height',
        'Orientation','Glazing_Area',
        'Glazing_Area_Distribution']])
        sample = self.scale_X(self.scaler, sample)

        if self.model_choose.currentText() == 'Liniar Regressor':
            y_Heating_Load = self.Heating_Load_Liniar_Regressor.predict(sample)[0]
            y_Cooling_Load = self.Cooling_Load_Liniar_Regressor.predict(sample)[0]
            y_Heating_Load, y_Cooling_Load = self.inverse_y_scalling(y_Heating_Load, y_Cooling_Load)

        elif self.model_choose.currentText() == 'Decision Tree':
            y_Heating_Load = self.Heating_Load_Decision_Tree_Regressor.predict(sample)[0]
            y_Cooling_Load = self.Cooling_Load_Decision_Tree_Regressor.predict(sample)[0]
            y_Heating_Load, y_Cooling_Load = self.inverse_y_scalling(y_Heating_Load, y_Cooling_Load)

        elif self.model_choose.currentText() == 'Random Forest':
            y_Heating_Load = self.Heating_Load_Random_Forest_Regressor.predict(sample)[0]
            y_Cooling_Load = self.Cooling_Load_Random_Forest_Regressor.predict(sample)[0]
            y_Heating_Load, y_Cooling_Load = self.inverse_y_scalling(y_Heating_Load, y_Cooling_Load)

        elif self.model_choose.currentText() == 'Neural Network':
            y_Heating_Load = self.Heating_Load_Neural_Network.predict(sample)[0, 0]
            y_Cooling_Load = self.Cooling_Load_Neural_Network.predict(sample)[0, 0]
            y_Heating_Load, y_Cooling_Load = self.inverse_y_scalling(y_Heating_Load, y_Cooling_Load)

        self.label9.setText(f'Heating Load = {str(round(y_Heating_Load, 2))}')
        self.label10.setText(f'Cooling Load = {str(round(y_Cooling_Load, 2))}')

if __name__ == '__main__':
    clear_console()
    app = qtw.QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()
    clear_console()