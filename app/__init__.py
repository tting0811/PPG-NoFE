# -*- coding: UTF-8 -*-
import app.model as model
import numpy as np
import subprocess
import json
import pandas as pd
import tempfile
import os
import math

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def postInput():
    # 取得前端傳來的數值
    data = request.get_json()
    userinfo = data['userinfo']
    IR = data['IR']
    RED = data['RED']
    GR = data['GR']

    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as f_userinfo, \
         tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as f_IR, \
         tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as f_RED, \
         tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as f_GR:
        
        json.dump(userinfo, f_userinfo)
        json.dump(IR, f_IR)
        json.dump(RED, f_RED)
        json.dump(GR, f_GR)
        
        userinfo_path = f_userinfo.name
        IR_path = f_IR.name
        RED_path = f_RED.name
        GR_path = f_GR.name

    completed_process = subprocess.run(
        ["python", "DataPreprocessing-Predict.py", userinfo_path, IR_path, RED_path, GR_path],
        capture_output=True,
        text=True,
        check=True
    )
    
    output = completed_process.stdout.splitlines()
    IR_output_path = output[-3]
    RED_output_path = output[-2]
    GR_output_path = output[-1]

    with open(IR_output_path, 'r') as f:
        Inputs_IR = json.load(f)
    with open(RED_output_path, 'r') as f:
        Inputs_RED = json.load(f)
    with open(GR_output_path, 'r') as f:
        Inputs_GR = json.load(f)
        
    os.remove(userinfo_path)
    os.remove(IR_path)
    os.remove(RED_path)
    os.remove(GR_path)
    os.remove(IR_output_path)
    os.remove(RED_output_path)
    os.remove(GR_output_path)

    # 將列表轉換為 Numpy 陣列
    Inputs_IR = np.array(Inputs_IR)
    Inputs_RED = np.array(Inputs_RED)
    Inputs_GR = np.array(Inputs_GR)

    # 計算HR
    if len(Inputs_IR) <= 5:
        HR = 0
        Blood_Glucose_Level = 0
        print("Fewer than 5 reliable pulses.")
    else:
        HR = format(200/np.mean(Inputs_IR[:, 0])*60, ".2f")

        # 轉換為ML輸入形式
        df_IR = pd.DataFrame(Inputs_IR)
        df_RED = pd.DataFrame(Inputs_RED)
        df_GR = pd.DataFrame(Inputs_GR)

        ppg_data = pd.concat([df_IR.iloc[:, 3:], df_RED.iloc[:, 3:], df_GR.iloc[:, 3:]], axis=1)
        ppg_array = ppg_data.to_numpy()

        ppg_array = ppg_array.reshape(-1, 3, 200)
        ppg = np.transpose(ppg_array, (0, 2, 1))

        userinfo_expanded = np.tile(userinfo, (df_IR.shape[0], 1))
        df_userinfo_expanded = pd.DataFrame(userinfo_expanded)
        ppg_info = pd.concat([df_userinfo_expanded, df_IR.iloc[:, :3], df_RED.iloc[:, 1:3], df_GR.iloc[:, 1:3]], axis=1) # [age, height, weight, bmi, gender, N, AC_IR, DC_IR, AC_RED, DC_RED, AC_GR, DC_GR]
        
        # 預測
        result = model.predict(ppg, ppg_info)
        sorted_result = sorted(result)
        lower_bound = math.floor(0.2 * len(sorted_result))
        upper_bound = math.ceil(0.8 * len(sorted_result))
        filtered_predictions = sorted_result[lower_bound:upper_bound]
        Blood_Glucose_Level = format(np.mean(filtered_predictions), ".2f")
        print(ppg_info.shape[0], " reliable pulse.")
        print("HR: ",HR, "bpm")
        print("BGL Predict: ",Blood_Glucose_Level," mg/dL")

    return jsonify({'BGL': str(Blood_Glucose_Level), 'HR': str(HR)})