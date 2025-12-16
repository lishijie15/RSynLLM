import torch
from metrics import All_Metrics
import json
import numpy as np
import os
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
import math


def compute_rarr(y_true, y_pred):
    indicator = (y_pred >= y_true).astype(np.float32)
    return indicator.mean()

def worst_n_mae(pred, gt, n):
    if pred.ndim == 2:
        pred = pred[:, None, :]
        gt = gt[:, None, :]
    elif pred.ndim != 3:
        raise ValueError("pred shape must be (B, N) or (B, T, N)")

    sample_mae = np.mean(np.abs(pred - gt), axis=(1, 2))
    batch_size = sample_mae.shape[0]
    k = max(1, math.ceil(batch_size * n / 100))
    
    worst_indices = np.argpartition(sample_mae, -k)[-k:]
    worst_mae = sample_mae[worst_indices].mean()
    return worst_mae

def test(mode, mae_thresh=None, mape_thresh=0.0):
    len_nums = 0
    y_pred_load = []
    y_true_load = []
    y_pred_gas = []
    y_true_gas = []
    y_pred_heat = []
    y_true_heat = []

    y_true_load_regionlist = []
    y_pred_load_regionlist = []
    y_true_gas_regionlist = []
    y_pred_gas_regionlist = []
    y_pred_heat_regionlist = []
    y_true_heat_regionlist = []

    index_all = 0

    # Retrieve all JSON files from a folder and sort them by filename
    file_list = sorted([filename for filename in os.listdir(folder_path) if filename.endswith(".json")])

    for idx, filename in enumerate(file_list):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        with open(file_path, "r") as file:
            data_t = json.load(file)

        for i in range(len(data_t)):
            i_data = data_t[i]
            y_load = np.array(i_data["y_load"])
            y_gas = np.array(i_data["y_pv"])
            y_heat = np.array(i_data["y_wind"])
            st_pre_load = np.array(i_data["st_pre_load"])
            st_pre_gas = np.array(i_data["st_pre_pv"])
            st_pre_heat = np.array(i_data["st_pre_wind"])

            i4data_all = int(data_t[i]["id"].split('_')[6])
            if index_all != i4data_all :
                len_nums = len_nums + 1
                y_true_load_region = np.stack(y_true_load, axis=-1)
                y_pred_load_region = np.stack(y_pred_load, axis=-1)
                y_true_gas_region = np.stack(y_true_gas, axis=-1)
                y_pred_gas_region = np.stack(y_pred_gas, axis=-1)
                y_true_heat_region = np.stack(y_true_heat, axis=-1)
                y_pred_heat_region = np.stack(y_pred_heat, axis=-1)    

                y_true_load_regionlist.append(y_true_load_region)
                y_pred_load_regionlist.append(y_pred_load_region)
                y_true_gas_regionlist.append(y_true_gas_region)
                y_pred_gas_regionlist.append(y_pred_gas_region)
                y_true_heat_regionlist.append(y_true_heat_region)
                y_pred_heat_regionlist.append(y_pred_heat_region)

                y_pred_load = []
                y_true_load = []
                y_pred_gas = []
                y_true_gas = []
                y_pred_heat = []
                y_true_heat = []
                index_all = i4data_all
            y_true_load.append(y_load)
            y_pred_load.append(st_pre_load)
            y_true_gas.append(y_gas)
            y_pred_gas.append(st_pre_gas)
            y_true_heat.append(y_heat)
            y_pred_heat.append(st_pre_heat)

            if (i == len(data_t) - 1 and idx == len(file_list) - 1):
                y_true_load_region = np.stack(y_true_load, axis=-1)
                print(y_true_load_region.shape)
                y_pred_load_region = np.stack(y_pred_load, axis=-1)
                y_true_gas_region = np.stack(y_true_gas, axis=-1)
                y_pred_gas_region = np.stack(y_pred_gas, axis=-1)
                y_true_heat_region = np.stack(y_true_heat, axis=-1)
                y_pred_heat_region = np.stack(y_pred_heat, axis=-1)
                y_true_load_regionlist.append(y_true_load_region)
                y_pred_load_regionlist.append(y_pred_load_region)
                y_true_gas_regionlist.append(y_true_gas_region)
                y_pred_gas_regionlist.append(y_pred_gas_region)
                y_true_heat_regionlist.append(y_true_heat_region)
                y_pred_heat_regionlist.append(y_pred_heat_region)
                y_pred_load = []
                y_true_load = []
                y_pred_gas = []
                y_true_gas = []
                y_pred_heat = []
                y_true_heat = []
                
    print('len_nums', len_nums)

    y_true_load = np.stack(y_true_load_regionlist, axis=0)
    y_pred_load = np.stack(y_pred_load_regionlist, axis=0)
    y_true_gas = np.stack(y_true_gas_regionlist, axis=0)
    y_pred_gas = np.stack(y_pred_gas_regionlist, axis=0)
    y_true_heat = np.stack(y_true_heat_regionlist, axis=0)
    y_pred_heat = np.stack(y_pred_heat_regionlist, axis=0)
    y_pred_load, y_pred_gas, y_pred_heat = np.abs(y_pred_load), np.abs(y_pred_gas), np.abs(y_pred_heat)
    print(y_true_load.shape, y_pred_load.shape, y_true_gas.shape, y_pred_gas.shape, y_true_heat.shape, y_pred_heat.shape)

    if mode == 'classification':
        test_classfication(y_true_load, y_pred_load, y_true_gas, y_pred_gas, y_true_heat, y_pred_heat)
    else:
        print("************* load EVAL *************")
        for t in range(y_true_load.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred_load[:, t, ...], y_true_load[:, t, ...], mae_thresh, mape_thresh, None)
            rarr = compute_rarr(y_pred=y_pred_load[:, t, ...], y_true=y_true_load[:, t, ...])
            worst_mae = worst_n_mae(y_pred_load[:, t, ...],y_true_load[:, t, ...],5)
            print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}% RARR: {:.4f}% worst5% mae: {:.4f}".format(t + 1, mae, rmse, mape * 100,rarr * 100, worst_mae))
        mae, rmse, mape, _, _ = All_Metrics(y_pred_load, y_true_load, mae_thresh, mape_thresh, None)
        rarr = compute_rarr(y_pred=y_pred_load, y_true=y_true_load)
        worst_mae = worst_n_mae(y_pred_load,y_true_load,5)
        print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, RARR: {:.4f}% worst5% mae: {:.4f}".format(mae, rmse, mape * 100,rarr * 100, worst_mae))
        print("************* gas EVAL *************")
        for t in range(y_true_load.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred_gas[:, t, ...], y_true_gas[:, t, ...], mae_thresh, mape_thresh, None)
            rarr = compute_rarr(y_pred=y_pred_gas[:, t, ...], y_true=y_true_gas[:, t, ...])
            worst_mae = worst_n_mae(y_pred_gas[:, t, ...],y_true_gas[:, t, ...],5)
            print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}% RARR: {:.4f}% worst5% mae: {:.4f}".format(t + 1, mae, rmse, mape * 100,rarr * 100, worst_mae))
        mae, rmse, mape, _, _ = All_Metrics(y_pred_gas, y_true_gas, mae_thresh, mape_thresh, None)
        rarr = compute_rarr(y_pred=y_pred_gas, y_true=y_true_gas)
        worst_mae = worst_n_mae(y_pred_gas,y_true_gas,5)
        print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, RARR: {:.4f}% worst5% mae: {:.4f}".format(mae, rmse, mape * 100,rarr * 100, worst_mae))
        print("************* heat EVAL *************")
        for t in range(y_true_load.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred_heat[:, t, ...], y_true_heat[:, t, ...], mae_thresh, mape_thresh, None)
            rarr = compute_rarr(y_pred=y_pred_heat[:, t, ...], y_true=y_true_heat[:, t, ...])
            worst_mae = worst_n_mae(y_pred_heat[:, t, ...],y_true_heat[:, t, ...],5)
            print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}% RARR: {:.4f}% worst5% mae: {:.4f}".format(t + 1, mae, rmse, mape * 100,rarr * 100, worst_mae))
        mae, rmse, mape, _, _ = All_Metrics(y_pred_heat, y_true_heat, mae_thresh, mape_thresh, None)
        rarr = compute_rarr(y_pred=y_pred_heat, y_true=y_true_heat)
        worst_mae = worst_n_mae(y_pred_heat,y_true_heat,5)
        print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, RARR: {:.4f}% worst5% mae: {:.4f}".format(mae, rmse, mape * 100,rarr * 100, worst_mae))
        print("************* average EVAL *************")
        for t in range(y_true_load.shape[1]):
            heat_mae, heat_rmse, heat_mape, _, _ = All_Metrics(y_pred_heat[:, t, ...], y_true_heat[:, t, ...], mae_thresh, mape_thresh, None)
            heat_rarr = compute_rarr(y_pred=y_pred_heat[:, t, ...], y_true=y_true_heat[:, t, ...])
            heat_worst_mae = worst_n_mae(y_pred_heat[:, t, ...],y_true_heat[:, t, ...],5)

            gas_mae, gas_rmse, gas_mape, _, _ = All_Metrics(y_pred_gas[:, t, ...], y_true_gas[:, t, ...], mae_thresh, mape_thresh, None)
            gas_rarr = compute_rarr(y_pred=y_pred_gas[:, t, ...], y_true=y_true_gas[:, t, ...])
            gas_worst_mae = worst_n_mae(y_pred_gas[:, t, ...],y_true_gas[:, t, ...],5)

            load_mae, load_rmse, load_mape, _, _ = All_Metrics(y_pred_load[:, t, ...], y_true_load[:, t, ...], mae_thresh, mape_thresh, None)
            load_rarr = compute_rarr(y_pred=y_pred_load[:, t, ...], y_true=y_true_load[:, t, ...])
            load_worst_mae = worst_n_mae(y_pred_load[:, t, ...],y_true_load[:, t, ...],5)
            mae = (heat_mae + gas_mae + load_mae) / 3
            rmse = (heat_rmse + gas_rmse + load_rmse) / 3
            mape = (heat_mape + gas_mape + load_mape) / 3
            rarr = (heat_rarr + gas_rarr + load_rarr) / 3
            worst_mae = (heat_worst_mae + gas_worst_mae + load_worst_mae) / 3
            print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}% RARR: {:.4f}% worst5% mae: {:.4f}".format(t + 1, mae, rmse, mape * 100,rarr * 100, worst_mae))
        heat_mae, heat_rmse, heat_mape, _, _ = All_Metrics(y_pred_heat, y_true_heat, mae_thresh, mape_thresh, None)
        heat_rarr = compute_rarr(y_pred=y_pred_heat, y_true=y_true_heat)
        heat_worst_mae = worst_n_mae(y_pred_heat,y_true_heat,5)

        gas_mae, gas_rmse, gas_mape, _, _ = All_Metrics(y_pred_gas, y_true_gas, mae_thresh, mape_thresh, None)
        gas_rarr = compute_rarr(y_pred=y_pred_gas, y_true=y_true_gas)
        gas_worst_mae = worst_n_mae(y_pred_gas,y_true_gas,5)

        load_mae, load_rmse, load_mape, _, _ = All_Metrics(y_pred_load, y_true_load, mae_thresh, mape_thresh, None)
        load_rarr = compute_rarr(y_pred=y_pred_load, y_true=y_true_load)
        load_worst_mae = worst_n_mae(y_pred_load,y_true_load,5)
        mae = (heat_mae + gas_mae + load_mae) / 3
        rmse = (heat_rmse + gas_rmse + load_rmse) / 3
        mape = (heat_mape + gas_mape + load_mape) / 3
        rarr = (heat_rarr + gas_rarr + load_rarr) / 3
        worst_mae = (heat_worst_mae + gas_worst_mae + load_worst_mae) / 3
        print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, RARR: {:.4f}% worst5% mae: {:.4f}".format(mae, rmse, mape * 100,rarr * 100, worst_mae))


def test_classfication(y_true_load, y_pred_load, y_true_gas, y_pred_gas, y_true_heat, y_pred_heat):

    for i in range(4):
        if i == 0:
            y_true = y_true_load
            y_pred = y_pred_load
        elif i == 1:
            y_true = y_true_gas
            y_pred = y_pred_gas
        elif i == 2:
            y_true = y_true_heat
            y_pred = y_pred_heat
        y_true[y_true > 1] = 1
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)

        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"MicroF1: {micro_f1:.2f}")
        print(f"MacroF1: {macro_f1:.2f}")
        print(f"f1 Score: {f1:.2f}")

################################ result path ################################
#folder_path = 'result_test_file/tw2t_multi_reg-cla_NYC_taxi_final'
# folder_path = 'result_test_file/tw2t_multi_reg-cla_NYC_bike_final'

# 'BURGLARY': 0, 'GRAND LARCENY': 1, 'ROBBERY': 2, 'FELONY ASSAULT': 3
# folder_path = 'result_test_file/tw2t_multi_reg-cla_NYC_crime1_final'
# folder_path = 'result_test_file/tw2t_multi_reg-cla_NYC_crime2_final'
folder_path = '../result_test/MoE_Encoder_Heat_GCN_7b_loss_final_1024_'

# mode = 'classification' # regression  or  classification
mode = 'regression'

# Make sure that the total length of your json file(s) a multiple of 80.
test(mode)
