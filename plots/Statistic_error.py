import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.stats as stats

def plot_models_error():
    bkl_group, L_bl_group = [], []
    bkl_group1, L_bl_group1 = [], []
    rmse_dfs = []
    me_dfs = []
    for model_name in model_list:
        rmse_me = np.load("./figures/Data_with_initialization_Sinus12Stair5s/{}/{}_error{}.npy".format(model_name, test_data, suffix))
        print(rmse_me.shape)
        rmse_dfs.append(rmse_me[:, 0])
        me_dfs.append(rmse_me[:, 1])
        if "bkl" in model_name:
            bkl_group = rmse_me[:, 0]
            bkl_group1 = rmse_me[:, 1]
        if "L_bl" in model_name:
            L_bl_group = rmse_me[:, 0]
            L_bl_group1 = rmse_me[:, 1]

    plt.figure(figsize=(88.9 / 25.4 * 2, 88.9 / 25.4 * 1.2))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.subplot(1, 2, 1)
    plt.tick_params(labelsize=10, pad=0.1, length=2)
    plt.boxplot(rmse_dfs)
    plt.xticks([1, 2, 3, 4, 5, 6, 7], model_name_list, rotation=60)
    plt.ylabel("RMSE ({})".format(unit))
    plt.subplot(1, 2, 2)
    plt.tick_params(labelsize=10, pad=0.1, length=2)
    plt.boxplot(me_dfs)
    plt.xticks([1, 2, 3, 4, 5, 6, 7], model_name_list, rotation=60)
    plt.ylabel("Max Error ({})".format(unit))
    plt.tight_layout()
    plt.savefig(
        "./figures/Data_with_initialization_Sinus12Stair5s/{}_trainOn{}_testOn{}{}.png".format(model_type, train_data,
                                                                                           test_data, suffix), dpi=600)
    plt.show()

    # Perform two-sample t-test
    t_stat, p_value = stats.ttest_ind(bkl_group, L_bl_group)
    print(f"Two-Sample T-Test:\nT-statistic: {t_stat}, P-value: {p_value}")

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(bkl_group, L_bl_group)
    print(f"Paired T-Test:\nT-statistic: {t_stat}, P-value: {p_value}")

if __name__ == '__main__':
    model_name_list = ["Backlash", "RDPI", "LSTM", "GRU", "FNN-HIB", "LSTM-Backlash", "LSTM-Backlash-sum"] # "LSTM-Backlash-sum"
    suffix = ""
    # model_type, train_data, test_data = "forward", "Sinus12", "stair1_10s"
    # model_type, train_data, test_data = "forward", "Sinus12", "sinus"
    # model_type, train_data, test_data = "forward", "Stair5s", "stair1_10s"
    # model_type, train_data, test_data = "forward", "Stair5s", "sinus"
    model_type, train_data, test_data = "inverse", "Sinus12", "stair1_10s"
    model_type, train_data, test_data = "inverse", "Sinus12", "sinus"
    # model_type, train_data, test_data = "inverse", "Stair5s", "stair1_10s"
    # model_type, train_data, test_data = "inverse", "Stair5s", "sinus"
    # model_type, train_data, test_data, suffix = "inverse", "Stair5s", "stair1_10s", "_forwardSameModel"
    # model_type, train_data, test_data, suffix = "inverse", "Stair5s", "stair1_10s", "_L_bl"
    # model_type = "inverse"
    # train_data = "Sinus"
    # train_data = "Stair5s"
    # train_data = "SinusStair5s"
    model_list = [model_type + str + train_data for str in ["_bkl_", "_PI_", "_LSTM_", "_GRU_", "_FNN_", "_L_bl_", "_sum_"]] #

    # test_data = "stair"
    # test_data = "sinus"
    if model_type == "forward":
        unit = "deg"
    else:
        unit = "mm"
    plot_models_error()



    # coef_x = [-3.62136415e-03, -4.61111115e-02,  7.04893736e+01]
    # coef_y = [-0.00326109,  0.77266619,  0.21515682]
    # p_x = np.poly1d(coef_x)
    # p_y = np.poly1d(coef_y)

    # model_name_list = ["LSTM", "LSTM_mapXY", "LSTM_st", "LSTM_st_mapXY"]
    # model_list = ["forward_LSTM_Sinus", "forward_LSTM_Sinus_map", "forward_LSTM_Stair5s", "forward_LSTM_Stair5s_map"] # "forward_LSTM_Sinus_map"

    # model_name_list = ["LSTM-backlash", "LSTM-backlash_mapXY", "LSTM-backlash_st", "LSTM-backlash_st_mapXY"]
    # model_list = ["forward_L-bl_Sinus", "forward_L-bl_Sinus_map", "forward_L-bl_Stair5s", "forward_L-bl_Stair5s_map"]
    model_type, train_data, test_data = "forward", "Stair5s", "stair1_10s"
    model_type, train_data, test_data = "forward", "Stair5s", "sinus"
    model_name_list = ["Backlash", "RDPI", "LSTM", "GRU", "FNN", "LSTM-Backlash", "LSTM-Backlash-sum"]
    model_list = ["forward_bkl_Stair5s_map", "forward_PI_Stair5s_map", "forward_LSTM_Stair5s","forward_GRU_Stair5s", "forward_FNN_Stair5s", "forward_L_bl_Stair5s","forward_sum_Stair5s"]

    model_type, train_data, test_data = "inverse", "Stair5s", "stair1_10s"
    model_name_list = ["LSTM", "GRU", "FNN", "LSTM-Backlash", "LSTM-Backlash-sum"]
    model_list = ["inverse_LSTM_Stair5s", "inverse_GRU_Stair5s",
                  "inverse_FNN_Stair5s", "inverse_L_bl_Stair5s", "inverse_sum_Stair5s"]

    rmse_X_dfs = []
    rmse_Y_dfs = []
    me_X_dfs = []
    me_Y_dfs = []
    for model_name in model_list:
        rmse_me = np.load(
            "./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}/{}_error.npy".format(model_name, test_data))
        print(rmse_me.shape)
        if rmse_me.shape[1]==4:
            rmse_X_dfs.append(rmse_me[:, 0])
            rmse_Y_dfs.append(rmse_me[:, 1])
            me_X_dfs.append(rmse_me[:, 2])
            me_Y_dfs.append(rmse_me[:, 3])
        else:
            rmse_Y_dfs.append(rmse_me[:, 0])
            me_Y_dfs.append(rmse_me[:, 1])
    tick_list = [i+1 for i in range(len(model_list))]
    # plt.figure(figsize=(88.9 / 25.4 * 2, 88.9 / 25.4 * 1.2))
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.subplot(1, 2, 1)
    # plt.tick_params(labelsize=10, pad=0.1, length=2)
    # plt.boxplot(rmse_X_dfs)
    # plt.xticks(tick_list, model_name_list, rotation=60)
    # plt.ylabel("X RMSE ({})".format(unit))
    # plt.subplot(1, 2, 2)
    # plt.tick_params(labelsize=10, pad=0.1, length=2)
    # plt.boxplot(me_X_dfs)
    # plt.xticks(tick_list, model_name_list, rotation=60)
    # plt.ylabel("X Max Error ({})".format(unit))
    # plt.tight_layout()
    # plt.savefig(
    #     "./figures/Data_with_initialization_Sinus12Stair5s/tipD/TipDX_{}_trainOn{}_testOn{}.png".format(model_type, train_data,
    #                                                                                             test_data), dpi=600)
    plt.figure(figsize=(88.9 / 25.4 * 2, 88.9 / 25.4 * 1.2))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.subplot(1, 2, 1)
    plt.tick_params(labelsize=10, pad=0.1, length=2)
    plt.boxplot(rmse_Y_dfs)
    plt.xticks(tick_list, model_name_list, rotation=60)
    plt.ylabel("Y RMSE ({})".format(unit))
    plt.subplot(1, 2, 2)
    plt.tick_params(labelsize=10, pad=0.1, length=2)
    plt.boxplot(me_Y_dfs)
    plt.xticks(tick_list, model_name_list, rotation=60)
    plt.ylabel("Y Max Error ({})".format(unit))
    plt.tight_layout()
    plt.savefig(
        "./figures/Data_with_initialization_Sinus12Stair5s/tipD/TipDY_{}_trainOn{}_testOn{}.png".format(model_type, train_data, test_data), dpi=600)
    plt.show()