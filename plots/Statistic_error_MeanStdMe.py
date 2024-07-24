import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.stats as stats

def plot_models_error():
    bkl_group, L_bl_group = [], []
    bkl_group1, L_bl_group1 = [], []
    rmse_mean_dfs = []
    rmse_std_dfs = []
    me_dfs = []
    for model_name in model_list:
        rmse_me = np.load("./figures/Data_with_initialization_Sinus12Stair5s/{}/{}_error{}.npy".format(model_name, test_data, suffix))
        print(rmse_me.shape)
        rmse_mean_dfs.append(np.mean(rmse_me[:, 0]))
        rmse_std_dfs.append(np.std(rmse_me[:, 0]))
        me_dfs.append(np.max(rmse_me[:, 1]))
    tick_list = [i + 1 for i in range(len(model_list))]
    plt.figure(figsize=(88.9 / 25.4 * 2, 88.9 / 25.4 * 2))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.tick_params(labelsize=10, pad=0.1, length=2)
    plt.errorbar(tick_list, rmse_mean_dfs, yerr=rmse_std_dfs, fmt='o', label='Mean ± std', capsize=5)
    plt.plot(tick_list, me_dfs, 'r*')
    plt.xticks(tick_list, model_name_list, rotation=60)
    plt.ylabel("RMSE and MaxE({})".format(unit))
    plt.title(["{:.2f}_{:.2f}_{:.2f}".format(mean, std, maxe) for mean, std, maxe in zip(rmse_mean_dfs, rmse_std_dfs, me_dfs)], fontsize=9)
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


def plot_Y_error():
    rmse_X_dfs = []
    me_X_dfs = []
    rmse_Y_mean_dfs = []
    rmse_Y_std_dfs = []
    me_Y_dfs = []
    for model_name in model_list:
        rmse_me = np.load(
            "./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}/{}_error.npy".format(model_name, test_data))
        print(rmse_me.shape)
        if rmse_me.shape[1] == 4:
            rmse_Y_mean_dfs.append(np.mean(rmse_me[:, 1]))
            rmse_Y_std_dfs.append(np.std(rmse_me[:, 1]))
            me_Y_dfs.append(np.max(rmse_me[:, 3]))
        else:
            rmse_Y_mean_dfs.append(np.mean(rmse_me[:, 0]))
            rmse_Y_std_dfs.append(np.std(rmse_me[:, 0]))
            me_Y_dfs.append(np.max(rmse_me[:, 1]))
    tick_list = [i + 1 for i in range(len(model_list))]

    plt.figure(figsize=(88.9 / 25.4 * 2, 88.9 / 25.4 * 2))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.tick_params(labelsize=10, pad=0.1, length=2)
    plt.errorbar(tick_list, rmse_Y_mean_dfs, yerr=rmse_Y_std_dfs, fmt='o', label='Mean ± std', capsize=5)
    plt.plot(tick_list, me_Y_dfs, 'r*')
    plt.xticks(tick_list, model_name_list, rotation=60)
    plt.ylabel("Y RMSE and MaxE({})".format("mm"))
    plt.title(["{:.2f}_{:.2f}_{:.2f}".format(mean, std, maxe) for mean, std, maxe in
               zip(rmse_Y_mean_dfs, rmse_Y_std_dfs, me_Y_dfs)], fontsize=9)
    plt.tight_layout()
    plt.savefig(
        "./figures/Data_with_initialization_Sinus12Stair5s/tipD/TipDY_{}_trainOn{}_testOn{}.png".format(model_type,
                                                                                                        train_data,
                                                                                                        test_data),
        dpi=600)
    plt.show()

def plot_theta_error_merged(version, model_name_list, str_list, suffix=""):
    for model_type in ["forward", "inverse"]:
        unit = "deg" if model_type == "forward" else "mm"
        rmse_mean_dfs = []
        rmse_std_dfs = []
        me_dfs = []
        if version == "v1":
            for train_data in ["Sinus12", "Stair5s"]:
                train_data += suffix
                model_list = [model_type + str + train_data for str in str_list]  #
                for test_data in ["sinus", "stair1_10s"]:
                    for model_name in model_list:
                        rmse_me = np.load(
                            "./figures/Data_with_initialization_Sinus12Stair5s/{}/{}_error.npy".format(model_name,
                                                                                                       test_data))
                        rmse_mean_dfs.append(np.mean(rmse_me[:, 0]))
                        rmse_std_dfs.append(np.std(rmse_me[:, 0]))
                        me_dfs.append(np.max(rmse_me[:, 1]))
        else:
            for str in str_list:
                for train_data in ["Sinus12", "Stair5s"]:
                    train_data += suffix
                    model_name = model_type + str + train_data
                    for test_data in ["sinus", "stair1_10s"]:
                        rmse_me = np.load("./figures/Data_with_initialization_Sinus12Stair5s/{}/{}_error.npy".format(model_name, test_data))
                        rmse_mean_dfs.append(np.mean(rmse_me[:, 0]))
                        rmse_std_dfs.append(np.std(rmse_me[:, 0]))
                        me_dfs.append(np.max(rmse_me[:, 1]))
        tick_list = [i + 1 for i in range(len(model_name_list)*4)]
        plt.figure(figsize=(88.9 / 25.4*2, 88.9 / 25.4))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.tick_params(labelsize=10, pad=0.1, length=2)
        symbol_list = ["BK", "PI", "LSTM", "GRU", "FNN", "HB1", "HB2"]
        if model_type=="forward":
            offset = 0.15
        else:
            offset = 0.02
        for i in range(len(me_dfs)):
            plt.text(tick_list[i], rmse_mean_dfs[i] + rmse_std_dfs[i], f'{me_dfs[i]:.2f}', ha='center', va='bottom', fontsize=7)
            plt.text(tick_list[i], rmse_mean_dfs[i] - rmse_std_dfs[i]-offset, symbol_list[i%7], ha='center', va='bottom', fontsize=7)
        # plt.xticks(tick_list, model_name_list, rotation=60)
        if version == "v1":
            # fmt_list = ['b'+f for f in ['o', '^', 'p', 'h', '*', '+', 'x']]
            color_list = ['red', 'orange', 'green', 'blue', 'k', 'purple', 'pink']
            for i in range(0, 7):
                for j in range(0, 4):
                    plt.errorbar(tick_list[i+7*j], rmse_mean_dfs[i+7*j], yerr=rmse_std_dfs[i+7*j], fmt="o", color=color_list[i], label='Mean ± std', capsize=5) #color=color_list[i],
            for x in np.arange(1, 4)*7+0.5:
                plt.axvline(x, linestyle='--', color='k', linewidth=0.5)
            plt.xticks(np.arange(0, 4) * 7 + 4, ["TrainSinus_TestSinus", "TrainSinus_TestStat", "TrainStat_TestSinus", "TrainStat_TestStat"])
        else:
            label_list = ["TrainSinus_TestSinus", "TrainSinus_TestStat", "TrainStat_TestSinus", "TrainStat_TestStat"]
            color_list = ['red', 'orange', 'green', 'blue']
            for i in range(0, 4):
                for j in range(0, 7):
                    plt.errorbar(tick_list[i + 4 * j], rmse_mean_dfs[i + 4 * j], yerr=rmse_std_dfs[i + 4 * j],
                                 fmt='o', color=color_list[i], label='Mean ± std', capsize=5)  # color=color_list[i],
            for x in np.arange(1, 7)*4+0.5:
                plt.axvline(x, linestyle='--', color='k', linewidth=0.5)
            plt.xticks(np.arange(0, 7)*4+2.5, model_name_list, rotation=20)
        if model_type == "forward":
            plt.ylim([0, 3.2])
        else:
            plt.ylim([0, 0.37])
        plt.ylabel("RMSE and MaxE({})".format(unit))
        plt.tight_layout()
        plt.savefig(
            "./figures/Data_with_initialization_Sinus12Stair5s/{}_{}_{}.png".format(model_type, version, suffix), dpi=600)
        plt.show()

def plot_Y_error_merged(version, model_name_list, str_list, suffix=""):
    for model_type in ["forward", "inverse"]:
        unit = "mm"
        rmse_mean_dfs = []
        rmse_std_dfs = []
        me_dfs = []
        if version == "v1":
            for train_data in ["Sinus12", "Stair5s"]:
                train_data += suffix
                model_list = [model_type + str + train_data + "_map" for str in ["_bkl_", "_PI_"]]
                model_list = model_list + [model_type + str + train_data for str in ["_LSTM_", "_GRU_", "_FNN_", "_L_bl_", "_sum_"]]
                for test_data in ["sinus", "stair1_10s"]:
                    for model_name in model_list:
                        rmse_me = np.load(
                            "./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}/{}_error.npy".format(model_name,
                                                                                                       test_data))
                        if rmse_me.shape[1] == 4:
                            print(model_name)
                            rmse_mean_dfs.append(np.mean(rmse_me[:, 1]))
                            rmse_std_dfs.append(np.std(rmse_me[:, 1]))
                            me_dfs.append(np.max(rmse_me[:, 3]))
                        else:
                            rmse_mean_dfs.append(np.mean(rmse_me[:, 0]))
                            rmse_std_dfs.append(np.std(rmse_me[:, 0]))
                            me_dfs.append(np.max(rmse_me[:, 1]))
        else:
            for str in str_list:
                for train_data in ["Sinus12", "Stair5s"]:
                    train_data += suffix
                    if str == "_bkl_" or str == "_PI_":
                        model_name = model_type + str + train_data + "_map"
                    else:
                        model_name = model_type + str + train_data
                    for test_data in ["sinus", "stair1_10s"]:
                        rmse_me = np.load("./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}/{}_error.npy".format(model_name, test_data))
                        if rmse_me.shape[1] == 4:
                            rmse_mean_dfs.append(np.mean(rmse_me[:, 1]))
                            rmse_std_dfs.append(np.std(rmse_me[:, 1]))
                            me_dfs.append(np.max(rmse_me[:, 3]))
                        else:
                            rmse_mean_dfs.append(np.mean(rmse_me[:, 0]))
                            rmse_std_dfs.append(np.std(rmse_me[:, 0]))
                            me_dfs.append(np.max(rmse_me[:, 1]))
        tick_list = [i + 1 for i in range(len(model_name_list)*4)]
        plt.figure(figsize=(88.9 / 25.4*2, 88.9 / 25.4))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.tick_params(labelsize=10, pad=0.1, length=2)
        # plt.plot(tick_list, me_dfs, 'r*')
        symbol_list = ["BK", "PI", "LSTM", "GRU", "FNN", "HB1", "HB2"]
        if model_type == "forward":
            offset = 0.15
        else:
            offset = 0.02
        for i in range(len(me_dfs)):
            plt.text(tick_list[i], rmse_mean_dfs[i] + rmse_std_dfs[i], f'{me_dfs[i]:.2f}', ha='center', va='bottom', fontsize=7)
            plt.text(tick_list[i], rmse_mean_dfs[i] - rmse_std_dfs[i] - offset, symbol_list[i % 7], ha='center', va='bottom', fontsize=7)
        # plt.xticks(tick_list, model_name_list, rotation=60)
        if version == "v1":
            color_list = ['red', 'orange', 'green', 'blue', 'k', 'purple', 'pink']
            for i in range(0, 7):
                for j in range(0, 4):
                    plt.errorbar(tick_list[i + 7 * j], rmse_mean_dfs[i + 7 * j], yerr=rmse_std_dfs[i + 7 * j], fmt="o",
                                 color=color_list[i], label='Mean ± std', capsize=5)
            for x in np.arange(1, 4)*7+0.5:
                plt.axvline(x, linestyle='--', color='k', linewidth=0.5)
            plt.xticks(np.arange(0, 4) * 7 + 4, ["TrainSinus_TestSinus", "TrainSinus_TestStat", "TrainStat_TestSinus", "TrainStat_TestStat"])
        else:
            label_list = ["TrainSinus_TestSinus", "TrainSinus_TestStat", "TrainStat_TestSinus", "TrainStat_TestStat"]
            color_list = ['red', 'orange', 'green', 'blue']
            for i in range(0, 4):
                for j in range(0, 7):
                    plt.errorbar(tick_list[i + 4 * j], rmse_mean_dfs[i + 4 * j], yerr=rmse_std_dfs[i + 4 * j],
                                 fmt='o', color=color_list[i], label='Mean ± std', capsize=5)  # color=color_list[i],
            for x in np.arange(1, 7)*4+0.5:
                plt.axvline(x, linestyle='--', color='k', linewidth=0.5)
            plt.xticks(np.arange(0, 7)*4+2.5, model_name_list, rotation=20)
        if model_type == "forward":
            plt.ylim([0, 2.5])
        else:
            plt.ylim([0, 0.37])
        plt.ylabel("RMSE and MaxE({})".format(unit))
        plt.tight_layout()
        plt.savefig(
            "./figures/Data_with_initialization_Sinus12Stair5s/tipD/Y_{}_{}_{}.png".format(model_type, version, suffix), dpi=600)
        plt.show()

def plot_theta_error_merged_bestLoss(version, model_name_list, str_list):
    for model_type in ["forward", "inverse"]:
        unit = "degrees" if model_type == "forward" else "mm"
        rmse_mean_dfs = []
        rmse_std_dfs = []
        me_dfs = []
        old_new_loss = []
        if version == "v1":
            for train_data in ["Sinus12", "Stair5s"]:
                model_list = [model_type + str + train_data for str in str_list]  #
                for test_data in ["sinus", "stair1_10s"]:
                    if train_data=="Sinus12" and test_data=="stair1_10s":
                        continue
                    if train_data=="Stair5s" and test_data=="sinus":
                        continue
                    for model_name in model_list:
                        rmse_me = np.load("./figures/Data_with_initialization_Sinus12Stair5s/{}/{}_error.npy".format(model_name, test_data))
                        rmse_me_StdL = np.load("./figures/Data_with_initialization_Sinus12Stair5s/{}StdL/{}_error.npy".format(model_name, test_data))
                        if np.mean(rmse_me[:, 0]) <= np.mean(rmse_me_StdL[:, 0]):
                            rmse_mean_dfs.append(np.mean(rmse_me[:, 0]))
                            rmse_std_dfs.append(np.std(rmse_me[:, 0]))
                            me_dfs.append(np.max(rmse_me[:, 1]))
                            old_new_loss.append(0)
                        else:
                            rmse_mean_dfs.append(np.mean(rmse_me_StdL[:, 0]))
                            rmse_std_dfs.append(np.std(rmse_me_StdL[:, 0]))
                            me_dfs.append(np.max(rmse_me_StdL[:, 1]))
                            old_new_loss.append(1)

        tick_list = [i + 1 for i in range(len(model_name_list)*2)]
        plt.figure(figsize=(88.9 / 25.4*1.1, 88.9 / 25.4))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.tick_params(labelsize=10, pad=0.1, length=2)
        symbol_list = ["BK", "PI", "LSTM", "GRU", "FNN", "HB1", "HB2"]
        symbol_list2 = ["Backlash", "PI", "LSTM", "GRU", "FNN", "Serial LSTM-Backlash", "Parallel LSTM-Backlash"]
        if model_type == "forward":
            offset = 0.08
        else:
            offset = 0.008
        for i in range(len(me_dfs)):
            plt.text(tick_list[i], rmse_mean_dfs[i] + rmse_std_dfs[i], f'{me_dfs[i]:.2f}', ha='center', va='bottom', fontsize=7)
            if old_new_loss[i] == 1:
                plt.text(tick_list[i], rmse_mean_dfs[i] - rmse_std_dfs[i]-offset, symbol_list[i%7], ha='center', va='bottom', fontsize=7, fontweight='bold')
            else:
                plt.text(tick_list[i], rmse_mean_dfs[i] - rmse_std_dfs[i] - offset, symbol_list[i % 7], ha='center', va='bottom', fontsize=7)
        # plt.xticks(tick_list, model_name_list, rotation=60)
        if version == "v1":
            # fmt_list = ['b'+f for f in ['o', '^', 'p', 'h', '*', '+', 'x']]
            color_list = ['red', 'orange', 'green', 'blue', 'k', 'purple', 'pink']
            for i in range(0, 7):
                for j in range(0, 2):
                    if j == 0:
                        plt.errorbar(tick_list[i+7*j], rmse_mean_dfs[i+7*j], yerr=rmse_std_dfs[i+7*j], fmt="o", color=color_list[i], label=symbol_list2[i], capsize=5) #color=color_list[i],
                    else:
                        plt.errorbar(tick_list[i+7*j], rmse_mean_dfs[i+7*j], yerr=rmse_std_dfs[i+7*j], fmt="o", color=color_list[i], capsize=5) #color=color_list[i],
            for x in np.arange(1, 2)*7+0.5:
                plt.axvline(x, linestyle='--', color='k', linewidth=0.5)
            plt.xticks(np.arange(0, 2) * 7 + 4, ["TrainSinus_TestSinus", "TrainStat_TestStat"])

        # if model_type == "forward":
        #     plt.ylim([0, 3.2])
        # else:
        #     plt.ylim([0, 0.37])
        # plt.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15))
        plt.ylabel("RMS Error / Maximum Error ({})".format(unit))
        plt.tight_layout()
        plt.savefig("./figures/Data_with_initialization_Sinus12Stair5s/BestLoss_2groups_{}_{}.png".format(model_type, version), dpi=600)
        plt.show()

def plot_theta_error_merged_bestLoss8(version, model_name_list, str_list):
    for model_type in ["forward", "inverse"]:
        unit = "degrees" if model_type == "forward" else "mm"
        rmse_mean_dfs = []
        rmse_std_dfs = []
        me_dfs = []
        old_new_loss = []
        if version == "v1":
            for train_data in ["Sinus12", "Stair5s"]:
                model_list = [model_type + str + train_data for str in str_list]  #
                for test_data in ["sinus", "stair1_10s"]:
                    if train_data=="Sinus12" and test_data=="stair1_10s":
                        continue
                    if train_data=="Stair5s" and test_data=="sinus":
                        continue
                    for model_name in model_list:
                        rmse_me = np.load("./figures/Data_with_initialization_Sinus12Stair5s/{}/{}_error.npy".format(model_name, test_data))
                        rmse_me_StdL = np.load("./figures/Data_with_initialization_Sinus12Stair5s/{}StdL/{}_error.npy".format(model_name, test_data))
                        if np.mean(rmse_me[:, 0]) <= np.mean(rmse_me_StdL[:, 0]):
                            rmse_mean_dfs.append(np.mean(rmse_me[:, 0]))
                            rmse_std_dfs.append(np.std(rmse_me[:, 0]))
                            me_dfs.append(np.max(rmse_me[:, 1]))
                            old_new_loss.append(0)
                        else:
                            rmse_mean_dfs.append(np.mean(rmse_me_StdL[:, 0]))
                            rmse_std_dfs.append(np.std(rmse_me_StdL[:, 0]))
                            me_dfs.append(np.max(rmse_me_StdL[:, 1]))
                            old_new_loss.append(1)

        tick_list = [i + 1 for i in range(len(model_name_list)*2)]
        plt.figure(figsize=(88.9 / 25.4*1.2, 88.9 / 25.4))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.tick_params(labelsize=10, pad=0.1, length=2)
        symbol_list = ["FNN", "BK", "PI", "LSTM", "GRU", "FNN-HIB", "HB1", "HB2"]
        symbol_list2 = ["FNN", "Backlash", "PI", "LSTM", "GRU", "FNN-HIB", "HB1: Serial LSTM-Backlash", "HB2: Parallel LSTM-Backlash"]
        if model_type == "forward":
            offset = 0.1
        else:
            offset = 0.01
        for i in range(len(me_dfs)):
            plt.text(tick_list[i], rmse_mean_dfs[i] + rmse_std_dfs[i], f'{me_dfs[i]:.2f}', ha='center', va='bottom', fontsize=7)
            if old_new_loss[i] == 1:
                plt.text(tick_list[i], rmse_mean_dfs[i] - rmse_std_dfs[i]-offset, symbol_list[i%8], ha='center', va='bottom', fontsize=7, fontweight='bold')
            else:
                plt.text(tick_list[i], rmse_mean_dfs[i] - rmse_std_dfs[i] - offset, symbol_list[i % 8], ha='center', va='bottom', fontsize=7)
        # plt.xticks(tick_list, model_name_list, rotation=60)
        if version == "v1":
            # fmt_list = ['b'+f for f in ['o', '^', 'p', 'h', '*', '+', 'x']]
            color_list = ['brown', 'red', 'orange', 'green', 'blue', 'k', 'purple', 'pink']
            for i in range(0, 8):
                for j in range(0, 2):
                    if j == 0:
                        plt.errorbar(tick_list[i+8*j], rmse_mean_dfs[i+8*j], yerr=rmse_std_dfs[i+8*j], fmt="o", color=color_list[i], label=symbol_list2[i], capsize=5) #color=color_list[i],
                    else:
                        plt.errorbar(tick_list[i+8*j], rmse_mean_dfs[i+8*j], yerr=rmse_std_dfs[i+8*j], fmt="o", color=color_list[i], capsize=5) #color=color_list[i],
            for x in np.arange(1, 2)*8+0.5:
                plt.axvline(x, linestyle='--', color='k', linewidth=0.5)
            plt.xticks(np.arange(0, 2) * 8 + 4, ["TrainSinus_TestSinus", "TrainStat_TestStat"])

        # if model_type == "forward":
        #     plt.ylim([0, 3.2])
        # else:
        #     plt.ylim([0, 0.37])
        plt.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15))
        plt.ylabel("RMS Error / Maximum Error ({})".format(unit))
        plt.tight_layout()
        plt.savefig("./figures/Data_with_initialization_Sinus12Stair5s/BestLoss_2groups_{}_{}.png".format(model_type, version), dpi=600)
        plt.show()

def plot_Y_error_merged_BestLoss(version, model_name_list, str_list):
    for model_type in ["forward", "inverse"]:
        unit = "mm"
        rmse_mean_dfs = []
        rmse_std_dfs = []
        me_dfs = []
        old_new_loss = []
        if version == "v1":
            for train_data in ["Sinus12", "Stair5s"]:
                model_list = [model_type + str + train_data + "_map" for str in ["_bkl_", "_PI_"]]
                model_list = model_list + [model_type + str + train_data for str in ["_LSTM_", "_GRU_", "_FNN_", "_L_bl_", "_sum_"]]
                model_StdL_list = [model_type + str + train_data + "StdL" + "_map" for str in ["_bkl_", "_PI_"]]
                model_StdL_list = model_StdL_list + [model_type + str + train_data + "StdL" for str in ["_LSTM_", "_GRU_", "_FNN_", "_L_bl_", "_sum_"]]
                for test_data in ["sinus", "stair1_10s"]:
                    if train_data=="Sinus12" and test_data=="stair1_10s":
                        continue
                    if train_data=="Stair5s" and test_data=="sinus":
                        continue
                    for model_name, model_name_StdL in zip(model_list, model_StdL_list):
                        rmse_me = np.load(
                            "./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}/{}_error.npy".format(model_name,
                                                                                                       test_data))
                        rmse_me_StdL = np.load(
                            "./figures/Data_with_initialization_Sinus12Stair5s/tipD/{}/{}_error.npy".format(model_name_StdL,
                                                                                                           test_data))

                        if rmse_me.shape[1] == 4:
                            if np.mean(rmse_me[:, 1]) <= np.mean(rmse_me_StdL[:, 1]):
                                rmse_mean_dfs.append(np.mean(rmse_me[:, 1]))
                                rmse_std_dfs.append(np.std(rmse_me[:, 1]))
                                me_dfs.append(np.max(rmse_me[:, 3]))
                                old_new_loss.append(0)
                            else:
                                rmse_mean_dfs.append(np.mean(rmse_me_StdL[:, 1]))
                                rmse_std_dfs.append(np.std(rmse_me_StdL[:, 1]))
                                me_dfs.append(np.max(rmse_me_StdL[:, 3]))
                                old_new_loss.append(1)
                        else:
                            if np.mean(rmse_me[:, 0]) <= np.mean(rmse_me_StdL[:, 0]):
                                rmse_mean_dfs.append(np.mean(rmse_me[:, 0]))
                                rmse_std_dfs.append(np.std(rmse_me[:, 0]))
                                me_dfs.append(np.max(rmse_me[:, 1]))
                                old_new_loss.append(0)
                            else:
                                rmse_mean_dfs.append(np.mean(rmse_me_StdL[:, 0]))
                                rmse_std_dfs.append(np.std(rmse_me_StdL[:, 0]))
                                me_dfs.append(np.max(rmse_me_StdL[:, 1]))
                                old_new_loss.append(1)


        tick_list = [i + 1 for i in range(len(model_name_list)*2)]
        plt.figure(figsize=(88.9 / 25.4 *1.1, 88.9 / 25.4))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.tick_params(labelsize=10, pad=0.1, length=2)
        # plt.plot(tick_list, me_dfs, 'r*')
        symbol_list = ["BK", "PI", "LSTM", "GRU", "FNN", "HB1", "HB2"]
        if model_type == "forward":
            offset = 0.05
        else:
            offset = 0.01
        for i in range(len(me_dfs)):
            plt.text(tick_list[i], rmse_mean_dfs[i] + rmse_std_dfs[i], f'{me_dfs[i]:.2f}', ha='center', va='bottom', fontsize=7)
            if old_new_loss[i]==1:
                plt.text(tick_list[i], rmse_mean_dfs[i] - rmse_std_dfs[i] - offset, symbol_list[i % 7], ha='center', va='bottom', fontsize=7, fontweight='bold')
            else:
                plt.text(tick_list[i], rmse_mean_dfs[i] - rmse_std_dfs[i] - offset, symbol_list[i % 7], ha='center', va='bottom', fontsize=7)
        # plt.xticks(tick_list, model_name_list, rotation=60)
        if version == "v1":
            color_list = ['red', 'orange', 'green', 'blue', 'k', 'purple', 'pink']
            for i in range(0, 7):
                for j in range(0, 2):
                    plt.errorbar(tick_list[i + 7 * j], rmse_mean_dfs[i + 7 * j], yerr=rmse_std_dfs[i + 7 * j], fmt="o",
                                 color=color_list[i], label='Mean ± std', capsize=5)
            for x in np.arange(1, 2)*7+0.5:
                plt.axvline(x, linestyle='--', color='k', linewidth=0.5)
            plt.xticks(np.arange(0, 2) * 7 + 4, ["TrainSinus_TestSinus", "TrainStat_TestStat"])
        # if model_type == "forward":
        #     plt.ylim([0, 2.5])
        # else:
        #     plt.ylim([0, 0.37])
        plt.ylabel("RMS Error / Maximum Error ({})".format(unit))
        plt.tight_layout()
        plt.savefig(
            "./figures/Data_with_initialization_Sinus12Stair5s/tipD/BestLoss_2groups_Y_{}_{}.png".format(model_type, version), dpi=600)
        plt.show()

if __name__ == '__main__':

    suffix = ""
    # model_type, train_data, test_data = "forward", "Sinus12", "stair1_10s"
    # model_type, train_data, test_data = "forward", "Sinus12", "sinus"
    model_type, train_data, test_data = "forward", "Stair5s", "stair1_10s"
    # model_type, train_data, test_data = "forward", "Stair5sStdL", "stair1_10s"
    model_type, train_data, test_data = "forward", "Sinus12StdL", "stair1_10s"
    # model_type, train_data, test_data = "forward", "Stair5s", "sinus"
    # model_type, train_data, test_data = "inverse", "Sinus12", "stair1_10s"
    # model_type, train_data, test_data = "inverse", "Sinus12", "sinus"
    # model_type, train_data, test_data = "inverse", "Stair5s", "stair1_10s"
    # model_type, train_data, test_data = "inverse", "Stair5s", "sinus"
    # model_type, train_data, test_data, suffix = "inverse", "Stair5s", "stair1_10s", "_forwardSameModel"
    # model_type, train_data, test_data, suffix = "inverse", "Stair5s", "stair1_10s", "_L_bl"
    # model_name_list = ["Backlash", "RDPI", "LSTM", "GRU", "FNN-HIB", "LSTM-Backlash", "LSTM-Backlash-sum"]  # "LSTM-Backlash-sum"
    model_name_list = ["LSTM", "GRU"]  # "LSTM-Backlash-sum"
    # model_list = [model_type + str + train_data for str in ["_bkl_", "_PI_", "_LSTM_", "_GRU_", "_FNN_", "_L_bl_", "_sum_"]] #
    model_list = [model_type + str + train_data for str in ["_LSTM_", "_GRU_"]] #
    if model_type == "forward":
        unit = "deg"
    else:
        unit = "mm"
    # plot_models_error()


    # coef_x = [-3.62136415e-03, -4.61111115e-02,  7.04893736e+01]
    # coef_y = [-0.00326109,  0.77266619,  0.21515682]
    # p_x = np.poly1d(coef_x)
    # p_y = np.poly1d(coef_y)

    # model_name_list = ["LSTM", "LSTM_mapXY", "LSTM_st", "LSTM_st_mapXY"]
    # model_list = ["forward_LSTM_Sinus", "forward_LSTM_Sinus_map", "forward_LSTM_Stair5s", "forward_LSTM_Stair5s_map"] # "forward_LSTM_Sinus_map"

    # model_name_list = ["LSTM-backlash", "LSTM-backlash_mapXY", "LSTM-backlash_st", "LSTM-backlash_st_mapXY"]
    # model_list = ["forward_L-bl_Sinus", "forward_L-bl_Sinus_map", "forward_L-bl_Stair5s", "forward_L-bl_Stair5s_map"]
    model_type, train_data, test_data = "forward", "Sinus12", "stair1_10s"
    model_type, train_data, test_data = "forward", "Sinus12", "sinus"
    model_type, train_data, test_data = "forward", "Stair5s", "stair1_10s"
    model_type, train_data, test_data = "forward", "Stair5s", "sinus"
    model_type, train_data, test_data = "inverse", "Sinus12", "stair1_10s"
    model_type, train_data, test_data = "inverse", "Sinus12", "sinus"
    model_type, train_data, test_data = "inverse", "Stair5s", "stair1_10s"
    model_type, train_data, test_data = "inverse", "Stair5s", "sinus"
    model_name_list = ["Backlash", "RDPI", "LSTM", "GRU", "FNN", "LSTM-Backlash", "LSTM-Backlash-sum"]
    # model_list = ["forward_bkl_Stair5s_map", "forward_PI_Stair5s_map", "forward_LSTM_Stair5s", "forward_GRU_Stair5s", "forward_FNN_Stair5s", "forward_L_bl_Stair5s", "forward_sum_Stair5s"]
    model_list = [model_type + str + train_data + "_map" for str in ["_bkl_", "_PI_"]]
    model_list = model_list + [model_type + str + train_data for str in ["_LSTM_", "_GRU_", "_FNN_", "_L_bl_", "_sum_"]]
    # plot_Y_error()



    # --------------------------------------           --------------------------------------------
    model_name_list = ["Backlash", "RDPI", "LSTM", "GRU", "FNN-HIB", "LSTM-Backlash", "LSTM-Backlash-sum"]  # "LSTM-Backlash-sum"
    str_list = ["_bkl_", "_PI_", "_LSTM_", "_GRU_", "_FNN_", "_L_bl_", "_sum_"]
    version = "v1"
    # plot_theta_error_merged(version, model_name_list, str_list)
    # plot_theta_error_merged(version, model_name_list, str_list, suffix="StdL")
    # plot_theta_error_merged_bestLoss(version, model_name_list, str_list)
    model_name_list = ["FNN", "Backlash", "RDPI", "LSTM", "GRU", "FNN-HIB", "LSTM-Backlash",
                       "LSTM-Backlash-sum"]  # "LSTM-Backlash-sum"
    str_list = ["_FEED_", "_bkl_", "_PI_", "_LSTM_", "_GRU_", "_FNN_", "_L_bl_", "_sum_"]
    plot_theta_error_merged_bestLoss8(version, model_name_list, str_list)

    # model_list = [model_type + str + train_data + "_map" for str in ["_bkl_", "_PI_"]]
    # model_list = model_list + [model_type + str + train_data for str in ["_LSTM_", "_GRU_", "_FNN_", "_L_bl_", "_sum_"]]
    # plot_Y_error_merged(version, model_name_list, str_list)
    # plot_Y_error_merged(version, model_name_list, str_list, suffix="StdL")
    # plot_Y_error_merged_BestLoss(version, model_name_list, str_list)