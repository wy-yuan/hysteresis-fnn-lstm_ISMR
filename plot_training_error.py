import pickle
import numpy as np
from matplotlib import pyplot as plt
import os


if __name__ == '__main__':
    # backlash
    forward_backlash1_path = "./checkpoints/Sinus_BacklashNet_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch73_best8.678210576625256e-05.pt"
    forward_backlash2_path = "./checkpoints/Stair5s_BacklashNet_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch167_best4.843862231306654e-05.pt"
    forward_backlash3_path = "./checkpoints/SinusStair5s_Backlash_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch192_best9.588336713534795e-05.pt"
    inverse_backlash1_path = "./checkpoints/Sinus_BacklashInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch236_best0.00036487386042007513.pt"
    inverse_backlash2_path = "./checkpoints/Stair5s_BacklashInv_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch178_best0.00021264683127810712.pt"
    inverse_backlash3_path = "./checkpoints/SinusStair5s_BacklashInv_layer2_seg50_sr25/TC_Backlash_L2_bs16_epoch256_best0.0003787306132002444.pt"

    # LSTM
    forward_LSTM1_path = "./checkpoints/Sinus_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch80_best7.461752777869281e-05.pt"
    forward_LSTM2_path = "./checkpoints/Stair5s_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch89_best0.0003633003450280133.pt"
    forward_LSTM3_path = "./checkpoints/SinusStair5s_LSTM_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch108_best0.0002405683620917971.pt"
    inverse_LSTM1_path = "./checkpoints/Sinus_LSTMInv_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch99_best0.00019427379040773643.pt"
    inverse_LSTM2_path = "./checkpoints/Stair5s_LSTMInv_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch97_best0.0007411067635985091.pt"
    inverse_LSTM3_path = "./checkpoints/SinusStair5s_LSTMInv_layer2_seg50_sr25/TC_LSTM_L2_bs16_epoch95_best0.0004180659107842381.pt"

    # GRU
    forward_GRU1_path = "./checkpoints/Sinus_GRU_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch97_best7.328718462396587e-05.pt"
    forward_GRU2_path = "./checkpoints/Stair5s_GRU_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch99_best0.0003257222237152746.pt"
    forward_GRU3_path = "./checkpoints/SinusStair5s_GRU_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch49_best0.0002430327041730081.pt"
    inverse_GRU1_path = "./checkpoints/Sinus_GRUInv_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch91_best0.0001758793646408545.pt"
    inverse_GRU2_path = "./checkpoints/Stair5s_GRUInv_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch92_best0.0005654142509835462.pt"
    inverse_GRU3_path = "./checkpoints/SinusStair5s_GRUInv_layer2_seg50_sr25/TC_GRU_L2_bs16_epoch92_best0.00036274320216059.pt"

    # FNN-HIB
    forward_FNN1_path = "./checkpoints/Sinus_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch92_best2.9046731570916665e-05.pt"
    forward_FNN2_path = "./checkpoints/Stair5s_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch91_best0.00019324814696567103.pt"
    forward_FNN3_path = "./checkpoints/SinusStair5s_FNN_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch81_best0.00014734179199472742.pt"
    inverse_FNN1_path = "./checkpoints/Sinus_FNNInv_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch74_best0.0002357431442919372.pt"
    inverse_FNN2_path = "./checkpoints/Stair5s_FNNInv_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch90_best0.0004392276931044069.pt"
    inverse_FNN3_path = "./checkpoints/SinusStair5s_FNNInv_layer2_seg50_sr25/TC_FNN_L2_bs16_epoch74_best0.00046465756479717257.pt"

    # LSTMbacklash
    forward_LSTMBacklash1_path = "./checkpoints/Sinus_LSTMbacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch85_best2.3004282411420718e-05.pt"
    forward_LSTMBacklash2_path = "./checkpoints/Stair5s_LSTMBacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch57_best3.1443552133764724e-05.pt"
    forward_LSTMBacklash3_path = "./checkpoints/SinusStair5s_LSTMBacklash_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch71_best4.309366758046535e-05.pt"
    inverse_LSTMBacklash1_path = "./checkpoints/Sinus_LSTMBacklashInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch98_best0.00017395542916778665.pt"
    inverse_LSTMBacklash2_path = "./checkpoints/Stair5s_LSTMBacklashInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch99_best0.00014391199280604876.pt"
    inverse_LSTMBacklash3_path = "./checkpoints/SinusStair5s_LSTMBacklashInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch86_best0.0002507692252573393.pt"

    # LSTMBacklashSum
    forward_LSTMBacklashSum1_path = "./checkpoints/Sinus_LSTMBacklashSum_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch79_best2.8625850279502753e-05.pt"
    forward_LSTMBacklashSum2_path = "./checkpoints/Stair5s_LSTMBacklashSum_layer2_seg50_sr25/TC_LSTMBacklashsum_L2_bs16_epoch59_best7.008730863162782e-05.pt"
    forward_LSTMBacklashSum3_path = "./checkpoints/SinusStair5s_LSTMBacklashSum_layer2_seg50_sr25/TC_LSTMBacklashsum_L2_bs16_epoch86_best7.081233312771069e-05.pt"
    inverse_LSTMBacklashSum1_path = "./checkpoints/Sinus_LSTMBacklashSumInv_layer2_seg50_sr25/TC_LSTMbacklash_L2_bs16_epoch89_best0.00012101194886905062.pt"
    inverse_LSTMBacklashSum2_path = "./checkpoints/Stair5s_LSTMBacklashsumInv_layer2_seg50_sr25/TC_LSTMBacklashsum_L2_bs16_epoch36_best0.0001648359628840505.pt"
    inverse_LSTMBacklashSum3_path = "./checkpoints/SinusStair5s_LSTMBacklashsumInv_layer2_seg50_sr25/TC_LSTMBacklashsum_L2_bs16_epoch43_best0.0002371085314383386.pt"


    # rate-dependent P-I model
    forward_PI1_path = "./checkpoints/Sinus_GRDPI_seg50_sr25_relu_noGamma_a4/TC_GRDPI_bs16_epoch111_best0.0002733594585028556.pt"
    forward_PI2_path = "./checkpoints/Stair5s_GRDPI_seg50_sr25_relu_noGamma_a4/TC_GRDPI_bs16_epoch208_best0.00044648534021689555.pt"
    forward_PI3_path = "./checkpoints/SinusStair5s_GRDPI_seg50_sr25_relu_noGamma_a4/TC_GRDPI_bs16_epoch106_best0.0004355512255413395.pt"
    inverse_PI1_path = "./checkpoints/Sinus_GRDPIInv_seg50_sr25_relu_inverseForm_a4/TC_GRDPI_bs16_epoch68_best0.00029903930667156387.pt"
    inverse_PI2_path = "./checkpoints/Stair5s_GRDPIInv_seg50_sr25_relu_noGamma_a4/TC_GRDPI_bs16_epoch202_best0.0002973405256246527.pt"
    inverse_PI3_path = "./checkpoints/SinusStair5s_GRDPIInv_seg50_sr25_relu_noGamma_a4/TC_GRDPI_bs16_epoch111_best0.0004364665205876085.pt"

    for k in ["forward", "inverse"]:
        for m in ["backlash", "LSTM", "GRU", "FNN", "LSTMBacklash", "LSTMBacklashSum", "PI"]:
            for i in [1, 2, 3]:
                model_path_name = k+"_"+m+str(i)+"_path"
                model_path = globals()[model_path_name]
                acc_file = model_path.split("bs16")[0] + "acc_bs16_epoch300.pkl"
                try:
                    acc = pickle.load(open(acc_file, "rb"))
                except:
                    print(model_path)
                    continue
                plt.figure()
                plt.plot(acc['train'], '-', label='train')
                plt.plot(acc['test'], '-', label='validation')
                plt.xlabel('Epochs')
                plt.ylabel('loss')
                plt.legend()
                plt.ylim([0, 0.0005])
                plt.savefig('./figures/training_error/loss_{}_{}_{}.jpg'.format(k,m,i))
                # plt.show()