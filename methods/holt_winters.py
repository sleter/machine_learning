import matplotlib.pyplot as plt
import numpy as np
from data_generator import DataGenerator
from sklearn.metrics import mean_squared_error
import os, shutil, errno, json

class HoltWinters:
    def __init__(self):
        pass

    def initial_trend(self, series, slen):
        sum = 0.0
        for i in range(slen):
            sum += float(series[i+slen] - series[i]) / slen
        return sum / slen

    def initial_seasonal_components(self, series, slen):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(series)/slen)
        # compute season averages
        for j in range(n_seasons):
            season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
        # compute initial values
        for i in range(slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals

    def triple_exponential_smoothing(self, series, slen, alpha, beta, gamma, n_preds):
        """ 
            series - input data 
            slen - season length
            n_preds - number of points to forecast
            alpha, beta, gamma - chosen for the smallest SSE
        """
        result = []
        seasonals = self.initial_seasonal_components(series, slen)
        for i in range(len(series)+n_preds):
            if i == 0: # initial values
                smooth = series[0]
                trend = self.initial_trend(series, slen)
                result.append(series[0])
                continue
            if i >= len(series): # forecasting
                m = i - len(series) + 1
                result.append((smooth + m*trend) + seasonals[i%slen])
            else:
                val = series[i]
                last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
                trend = beta * (smooth-last_smooth) + (1-beta)*trend
                seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
                result.append(smooth+trend+seasonals[i%slen])
        return result

    def show_output(self, array1, array2=[], array3=[]):
        plt.subplot(311)
        plt.plot(array1, 'bo', array1[:100], 'r--')
        plt.subplot(312)
        plt.plot(array3, 'bo-')
        plt.subplot(313)
        plt.plot(array2, 'go-')
        plt.show()

    def calculate_error(self, test, pred):
        all_rmse = []

        for t, p in zip(test, pred[:len(test)]):
            ts = []
            ts.append(t)
            pr = []
            pr.append(p)
            err = mean_squared_error(ts,pr)
            all_rmse.append(err)

        mse = mean_squared_error(test, pred[:len(test)])
        return all_rmse, mse

    def calulate_loss(self, array,test, training_data_amount, points_to_predict, error, step=50):
        self.array_r = self.triple_exponential_smoothing(array[:training_data_amount], 100, 0.516, 0.0029, 0.093, points_to_predict)

        reg_mse, mse1   = self.calculate_error(test.y_regular[:training_data_amount+points_to_predict], self.array_r)
        nois_mse, mse1  = self.calculate_error(test.y_noise[:training_data_amount+points_to_predict], self.array_r)
        multi_mse, mse1 = self.calculate_error(test.y_multiplied[:training_data_amount+points_to_predict], self.array_r)
        const_mse, mse1 = self.calculate_error(test.y_const[:training_data_amount+points_to_predict], self.array_r)

        
        # self.show_output(self.array_r, reg_mse, test.y_regular)
        # self.show_output(self.array_r, nois_mse, test.y_noise)
        # self.show_output(self.array_r, multi_mse, test.y_multiplied)
        # self.show_output(self.array_r, const_mse, test.y_const)

        return reg_mse,nois_mse,multi_mse,const_mse

        
    def return_binary_anomaly_table(self, array_out):
        output_vector = []
        for i in range(0,len(array_out)):
            mean = np.mean(array_out[i:i+30])
            print(mean)
            if array_out[i] < mean:
                output_vector.append(0)
            else:
                output_vector.append(1)
        return output_vector

    def binary_representation(self, array, error):
        output = []
        for i in array:
            if i>error: output.append(1)
            else: output.append(0)
        return output

    def out(self, array, test, size):
        array_y, array_n, array_m, array_c = self.calulate_loss(array, test, 500, size-500, 0.1, step=10)
        detector = {
            'HW Reg': {
                'pred': self.binary_representation(array_y, 0.05)},
            'HW Multi': {
                'pred': self.binary_representation(array_m, 0.05)},
            'HW Noise': {
                'pred': self.binary_representation(array_n, 0.05)},
            'HW Const': {
                'pred': self.binary_representation(array_c, 0.05)}
            }

        self.save_to_file(detector)
        #return detector, detector['HW Reg']['pred'], detector['HW Multi']['pred'], detector['HW Noise']['pred'], detector['HW Const']['pred'], self.array_r

    def save_to_file(self, detector):
        try:
            shutil.rmtree('models/Holt_Winters/')
        except FileNotFoundError:
            pass
        if not os.path.exists("models/Holt_Winters"):
            try:
                os.makedirs("models/Holt_Winters")
            except OSError:
                if OSError.errno != errno.EEXIST:
                    raise
        file_obj = open("models/Holt_Winters/dictionary.json", 'w')
        file_obj.write(json.dumps(detector))




    def read_HW(self):
        with open('models/Holt_Winters/dictionary.json', 'r') as file_obj:
            data = json.load(file_obj)
        return data, data['HW Reg']['pred'], data['HW Multi']['pred'], data['HW Noise']['pred'], data['HW Const']['pred'] 
            

        

if __name__ == "__main__":
    dg = DataGenerator(step=0.1, anomaly_length=50, anomaly_quantity=3)
    hw = HoltWinters()
    hw.out(dg.y_regular, dg, 2500)
    #hw.read_HW()
    