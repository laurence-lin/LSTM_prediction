# LSTM_prediction
This is the implementation of LSTM on energy prediction UCI dataset during practicing SVR

Dataset: UCI house energy Appliance regression dataset

For LSTM:

Applying keras LSTM model, only one LSTM cell is enough. 

Time step = 7 days, features = 1 (only past energy value is considered here), output size = 250 to a dense layer, which is final output.


Discussion:

LSTM could get a good prediction result on training set and testing set, while SVR could only obtain not very good performance.

Total data size is about 19000, while work only 4000 for SVR, the performance is still bad comparing to LSTM.

Note: Without scaling, LSTM could get bad result due to too slow converge speed. Apply MinMaxScaler and then it showed good performance. Data normalization is a must.

LSTM training performance:

![img](https://github.com/laurence-lin/LSTM_prediction/blob/master/Result.png)

LSTM testing performance:

![img](https://github.com/laurence-lin/LSTM_prediction/blob/master/Test%20result.png)
