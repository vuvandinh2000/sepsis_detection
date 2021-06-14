import os
import pandas as pd
import numpy as np
from itertools import chain

sep_index = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST',
             'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
             'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
             'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC', 'Platelets']
con_index = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']

def feature_informative_missingness(case, sep_columns):
    """
    informative missingness features reflecting measurement frequency
        or time interval of raw variables
    differential features, defined by calculating the difference between
        the current record and the previous measurement value
    :param case: one patient's EHR data
    :param sep_columns: selected variables
    :return: calculated features
    """
    temp_data = np.array(case)
    for sep_column in sep_columns:
        sep_data = np.array(case[sep_column])
        """
        np.where(~np.isnan(sep_data))[0] trả về 1 mảng các index của các phần tử notnull --> notnan_pos thì đúng hơn
        """
        nan_pos = np.where(~np.isnan(sep_data))[0]
        # Measurement frequency sequence
        """
        Số lần lặp lại (tần suất)
        VD: sep_data = [nan,  2.,  2., nan, nan,  1., nan, nan,  0., nan, nan,  2., nan,
                        3., nan,  3., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
                        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
                        nan, nan, nan, nan, nan]
        thì interval_f1=[0., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4., 5., 5.,
                        6., 6., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 
                        7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 
                        7., 7., 7., 7., 7.]
        """
        interval_f1 = sep_data.copy()
        # Measurement time interval
        """
        Số lần xuất hiện (times)
        VD: sep_data = [nan,  2.,  2., nan, nan,  1., nan, nan,  0., nan, nan,  2., nan,
                        3., nan,  3., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
                        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
                        nan, nan, nan, nan, nan]
        thì interval_f2=[-1.,  0.,  0.,  1.,  2.,  0.,  1.,  2.,  0.,  1.,  2.,  0.,  1.,
                        0.,  1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.,
                        11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.,
                        24., 25., 26., 27., 28.]
        """
        interval_f2 = sep_data.copy()
        if len(nan_pos) == 0: #Tất cả các phần tử là nan
            interval_f1[:] = 0
            temp_data = np.column_stack((temp_data, interval_f1))
            interval_f2[:] = -1
            temp_data = np.column_stack((temp_data, interval_f2))
        else:
            interval_f1[: nan_pos[0]] = 0
            for p in range(len(nan_pos)-1):
                interval_f1[nan_pos[p]: nan_pos[p+1]] = p + 1
            interval_f1[nan_pos[-1]:] = len(nan_pos)
            temp_data = np.column_stack((temp_data, interval_f1))

            interval_f2[:nan_pos[0]] = -1
            for q in range(len(nan_pos) - 1):
                length = nan_pos[q+1] - nan_pos[q]
                for l in range(length):
                    interval_f2[nan_pos[q] + l] = l

            length = len(case) - nan_pos[-1]
            for l in range(length):
                interval_f2[nan_pos[-1] + l] = l
            temp_data = np.column_stack((temp_data, interval_f2))

        # Differential features
        """
        features tính toán sự khác biệt giữa các điểm dữ liệu
        VD: sep_data = [nan,  2.,  2., nan, nan,  1., nan, nan,  0., nan, nan,  2., nan,
                        3., nan,  3., nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
                        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
                        nan, nan, nan, nan, nan]
        thì diff_f =   [nan, nan,  0.,  0.,  0., -1., -1., -1., -1., -1., -1.,  2.,  2.,
                        1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                        0.,  0.,  0.,  0.,  0.]
        """
        diff_f = sep_data.copy()
        diff_f = diff_f.astype(float)
        if len(nan_pos) <= 1:
            diff_f[:] = np.NaN
            temp_data = np.column_stack((temp_data, diff_f))
        else:
            diff_f[:nan_pos[1]] = np.NaN
            for p in range(1, len(nan_pos)-1):
                diff_f[nan_pos[p] : nan_pos[p+1]] = sep_data[nan_pos[p]] - sep_data[nan_pos[p-1]]
            diff_f[nan_pos[-1]:] = sep_data[nan_pos[-1]] - sep_data[nan_pos[-2]]
            temp_data = np.column_stack((temp_data, diff_f))

    return temp_data

def feature_slide_window(temp, con_index):
    """
    Calculate dynamic statistics in a six-hour sliding window
    :param temp: data after using a forward-filling strategy
    :param con_index: selected variables
    :return: time-series features
    """
    """
    Chỉ chọn ra những trường data ảnh hưởng lớn đến sepsis thông qua biến con_index
    [0, 1, 3, 4, 6] = ['HR' (Nhịp tim), 'O2Sat' (Độ bão hoà oxi), 'SBP' (Huyết áp tâm thu), 
                        'MAP' (Huyết áp động mạch trung bình), 'Resp' (Tần số hô hấp/Nhịp thở)]
    """
    sepdata = temp[:, con_index]

    """
    Tạo ra 4 LIST toàn 0 có shape = (len(sepdata), (len(con_index))
    """
    max_values = [[0 for col in range(len(sepdata))]
                  for row in range(len(con_index))]
    min_values = [[0 for col in range(len(sepdata))]
                  for row in range(len(con_index))]
    mean_values = [[0 for col in range(len(sepdata))]
                   for row in range(len(con_index))]
    median_values = [[0 for col in range(len(sepdata))]
                     for row in range(len(con_index))]
    std_values = [[0 for col in range(len(sepdata))]
                  for row in range(len(con_index))]
    diff_std_values = [[0 for col in range(len(sepdata))]
                       for row in range(len(con_index))]

    for i in range(len(sepdata)): #Duyệt qua 44h data thăm khám
        if i < 6:
            """
            Chỉ đơn giản là lấy ra window = 6 từ data và bù lại phần thời gian tương lai chưa biết bằng thời điểm i đang duyệt tới
            VD:
            Có sepdata = [[   nan,    nan,    nan,    nan,    nan],
                            [ 94.  , 100.  , 114.  ,  82.  ,  12.  ],   (1)
                            [ 94.  ,  97.5 , 105.  ,  76.5 ,  12.  ],   (2)
                            [ 88.5 ,  99.5 ,  97.5 ,  67.5 ,  12.  ],   (3)
                            [ 84.  ,  98.  , 106.  ,  73.  ,  14.5 ],   (4)
                            [ 86.5 ,  97.  , 111.5 ,  72.5 ,  14.  ],   (5)
                            [ 88.  ,  98.  , 112.  ,  74.  ,  17.  ],   (6)
                            [ 84.  ,  96.  , 113.  ,  72.  ,  18.5 ],   (7)
                            [ 92.  ,  96.  , 121.  ,  80.  ,  21.  ],   (8)
                            [ 93.  ,  95.  , 118.  ,  75.  ,  21.  ]]   (9)

            thì với i = 2:
                window = [[  nan,   nan,   nan,   nan,   nan],          (1)
                            [ 94. , 100. , 114. ,  82. ,  12. ],        (2)
                            [ 94. ,  97.5, 105. ,  76.5,  12. ],        (2)
                            [ 94. ,  97.5, 105. ,  76.5,  12. ],        (2)
                            [ 94. ,  97.5, 105. ,  76.5,  12. ],        (2)
                            [ 94. ,  97.5, 105. ,  76.5,  12. ]]        (2)

            với i = 5:
                window = [[  nan,   nan,   nan,   nan,   nan],          (1)
                            [ 94. , 100. , 114. ,  82. ,  12. ],        (2)
                            [ 94. ,  97.5, 105. ,  76.5,  12. ],        (3)
                            [ 88.5,  99.5,  97.5,  67.5,  12. ],        (4)
                            [ 84. ,  98. , 106. ,  73. ,  14.5],        (5)
                            [ 86.5,  97. , 111.5,  72.5,  14. ]]        (6)
            """
            win_data = sepdata[0:i + 1]
            for ii in range(6 - i):
                win_data = np.row_stack((win_data, sepdata[i]))
        else:
            # i-5 ???????
            win_data = sepdata[i - 6: i + 1]

        for j in range(len(con_index)): #Duyệt qua tất cả thuộc tính
            dat = win_data[:, j]    #Lấy từng thuộc tính ra
            if len(np.where(~np.isnan(dat))[0]) == 0: #nếu tất cả value trong cột thuộc tính đó là NaN
                max_values[j][i] = np.nan
                min_values[j][i] = np.nan
                mean_values[j][i] = np.nan
                median_values[j][i] = np.nan
                std_values[j][i] = np.nan
                diff_std_values[j][i] = np.nan
            else:
                #nanmax, nanmin,... -> bỏ qua NaN để tính max, min,...
                """
                Tính matrix max_values, min_values,... trên từng  cửa sổ
                VD:
                sepdata = [[   nan,    nan,    nan,    nan,    nan],
                            [ 94.  , 100.  , 114.  ,  82.  ,  12.  ],
                            [ 94.  ,  97.5 , 105.  ,  76.5 ,  12.  ],
                            [ 88.5 ,  99.5 ,  97.5 ,  67.5 ,  12.  ],
                            [ 84.  ,  98.  , 106.  ,  73.  ,  14.5 ],
                            [ 86.5 ,  97.  , 111.5 ,  72.5 ,  14.  ],
                            [ 88.  ,  98.  , 112.  ,  74.  ,  17.  ],
                            [ 84.  ,  96.  , 113.  ,  72.  ,  18.5 ],
                            [ 92.  ,  96.  , 121.  ,  80.  ,  21.  ],
                            [ 93.  ,  95.  , 118.  ,  75.  ,  21.  ],
                            [ 89.  ,  93.  , 125.  ,  75.  ,  23.  ],
                            [ 92.  ,  92.  , 100.  ,  68.  ,  21.  ],
                            [ 82.  ,  92.  , 119.  ,  73.  ,  24.  ],
                            [ 85.  ,  93.  , 129.  ,  77.  ,  21.  ],
                            [ 93.  ,  94.  ,  91.  ,  62.  ,  21.  ],
                            [ 94.  ,  95.  , 113.  ,  73.  ,  22.  ],
                            [ 88.  ,  92.  , 118.  ,  72.  ,  25.  ],
                            [ 89.  ,  93.  , 136.  ,  76.  ,  20.  ],
                            [ 97.  ,  96.  , 121.  ,  72.  ,  16.  ],
                            [ 87.  ,  97.  , 122.  ,  74.  ,  14.  ],
                            [ 85.  ,  96.  , 114.  ,  75.  ,  14.  ],
                            [ 89.  ,  94.  , 125.  ,  76.  ,  17.  ],
                            [ 86.5 ,  93.5 , 113.  ,  65.  ,  14.5 ],
                            [ 82.5 ,  92.5 ,  98.  ,  57.  ,  17.  ],
                            [ 83.  ,  93.  , 100.  ,  56.  ,  16.  ],
                            [ 78.  ,  96.  , 106.  ,  60.  ,  18.  ],
                            [ 77.5 ,  94.5 , 100.  ,  63.  ,  19.  ],
                            [ 76.  ,  95.  , 112.  ,  66.  ,  16.  ],
                            [ 73.5 ,  95.5 , 122.  ,  70.  ,  16.  ],
                            [ 84.5 ,  95.5 , 102.5 ,  61.67,  16.5 ],
                            [ 79.  ,  95.  ,  90.  ,  59.67,  15.  ],
                            [ 77.  ,  96.  ,  93.  ,  65.  ,  14.  ],
                            [ 78.  ,  94.  ,  83.  ,  50.33,  19.  ],
                            [ 78.  ,  94.  , 115.  ,  64.33,  16.  ],
                            [ 76.  ,  96.  , 133.  ,  87.  ,  14.  ],
                            [ 80.  ,  97.  , 127.  ,  81.67,  18.  ],
                            [ 74.  ,  95.  ,  97.  ,  62.33,  11.  ],
                            [ 79.  ,  98.  , 121.  ,  93.67,  26.  ],
                            [ 79.  ,  93.  , 126.  ,  81.33,  19.  ],
                            [ 89.  ,  93.  , 110.  ,  72.67,  24.  ],
                            [ 84.  ,  93.  , 114.  ,  77.33,  20.  ],
                            [ 86.  ,  93.  , 117.  ,  76.33,  19.  ],
                            [ 89.  ,  94.  , 104.  ,  68.67,  29.  ],
                            [ 82.  ,  97.  , 120.  ,  86.  ,  18.  ]]
                
                Ta có max_values = [[   nan,    nan,    nan,    nan,    nan],
                                    [ 94.  , 100.  , 114.  ,  82.  ,  12.  ],
                                    [ 94.  , 100.  , 114.  ,  82.  ,  12.  ],
                                    [ 94.  , 100.  , 114.  ,  82.  ,  12.  ],
                                    [ 94.  , 100.  , 114.  ,  82.  ,  14.5 ],
                                    [ 94.  , 100.  , 114.  ,  82.  ,  14.5 ],
                                    [ 94.  , 100.  , 114.  ,  82.  ,  17.  ],
                                    [ 94.  , 100.  , 114.  ,  82.  ,  18.5 ],
                                    [ 94.  ,  99.5 , 121.  ,  80.  ,  21.  ],
                                    [ 93.  ,  99.5 , 121.  ,  80.  ,  21.  ],
                                    [ 93.  ,  98.  , 125.  ,  80.  ,  23.  ],
                                    [ 93.  ,  98.  , 125.  ,  80.  ,  23.  ],
                                    [ 93.  ,  98.  , 125.  ,  80.  ,  24.  ],
                                    [ 93.  ,  96.  , 129.  ,  80.  ,  24.  ],
                                    [ 93.  ,  96.  , 129.  ,  80.  ,  24.  ],
                                    [ 94.  ,  95.  , 129.  ,  77.  ,  24.  ],
                                    [ 94.  ,  95.  , 129.  ,  77.  ,  25.  ],
                                    [ 94.  ,  95.  , 136.  ,  77.  ,  25.  ],
                                    [ 97.  ,  96.  , 136.  ,  77.  ,  25.  ],
                                    [ 97.  ,  97.  , 136.  ,  77.  ,  25.  ],
                                    [ 97.  ,  97.  , 136.  ,  76.  ,  25.  ],
                                    [ 97.  ,  97.  , 136.  ,  76.  ,  25.  ],
                                    [ 97.  ,  97.  , 136.  ,  76.  ,  25.  ],
                                    [ 97.  ,  97.  , 136.  ,  76.  ,  20.  ],
                                    [ 97.  ,  97.  , 125.  ,  76.  ,  17.  ],
                                    [ 89.  ,  97.  , 125.  ,  76.  ,  18.  ],
                                    [ 89.  ,  96.  , 125.  ,  76.  ,  19.  ],
                                    [ 89.  ,  96.  , 125.  ,  76.  ,  19.  ],
                                    [ 86.5 ,  96.  , 122.  ,  70.  ,  19.  ],
                                    [ 84.5 ,  96.  , 122.  ,  70.  ,  19.  ],
                                    [ 84.5 ,  96.  , 122.  ,  70.  ,  19.  ],
                                    [ 84.5 ,  96.  , 122.  ,  70.  ,  19.  ],
                                    [ 84.5 ,  96.  , 122.  ,  70.  ,  19.  ],
                                    [ 84.5 ,  96.  , 122.  ,  70.  ,  19.  ],
                                    [ 84.5 ,  96.  , 133.  ,  87.  ,  19.  ],
                                    [ 84.5 ,  97.  , 133.  ,  87.  ,  19.  ],
                                    [ 80.  ,  97.  , 133.  ,  87.  ,  19.  ],
                                    [ 80.  ,  98.  , 133.  ,  93.67,  26.  ],
                                    [ 80.  ,  98.  , 133.  ,  93.67,  26.  ],
                                    [ 89.  ,  98.  , 133.  ,  93.67,  26.  ],
                                    [ 89.  ,  98.  , 133.  ,  93.67,  26.  ],
                                    [ 89.  ,  98.  , 127.  ,  93.67,  26.  ],
                                    [ 89.  ,  98.  , 126.  ,  93.67,  29.  ],
                                    [ 89.  ,  98.  , 126.  ,  93.67,  29.  ]]

                Note: sở dĩ vector [ 94.  , 100.  , 114.  ,  82.  ,  12.  ] lặp lại tới 8 lần bởi 6 cửa sổ đầu ta thực hiện broadcasting giá trị
                + 2 cửa sổ không phải broadcasting nhưng tồn tại vector này
                """
                max_values[j][i] = np.nanmax(dat)
                min_values[j][i] = np.nanmin(dat)
                mean_values[j][i] = np.nanmean(dat)
                median_values[j][i] = np.nanmedian(dat)
                std_values[j][i] = np.nanstd(dat)
                diff_std_values[j][i] = np.std(np.diff(dat))

    win_features = list(chain(max_values, min_values, mean_values,
                              median_values, std_values, diff_std_values))
    win_features = (np.array(win_features)).T

    return win_features

def feature_empiric_score(dat):
    """
    empiric features scoring for
    heart rate (HR), systolic blood pressure (SBP), mean arterial pressure (MAP),
    respiration rate (Resp), temperature (Temp), creatinine, platelets and total bilirubin
    according to the scoring systems of NEWS, SOFA and qSOFA
    """
    scores = np.zeros((len(dat), 8))
    for ii in range(len(dat)): #Duyệt qua tất cả giờ dữ liệu (VD: 44h dữ liệu)
        HR = dat[ii, 0] #Phần tử đầu tiên của giờ dữ liệu thứ ii --> HR
        """
        Heart rate (Nhịp tim), ở đây để score như sau:
        score:          3      1      0     1     2        3
        range:      <--------|----|------|-----|-------|---------> (bpm)
        value:              40   50    90    110     130

        """
        if HR == np.nan:
            HR_score = np.nan
        elif (HR <= 40) | (HR >= 131):
            HR_score = 3
        elif 111 <= HR <= 130:
            HR_score = 2
        elif (41 <= HR <= 50) | (91 <= HR <= 110):
            HR_score = 1
        else: # 50 < HR < 90
            HR_score = 0
        scores[ii, 0] = HR_score

        Temp = dat[ii, 2]
        """
        Temp (Nhiệt độ cơ thể), ở đây để score như sau:
        score:          3      1      0         1      2
        range:      <--------|----|---------|-------|---------> (degree Celcius)
        value:              35   36         38      39
        """
        if Temp == np.nan:
            Temp_score = np.nan
        elif Temp <= 35:
            Temp_score = 3
        elif Temp >= 39.1:
            Temp_score = 2
        elif (35.1 <= Temp <= 36.0) | (38.1 <= Temp <= 39.0):
            Temp_score = 1
        else:
            Temp_score = 0
        scores[ii, 1] = Temp_score

        Resp = dat[ii, 6]
        """
        Resp (Respiration rate - Tần số hô hấp/Nhịp thở) 
        score:          3      1      0         2      3
        range:      <--------|----|---------|-------|---------> (breaths/min)
        value:               8   11         21      25
        """
        if Resp == np.nan:
            Resp_score = np.nan
        elif (Resp < 8) | (Resp > 25):
            Resp_score = 3
        elif 21 <= Resp <= 24:
            Resp_score = 2
        elif 9 <= Resp <= 11:
            Resp_score = 1
        else:
            Resp_score = 0
        scores[ii, 2] = Resp_score

        Creatinine = dat[ii, 19]
        """
        Creatinin máu - mg/dL 
        score:          0      1       2        3
        range:      <--------|----|---------|-------------> (mg/dL)
        value:              1.2   2        3.5
        """
        if Creatinine == np.nan:
            Creatinine_score = np.nan
        elif Creatinine < 1.2:
            Creatinine_score = 0
        elif Creatinine < 2:
            Creatinine_score = 1
        elif Creatinine < 3.5:
            Creatinine_score = 2
        else:
            Creatinine_score = 3
        scores[ii, 3] = Creatinine_score

        MAP = dat[ii, 4]
        """
        MAP (Mean arterial pressure – Huyết áp động mạch trung bình)
        score:          1      0
        range:      <-----|---------> (mmHg)
        value:            70
        """
        if MAP == np.nan:
            MAP_score = np.nan
        elif MAP >= 70:
            MAP_score = 0
        else:
            MAP_score = 1
        scores[ii, 4] = MAP_score

        SBP = dat[ii, 3]
        Resp = dat[ii, 6]
        """
        qsofa = SBP (Systolic BP – Huyết áp tâm thu) && Resp Resp (Respiration rate – Tần số hô hấp/Nhịp thở)
        score:                 0                  1
        range:      <-----------------------|-----------------> (breaths/min)
        value:               (SBP <= 100) & (Resp >= 22)
        """
        if SBP + Resp == np.nan:
            qsofa = np.nan
        elif (SBP <= 100) & (Resp >= 22):
            qsofa = 1
        else:
            qsofa = 0
        scores[ii, 5] = qsofa

        Platelets = dat[ii, 30]
        """
        Platelets (Mật độ tiểu cầu - Platenet count)
        score:          3      2       1        0
        range:      <--------|----|---------|-------------> (count/mL)
        value:               50   100       150
        """
        if Platelets == np.nan:
            Platelets_score = np.nan
        elif Platelets <= 50:
            Platelets_score = 3
        elif Platelets <= 100:
            Platelets_score = 2
        elif Platelets <= 150:
            Platelets_score = 1
        else:
            Platelets_score = 0
        scores[ii, 6] = Platelets_score

        Bilirubin = dat[ii, 25]
        """
        Bilirubin total (Bilirubin toàn phần)
        score:          0      1       2          3
        range:      <--------|----|---------|-------------> (mg/dL)
        value:               1.2  2         6
        """
        if Bilirubin == np.nan:
            Bilirubin_score = np.nan
        elif Bilirubin < 1.2:
            Bilirubin_score = 0
        elif Bilirubin < 2:
            Bilirubin_score = 1
        elif Bilirubin < 6:
            Bilirubin_score = 2
        else:
            Bilirubin_score = 3
        scores[ii, 7] = Bilirubin_score

    return scores

def feature_extraction(case):
    labels = np.array(case['SepsisLabel'])
    # drop three variables due to their massive missing values
    pid = case.drop(columns=['Bilirubin_direct', 'TroponinI', 'Fibrinogen', 'SepsisLabel'])

    """
    con_index + sep_index là những features thường hay bị NaN
    --> những features còn lại không có giá trị NaN --> không phải xử lý missing info
    """
    temp_data = feature_informative_missingness(pid, con_index + sep_index)
    #npArray --> pandas Frame
    temp = pd.DataFrame(temp_data)
    # Missing values used a forward-filling strategy
    temp = temp.fillna(method='ffill')
    # 62 informative missingness features, 31 differential features
    # and 37 raw variables
    feature_A = np.array(temp)
    # Statistics in a six-hour window for the selected measurements
    # [0, 1, 3, 4, 6] = ['HR', 'O2Sat', 'SBP', 'MAP', 'Resp']
    # 30 statistical features in the window
    feature_B = feature_slide_window(feature_A, [0, 1, 3, 4, 6])
    # 8 empiric features
    feature_C = feature_empiric_score(feature_A)
    # A total of 168 features were obtained
    features = np.column_stack((feature_A, feature_B, feature_C))

    return  features, labels

def data_process(data_set, data_path_dir):
    """
    Feature matrix across all patients in the data_set
    """
    frames_features = []
    frames_labels = []
    for psv in data_set:
        patient = pd.read_csv(os.path.join(data_path_dir, psv), sep='|')
        features, labels = feature_extraction(patient)
        features = pd.DataFrame(features)
        labels = pd.DataFrame(labels)
        frames_features.append(features)
        frames_labels.append(labels)

    dat_features = np.array(pd.concat(frames_features))
    dat_labels = (np.array(pd.concat(frames_labels)))[:, 0]

    index = [i for i in range(len(dat_labels))]
    np.random.shuffle(index)
    dat_features = dat_features[index]
    dat_labels = dat_labels[index]

    return dat_features, dat_labels
