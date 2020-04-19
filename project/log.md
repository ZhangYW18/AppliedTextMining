```shell
% python model.py
Using TensorFlow backend.
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:523: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:524: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:532: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])
sys:1: DtypeWarning: Columns (1,2) have mixed types. Specify dtype option on import or set low_memory=False.
     target          id                          date      flag             user                                               text
NaN  target          id                          date      flag             user                                               text
0.0       0  1467810369  Mon Apr 06 22:19:45 PDT 2009  NO_QUERY  _TheSpecialOne_       awww bummer shoulda got david carr third day
1.0       0  1467810672  Mon Apr 06 22:19:49 PDT 2009  NO_QUERY    scotthamilton  upset updat facebook text might cri result sch...
2.0       0  1467810917  Mon Apr 06 22:19:53 PDT 2009  NO_QUERY         mattycus       dive mani time ball manag save rest go bound
3.0       0  1467811184  Mon Apr 06 22:19:57 PDT 2009  NO_QUERY          ElleCTF                    whole bodi feel itchi like fire
Read csv successfully, number of tweets: 1600001

TRAIN size: 1280000
TEST size: 320000
          target          id                          date      flag             user                                               text
1374558.0      4  2051457557  Fri Jun 05 22:04:23 PDT 2009  NO_QUERY    JGoldsborough  ya quot like palm pre touchston charger readyn...
1389115.0      4  2053083567  Sat Jun 06 03:12:21 PDT 2009  NO_QUERY           Psioui              felt earthquak afternoon seem epicent
1137831.0      4  1976779404  Sat May 30 19:02:49 PDT 2009  NO_QUERY        adriville                             ruffl shirt like likey
790714.0       0  2325739990  Thu Jun 25 05:59:18 PDT 2009  NO_QUERY       Blondie128  pretti bad night crappi morn fml buttfac didnt...
1117911.0      4  1973503391  Sat May 30 11:16:35 PDT 2009  NO_QUERY         khrabrov                                    yeah clear view
...          ...         ...                           ...       ...              ...                                                ...
259178.0       0  1985361990  Sun May 31 16:57:39 PDT 2009  NO_QUERY      lutheasalom                 song middl chang want born arghhhh
1414414.0      4  2057029784  Sat Jun 06 12:14:24 PDT 2009  NO_QUERY           beeluz                                          good luck
131932.0       0  1835639354  Mon May 18 06:26:21 PDT 2009  NO_QUERY      lordmuttley                                      rather averag
671155.0       0  2246780174  Fri Jun 19 18:06:46 PDT 2009  NO_QUERY  MizSadittyFancy  pickin misstinayao waitin sadittysash hurri od...
121958.0       0  1833617173  Sun May 17 23:52:31 PDT 2009  NO_QUERY          dindahh              home studi math wooot im go fail shit

[1280000 rows x 6 columns]
TRAIN size: 1280000
TEST size: 320000

model_test (most similar words of `sad`) :

[('depress', 0.5953245759010315), ('upset', 0.5866713523864746), ('sadder', 0.47332102060317993), ('bum', 0.4693984091281891), ('cri', 0.4454968571662903), ('devast', 0.43920642137527466), ('bittersweet', 0.4272928237915039), ('saad', 0.4189494848251343), ('sadd', 0.41145211458206177), ('disappoint', 0.4103020429611206)]

x_train (1280000, 300)
y_train (1280000, 1)
x_test (320000, 300)
y_test (320000, 1)

Total words 232983
Vocab size 232983

embedding_matrix (232983, 300)


2020-04-18 12:11:01.636421: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 300, 300)          69894900  
_________________________________________________________________
dropout_1 (Dropout)          (None, 300, 300)          0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 100)               160400    
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 101       
=================================================================
Total params: 70,055,401
Trainable params: 160,501
Non-trainable params: 69,894,900
_________________________________________________________________
Train on 1152000 samples, validate on 128000 samples
Epoch 1/8
1152000/1152000 [==============================] - 11183s 10ms/step - loss: 0.5070 - acc: 0.7485 - val_loss: 0.4704 - val_acc: 0.7767
Epoch 2/8
1152000/1152000 [==============================] - 3471s 3ms/step - loss: 0.4869 - acc: 0.7626 - val_loss: 0.4621 - val_acc: 0.7798
Epoch 3/8
1152000/1152000 [==============================] - 5087s 4ms/step - loss: 0.4837 - acc: 0.7648 - val_loss: 0.4602 - val_acc: 0.7815
Epoch 4/8
1152000/1152000 [==============================] - 3521s 3ms/step - loss: 0.4777 - acc: 0.7689 - val_loss: 0.4586 - val_acc: 0.7828
Epoch 5/8
1152000/1152000 [==============================] - 4646s 4ms/step - loss: 0.4927 - acc: 0.7588 - val_loss: 0.4641 - val_acc: 0.7790
Epoch 6/8
1152000/1152000 [==============================] - 3567s 3ms/step - loss: 0.4837 - acc: 0.7646 - val_loss: 0.4601 - val_acc: 0.7810
Epoch 7/8
1152000/1152000 [==============================] - 3503s 3ms/step - loss: 0.4776 - acc: 0.7684 - val_loss: 0.4581 - val_acc: 0.7824
Epoch 8/8
1152000/1152000 [==============================] - 3487s 3ms/step - loss: 0.4751 - acc: 0.7701 - val_loss: 0.4558 - val_acc: 0.7844
320000/320000 [==============================] - 387s 1ms/step

ACCURACY:  0.7837125
LOSS: 0.4554874319076538

```



