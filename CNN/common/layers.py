# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from common.functions import *
from common.util import im2col, col2im
from collections import OrderedDict, defaultdict

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

    
class LeakyReLU:
    def __init__(self, alpha=0.1):
        self.mask = None
        self.alpha = alpha
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] *= self.alpha
        return out

    def backward(self, dout):
        dout[self.mask] *= self.alpha
        dx = dout
        return dx
    
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応(画像形式のxに対応させる)
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        
        # 初期値
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        """
        順伝播
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        """
        逆伝播
        伝播する値をバッチサイズで割ること
        dout=1は、他のレイヤと同じ使い方ができるように設定しているダミー変数
        """
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

class MeanSquaredLoss:
    def __init__(self):
        self.loss = None
        self.y = None # 2乗和誤差損失関数の出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = x #恒等関数
        self.loss = mean_squared_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
    
    
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    def __init__(self, gamma, beta, rho=0.9, moving_mean=None, moving_var=None):
        self.gamma = gamma # スケールさせるためのパラメータ, 学習によって更新させる.
        self.beta = beta # シフトさせるためのパラメータ, 学習によって更新させる
        self.rho = rho # 移動平均を算出する際に使用する係数

        # 予測時に使用する平均と分散
        self.moving_mean = moving_mean # muの移動平均
        self.moving_var = moving_var        # varの移動平均
        
        # 計算中に算出される値を保持しておく変数群
        self.batch_size = None
        self.x_mu = None
        self.x_std = None        
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        """
        順伝播計算
        x :  CNNの場合は4次元、全結合層の場合は2次元  
        """
        if x.ndim == 4:
            """
            画像形式の場合
            """
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1) # NHWCに入れ替え
            x = x.reshape(N*H*W, C) # (N*H*W,C)の2次元配列に変換
            out = self.__forward(x, train_flg)
            out = out.reshape(N, H, W, C)# 4次元配列に変換
            out = out.transpose(0, 3, 1, 2) # 軸をNCHWに入れ替え
        elif x.ndim == 2:
            """
            画像形式以外の場合
            """
            out = self.__forward(x, train_flg)           
            
        return out
            
    def __forward(self, x, train_flg, epsilon=1e-8):
        """
        x : 入力. n×dの行列. nはあるミニバッチのバッチサイズ. dは手前の層のノード数
        """
        if (self.moving_mean is None) or (self.moving_var is None):
            N, D = x.shape
            self.moving_mean = np.zeros(D)
            self.moving_var = np.zeros(D)
                        
        if train_flg:
            """
            学習時
            """
            # 入力xについて、nの方向に平均値を算出. 
            mu = np.mean(x, axis=0) # 要素数d個のベクトル
            
            # 入力xから平均値を引く
            x_mu = x - mu   # n*d行列
            
            # 入力xの分散を求める
            var = np.mean(x_mu**2, axis=0)  # 要素数d個のベクトル
            
            # 入力xの標準偏差を求める(epsilonを足してから標準偏差を求める)
            std = np.sqrt(var + epsilon)  # 要素数d個のベクトル
            
            # 標準化
            x_std = x_mu / std  # n*d行列
            
            # 値を保持しておく
            self.batch_size = x.shape[0]
            self.x_mu = x_mu
            self.x_std = x_std
            self.std = std
            self.moving_mean = self.rho * self.moving_mean + (1-self.rho) * mu
            self.moving_var = self.rho * self.moving_var + (1-self.rho) * var            
        else:
            """
            予測時
            """
            x_mu = x - self.moving_mean # n*d行列
            x_std = x_mu / np.sqrt(self.moving_var + epsilon) # n*d行列
            
        # gammaでスケールし、betaでシフトさせる
        out = self.gamma * x_std + self.beta # n*d行列
        return out

    def backward(self, dout):
        """
        逆伝播計算
        dout : CNNの場合は4次元、全結合層の場合は2次元  
        """
        if dout.ndim == 4:
            """
            画像形式の場合
            """            
            N, C, H, W = dout.shape
            dout = dout.transpose(0, 2, 3, 1) # NHWCに入れ替え
            dout = dout.reshape(N*H*W, C) # (N*H*W,C)の2次元配列に変換
            dx = self.__backward(dout)
            dx = dx.reshape(N, H, W, C)# 4次元配列に変換
            dx = dx.transpose(0, 3, 1, 2) # 軸をNCHWに入れ替え
        elif dout.ndim == 2:
            """
            画像形式以外の場合
            """
            dx = self.__backward(dout)

        return dx

    def __backward(self, dout):
        """
        ここを完成させるには、計算グラフを理解する必要があり、実装にかなり時間がかかる.
        """
        
        # betaの勾配
        dbeta = np.sum(dout, axis=0)
        
        # gammaの勾配(n方向に合計)
        dgamma = np.sum(self.x_std * dout, axis=0)
        
        # Xstdの勾配
        a1 = self.gamma * dout
        
        # Xmuの勾配(1つ目)
        a2 = a1 / self.std
        
        # 標準偏差の逆数の勾配(n方向に合計)
        a3 = np.sum(a1 * self.x_mu, axis=0)

        # 標準偏差の勾配
        a4 = -(a3) / (self.std * self.std)
        
        # 分散の勾配
        a5 = 0.5 * a4 / self.std
        
        # Xmuの2乗の勾配
        a6 = a5 / self.batch_size
        
        # Xmuの勾配(2つ目)
        a7 = 2.0  * self.x_mu * a6
        
        # muの勾配
        a8 = np.sum(-(a2+a7), axis=0)

        # Xの勾配
        dx = a2 + a7 +  a8 / self.batch_size # 第3項はn方向に平均
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx
        
        
class Conv2D:
    def __init__(self, filter_num, filter_size, pad, stride):
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.pad = pad
        self.stride = stride
        
class Pool2D:
    def __init__(self, pool_size, pad, stride):
        self.pool_size = pool_size
        self.pad = pad
        self.stride = stride
        
class MultiLayerConvNet:
    def __init__(self, input_dim=(1, 28, 28), cnn_layers_list=[],
                 hidden_size_list=[100,100],
                 output_size=10, weight_init_std=0.01,
                 use_batchnorm=False, weight_decay_lambda=0,
                 use_dropout=False, dropout_ration=0.2):
        """
        input_dim : tuple, 入力の配列形状(チャンネル数、画像の高さ、画像の幅)
        conv1**_param : dict, 畳み込みの条件
        pool1**_param : dict, プーリングの条件
        hidden_size_list : list, 隠れ層のノード
        output_size : int, 出力層のノード数
        weight_init_std ： float, 重みWを初期化する際に用いる標準偏差
        use_batchNorm: Batch Normalizationを使用するかどうか
        weight_decay_lambda : Weight Decay（L2ノルム）の強さ
        use_dropout: Dropoutを使用するかどうか
        dropout_ration : Dropout割合
        """
        
        self.use_batchnorm = use_batchnorm
        self.weight_d_lmda  = weight_decay_lambda
        self.use_dropout = use_dropout
        self.dropout_ration = dropout_ration
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        
        print('----------------------------------------------------')
        print('self.use_batchnorm <-'.ljust(40,'-'), self.use_batchnorm)
        print('self.weight_d_lmda <-'.ljust(40,'-'), self.weight_d_lmda)
        print('self.use_dropout <-'.ljust(40,'-'), self.use_dropout)
        print('self.dropout_ration <-'.ljust(40,'-'), self.dropout_ration)
        print('self.hidden_size_list <-'.ljust(40,'-'), self.hidden_size_list)
        print('----------------------------------------------------')
        
        # 出力サイズ算出
        last_conv_filter_num = 0
        out_size = [input_dim[1]]
        for i, dic in enumerate(cnn_layers_list):
            key = next(iter(dic))
            layer = dic[key]
            if(layer.__class__.__name__=='Conv2D'):
                out_size.append((out_size[-1] + 2*layer.pad - layer.filter_size) // layer.stride + 1)
                last_conv_filter_num = layer.filter_num
                
            elif(layer.__class__.__name__=='Pool2D'):
                out_size.append((out_size[-1] + 2*layer.pad - layer.pool_size) // layer.stride + 1)
            
        # 総ピクセルサイズ算出
        out_pixel = last_conv_filter_num * out_size[-1] * out_size[-1] # 畳み込み層のフィルター数×最終出力サイズ^2
        print('out_size <-'.ljust(40,'-'),out_size)
        print('out_pixel <-'.ljust(40,'-'),out_pixel)
        
        # ======================================================================
        # 重みの初期化
        # ======================================================================
        self.params = {}
        std = weight_init_std
        
        # Convolution + Pooling Layer
        ch_size=[input_dim[0]]
        for i, dic in enumerate(cnn_layers_list):
            key = next(iter(dic))
            layer = dic[key]
            id = key[-3:]
            if(layer.__class__.__name__=='Conv2D'):
                self.params['W'+id] = std * np.random.randn(layer.filter_num, ch_size[-1], layer.filter_size, layer.filter_size)
                self.params['b'+id] = np.zeros(layer.filter_num)
                ch_size.append(layer.filter_num)
        
        # Fully Connected Layer
        for idx in range(1, self.hidden_layer_num+1):            
            id = '2' + str(idx) + '1'             
            if(idx==1):
                self.params['W'+id] = std *  np.random.randn(out_pixel, hidden_size_list[0])
                self.params['b'+id] = np.zeros(hidden_size_list[0])
            else:
                self.params['W'+id] = std *  np.random.randn(hidden_size_list[idx-2], hidden_size_list[idx-1])
                self.params['b'+id] = np.zeros(hidden_size_list[idx-1])
        
        # Output Layer
        self.params['W311'] = std *  np.random.randn(hidden_size_list[-1], output_size)
        self.params['b311'] = np.zeros(output_size)
        
        # ======================================================================
        # レイヤの生成
        # ======================================================================
        self.layers = OrderedDict()
        
        #---------------------------------------------------------------------
        # Convolution + Pooling Layer
        for i, dic in enumerate(cnn_layers_list):
            key = next(iter(dic))
            layer = dic[key]
            id = key[-3:]
            if(layer.__class__.__name__=='Conv2D'):
                self.layers['Conv'+id] = Convolution(self.params['W'+id], self.params['b'+id],layer.stride, layer.pad)
                if(self.use_batchnorm == True):
                    self.params['gamma'+id] = np.ones(layer.filter_num)
                    self.params['beta'+id] = np.zeros(layer.filter_num)
                    self.layers['BatchNorm'+id] = BatchNormalization(self.params['gamma'+id], self.params['beta'+id])
                self.layers['ReLU'+id] = ReLU()
                
            elif(layer.__class__.__name__=='Pool2D'):
                self.layers['Pool'+id] = MaxPooling(pool_h=layer.pool_size, pool_w=layer.pool_size, stride=layer.stride)

        #---------------------------------------------------------------------
        # Fully Connected Layer
        for idx in range(1, self.hidden_layer_num+1):            
            id = '2' + str(idx) + '1'            
            self.layers['Affine'+id] = Affine(self.params['W'+id], self.params['b'+id])
            if self.use_batchnorm:
                self.params['gamma'+id] = np.ones(hidden_size_list[idx-1])
                self.params['beta'+id] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm'+id] = BatchNormalization(self.params['gamma'+id], self.params['beta'+id])
            self.layers['ReLU'+id] = ReLU()
            if self.use_dropout:
                self.layers['Dropout'+id] = Dropout(dropout_ration)
                
        #---------------------------------------------------------------------
        # Output Layer
        self.layers['Affine311'] = Affine(self.params['W311'], self.params['b311'])
        self.last_layer = SoftmaxWithLoss()
        
        print('----------------------------------------------------')
        for key, layer in self.layers.items():
            id = key[-3:]
            if "Affine" in key or "Conv" in key:
                print((key + ' <-').ljust(40,'-'), 'W'+id+':',self.params['W'+id].shape,'b'+id+':',self.params['b'+id].shape,)
            elif "BatchNorm" in key:
                print((key + ' <-').ljust(40,'-'), 'gamma'+id+':',self.params['gamma'+id].shape,'beta'+id+':',self.params['beta'+id].shape,)
            else:
                print((key + ' <-').ljust(40,'-'))
        print('----------------------------------------------------')
        #for k in self.params.keys():
        #    print((k + ' <-').ljust(40,'-'), self.params[k].shape)
        #print('----------------------------------------------------')

    # *************************************************************************
    # predict
    # *************************************************************************
    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            #print(('├─ x = ' + str(layer.__class__.__name__) + '.forward(x) <---').ljust(80,'-'),' x :',x.shape, ' train_flg:',train_flg)
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x
    
    # *************************************************************************
    # loss
    # *************************************************************************
    def loss(self, x, t, train_flg=True):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x, train_flg)
        
        weight_decay = 0
        
        for k in self.params.keys():
            if('W' in k):
                W = self.params[k]
                weight_decay += 0.5 * self.weight_d_lmda * np.sum(W**2)
        
        return self.last_layer.forward(y, t) + weight_decay

    # *************************************************************************
    # accuracy
    # *************************************************************************
    def accuracy(self, x, t, batch_size=100, train_flg=False):
        
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        n_correct = 0.0 # 正解数
        
        for i in range(int(x.shape[0] // batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg)
            y = np.argmax(y, axis=1) # 予測        
            n_correct += np.sum(y == tt) # 正解だったら足す
        
        return n_correct / x.shape[0]
    
    # *************************************************************************
    # get_miss_read_data
    # *************************************************************************
    def get_miss_read_data(self, x, t, batch_size=100):
        
        dic_idx = {0:"a",1:"i",2:"u",3:"e",4:"o",5:"ka",6:"ki",7:"ku",8:"ke",9:"ko",10:"sa",11:"si",12:"su",13:"se",14:"so"}
        
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        ng_dic = defaultdict(int)
        ng_img_list=[]
        miss_read_lbl_list=[]
        correct_lbl_list=[]
        
        for i in range(int(x.shape[0] // batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1) # 予測
            
            for m in range(len(y)):
                if(y[m] != tt[m]):
                    correct_label = dic_idx[tt[m]]
                    ng_dic[correct_label] += 1
                    
                    ng_img_list.append(tx[m,:,:,:,])
                    miss_read_lbl_list.append(dic_idx[y[m]])
                    correct_lbl_list.append(correct_label)
                    
        return ng_dic, ng_img_list, miss_read_lbl_list, correct_lbl_list
    
    # *************************************************************************
    # gradient
    # *************************************************************************
    def gradient(self, x, t):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        x : 入力データ
        t : 教師データ
        Returns：勾配
        -------
        """
        # forward
        #print('\n〇 1.勾配の計算 ---> 1-1.損失関数を求める gradient ---> forward')
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        #print('\n〇 1.勾配の計算 ---> 1-2.誤差逆伝播法 gradient ---> backward dout：',dout.shape)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            #print(('├─ dout = ' + str(layer.__class__.__name__) + '.backward(dout) <---').ljust(80,'-'),' dout :',dout.shape)
            dout = layer.backward(dout)

        # 勾配設定
        grads = {}
        for layer in self.layers.keys():
            id = layer[-3:]
            if('Conv' in layer or 'Affine' in layer):
                grads['W'+id] = self.layers[layer].dW + self.weight_d_lmda * self.params['W'+id]
                grads['b'+id] = self.layers[layer].db
            elif('BatchNorm' in layer):
                grads['gamma'+id] = self.layers[layer].dgamma
                grads['beta'+id] = self.layers[layer].dbeta
                
        return grads
        
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W # フィルターの重み(配列形状:フィルターの枚数, チャンネル数, フィルターの高さ, フィルターの幅)
        self.b = b #フィルターのバイアス
        self.stride = stride # ストライド数
        self.pad = pad # パディング数
        
        # インスタンス変数の宣言
        self.x = None   
        self.col = None
        self.col_W = None
        self.dcol = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        順伝播計算
        x : 入力(配列形状=(データ数, チャンネル数, 高さ, 幅))
        """
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = (H + 2*self.pad - FH) // self.stride + 1 # 出力の高さ(端数は切り捨てる)
        out_w =(W + 2*self.pad - FW) // self.stride + 1# 出力の幅(端数は切り捨てる)
        
        # 畳み込み演算を効率的に行えるようにするため、入力xを行列colに変換する
        col = im2col(x, FH, FW, self.stride, self.pad)
        
        # 重みフィルターを2次元配列に変換する
        # col_Wの配列形状は、(C*FH*FW, フィルター枚数)
        col_W = self.W.reshape(FN, -1).T

        # 行列の積を計算し、バイアスを足す
        #print('np.dot(col, col_W) + self.b <---------', 'col：',col.shape, ',col_W：',col_W.shape, ',b：',self.b.shape)
        out = np.dot(col, col_W) + self.b
        
        # 画像形式に戻して、チャンネルの軸を2番目に移動させる
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        """
        逆伝播計算
        Affineレイヤと同様の考え方で、逆伝播させる
        dout : 出力層側の勾配
        return : 入力層側へ伝える勾配
        """
        FN, C, FH, FW = self.W.shape
        
        # doutのチャンネル数軸を4番目に移動させ、2次元配列に変換する
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        # バイアスbはデータ数方向に総和をとる
        self.db = np.sum(dout, axis=0)
        
        # 重みWは、入力である行列colと行列doutの積になる
        self.dW = np.dot(self.col.T, dout)
        
        # (フィルター数, チャンネル数, フィルター高さ、フィルター幅)の配列形状に戻す
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 入力側の勾配は、doutにフィルターの重みを掛けて求める
        dcol = np.dot(dout, self.col_W.T)
        
        # 勾配を4次元配列(データ数, チャンネル数, 高さ, 幅)に変換する
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad, is_backward=True)

        self.dcol = dcol # 結果を確認するために保持しておく
            
        return dx
    
    
class MaxPooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):

        self.pool_h = pool_h # プーリングを適応する領域の高さ
        self.pool_w = pool_w # プーリングを適応する領域の幅
        self.stride = stride # ストライド数
        self.pad = pad # パディング数

        # インスタンス変数の宣言
        self.x = None
        self.arg_max = None
        self.col = None
        self.dcol = None
        
            
    def forward(self, x):
        """
        順伝播計算
        x : 入力(配列形状=(データ数, チャンネル数, 高さ, 幅))
        """        
        N, C, H, W = x.shape
        
        # 出力サイズ
        out_h = (H  + 2*self.pad - self.pool_h) // self.stride + 1 # 出力の高さ(端数は切り捨てる)
        out_w = (W + 2*self.pad - self.pool_w) // self.stride + 1# 出力の幅(端数は切り捨てる)    
        
        # プーリング演算を効率的に行えるようにするため、2次元配列に変換する
        # パディングする値は、マイナスの無限大にしておく
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad, constant_values=-np.inf)
        
        # チャンネル方向のデータが横に並んでいるので、縦に並べ替える
        # 変換後のcolの配列形状は、(N*C*out_h*out_w, H*W)になる 
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # 最大値のインデックスを求める
        # この結果は、逆伝播計算時に用いる
        arg_max = np.argmax(col, axis=1)
        
        # 最大値を求める
        out = np.max(col, axis=1)
        
        # 画像形式に戻して、チャンネルの軸を2番目に移動させる
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        """
        逆伝播計算
        マックスプーリングでは、順伝播計算時に最大値となった場所だけに勾配を伝える
        順伝播計算時に最大値となった場所は、self.arg_maxに保持されている        
        dout : 出力層側の勾配
        return : 入力層側へ伝える勾配
        """        
        
        # doutのチャンネル数軸を4番目に移動させる
        dout = dout.transpose(0, 2, 3, 1)
        
        # プーリング適応領域の要素数(プーリング適応領域の高さ × プーリング適応領域の幅)
        pool_size = self.pool_h * self.pool_w
        
        # 勾配を入れる配列を初期化する
        # dcolの配列形状 : (doutの全要素数, プーリング適応領域の要素数) 
        # doutの全要素数は、dout.size で取得できる
        dcol = np.zeros((dout.size, pool_size))
        
        # 順伝播計算時に最大値となった場所に、doutを配置する
        # dout.flatten()はdoutを1次元配列に変換している
        dcol[np.arange(dcol.shape[0]), self.arg_max] = dout.flatten()
        
        # 勾配を4次元配列(データ数, チャンネル数, 高さ, 幅)に変換する
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad, is_backward=True)
        
        self.dcol = dcol # 結果を確認するために保持しておく
        
        return dx
