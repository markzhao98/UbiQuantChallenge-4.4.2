'''
厦一代表队
'''

# 数据格式
# 0:'tick', 1:'stock', 2:'open', 3:'high', 
# 4:'low', 5:'close', 6:'volume', 7:'tvr', 
# 8:'bid1_price', 9:'bid1_volume', 10:'bid2_price', 11:'bid2_volume', 
# 12:'bid3_price', 13:'bid3_volume', 14:'bid4_price', 15:'bid4_volume', 
# 16:'bid5_price', 17:'bid5_volume', 18:'bid6_price', 19:'bid6_volume', 
# 20:'bid7_price', 21:'bid7_volume', 22:'bid8_price', 23:'bid8_volume', 
# 24:'bid9_price', 25:'bid9_volume', 26:'bid10_price', 27:'bid10_volume', 
# 28:'ask1_price', 29:'ask1_volume', 30:'ask2_price', 31:'ask2_volume', 
# 32:'ask3_price', 33:'ask3_volume', 34:'ask4_price', 35:'ask4_volume', 
# 36:'ask5_price', 37:'ask5_volume', 38:'ask6_price', 39:'ask6_volume', 
# 40:'ask7_price', 41:'ask7_volume', 42:'ask8_price', 43:'ask8_volume', 
# 44:'ask9_price', 45:'ask9_volume', 46:'ask10_price', 47:'ask10_volume'

import grpc
import contest_pb2
import contest_pb2_grpc
import question_pb2
import question_pb2_grpc

import time
import numpy as np

import saver
import training_th
from supp import *

class Client:
    
    # --- class attribute ---
    ID = 121 # your ID
    PIN = 's5eouCB3X1' # your PIN
    CHANNEL_LOGIN_SUBMIT = grpc.insecure_channel('47.100.97.93:40723')
    CHANNEL_GETDATA = grpc.insecure_channel('47.100.97.93:40722')
    
    stub_contest = contest_pb2_grpc.ContestStub(CHANNEL_LOGIN_SUBMIT)
    stub_question = question_pb2_grpc.QuestionStub(CHANNEL_GETDATA)
    
    def __init__(self):

        # login

        self.session_key = None # 用于提交position
        self.login_success = None # 是否成功login

        # get data

        self.sequence = None # 数据index
        self.has_next_question = None # 后续是否有数据
        self.capital = None # 总资产
        self.dailystk = None # 数据！共500支股票
        self.positons = None # 当前持仓

        # output

        self.is_initialized = True
        self.hf_portion = 2/3  # hf交易占全部交易的权重
        self.leverage = 1.99  # 杠杆率 (maximum = 2)

        '''hf部分(预测10ticks)'''
        self.hf_loaded_model = saver.hf_loaded_model # 初始hf模型
        self.hf_holding = 10  # hf持有周期（单位：tick）
        self.hf_num_newpos = 15  # hf每轮分别新做多和做空股数
        self.hf_weights_d = np.array([1.35, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05, 1, 
                                      0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65])
        self.hf_weights_a = np.flip(self.hf_weights_d)
        self.hf_track_newpos = np.zeros([self.hf_holding, 500])  # 记录hf新开仓

        '''hhf部分(预测5ticks)'''
        self.hhf_loaded_model = saver.hhf_loaded_model
        self.hhf_holding = 5  # hhf持有周期（单位：tick）
        self.hhf_num_newpos = 15  # hhf每轮分别新做多和做空股数
        self.hhf_weights_d = np.array([1.35, 1.3, 1.25, 1.2, 1.15, 1.1, 1.05, 1, 
                                      0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65])
        self.hhf_weights_a = np.flip(self.hhf_weights_d)
        self.hhf_track_newpos = np.zeros([self.hhf_holding, 500])  # 记录hhf新开仓
        
        # submit
        self.accepted = None
        
    def login(self):
        response_login = self.stub_contest.login(contest_pb2.LoginRequest(
            user_id=self.ID,
            user_pin=self.PIN
            ))
        self.session_key = response_login.session_key # 用于提交position
        self.login_success = response_login.success # 是否成功login
        
    def getdata(self):
        response_question = self.stub_question.get_question(question_pb2.QuestionRequest(
            user_id=self.ID,
            user_pin=self.PIN,
            session_key=self.session_key # 首次询问数据 0 # 注意用0有收不到数据风险
        ))
        return response_question 
        
    def output(self):

        K = self.capital * self.leverage  # 可操作资金

        """
        计算因子
        """
        
        saver.rolling_df = np.append(np.expand_dims(self.dailystk, axis=0).transpose(1, 2, 0), \
            saver.rolling_df, axis=2)[:, :, :-1]

        self.alphas = np.zeros((500, 23))

        # SOIR
        for i in range(10):
            self.alphas[:, i] = np.nan_to_num((saver.rolling_df[:, 9 + 2 * i, 0] - saver.rolling_df[:, 29 + 2 * i, 0]) 
                / (saver.rolling_df[:, 9 + 2 * i, 0] + saver.rolling_df[:, 29 + 2 * i, 0]))

        # MPC
        mpc_mid = (saver.rolling_df[:, 8, :11] + saver.rolling_df[:, 28, :11]) / 2
        for i in range(10):
            self.alphas[:, 10 + i] = np.nan_to_num(mpc_mid[:, 0] / mpc_mid[:, i + 1] - 1)
        
        # Momentum [5, 10, 20]
        self.alphas[:, 20] = np.nan_to_num(saver.rolling_df[:, 5, 0] / saver.rolling_df[:, 5, 5] - 1)
        self.alphas[:, 21] = np.nan_to_num(saver.rolling_df[:, 5, 0] / saver.rolling_df[:, 5, 10] - 1)
        self.alphas[:, 22] = np.nan_to_num(saver.rolling_df[:, 5, 0] / saver.rolling_df[:, 5, 20] - 1)

        """
        hf预测与选股
        """
        hf_pred = self.hf_loaded_model.predict(self.alphas)
        
        hf_longstock = hf_pred.argsort()[-self.hf_num_newpos:] # 做多哪些股票
        hf_shortstock = hf_pred.argsort()[:self.hf_num_newpos] # 做空哪些股票

        hf_newpos = np.zeros(500)

        hf_afford = K*self.hf_portion/(2*self.hf_num_newpos) \
            /self.hf_holding/self.dailystk[:,5]

        hf_newpos[hf_longstock] = hf_afford[hf_longstock] * self.hf_weights_a # 非等权做多
        hf_newpos[hf_shortstock] = -hf_afford[hf_shortstock] * self.hf_weights_d # 非等权做空

        self.hf_track_newpos = np.append(self.hf_track_newpos, np.array([hf_newpos]), axis = 0)[1:]

        hf_obj_pos = self.hf_track_newpos.sum(axis = 0)  # hf目标仓位

        """
        hhf预测与选股
        """
        hhf_pred = self.hhf_loaded_model.predict(self.alphas)
        
        hhf_longstock = hhf_pred.argsort()[-self.hhf_num_newpos:] # 做多哪些股票
        hhf_shortstock = hhf_pred.argsort()[:self.hhf_num_newpos] # 做空哪些股票

        hhf_newpos = np.zeros(500)

        hhf_afford = K*(1-self.hf_portion)/(2*self.hhf_num_newpos) \
            /self.hhf_holding/self.dailystk[:,5]

        hhf_newpos[hhf_longstock] = hhf_afford[hhf_longstock] * self.hhf_weights_a # 非等权做多
        hhf_newpos[hhf_shortstock] = -hhf_afford[hhf_shortstock] * self.hhf_weights_d # 非等权做空

        self.hhf_track_newpos = np.append(self.hhf_track_newpos, np.array([hhf_newpos]), axis = 0)[1:]

        hhf_obj_pos = self.hhf_track_newpos.sum(axis = 0)  # hhf目标仓位

        """
        仓位变化与报单价格
        """

        self.submit_pos = hf_obj_pos + hhf_obj_pos - self.positions  # 提交仓位变化

        # 做多报当前tick卖盘ask5，做空报当前tick买盘bid5（几乎保证成交）
        bid5 = self.dailystk[:,16]
        ask5 = self.dailystk[:,36]
        self.submit_price = ask5 * (self.submit_pos > 0) + bid5 * (self.submit_pos < 0)  # 提交报单价格

        return

    def submit(self):
        response_ansr = self.stub_contest.submit_answer_make(contest_pb2.AnswerMakeRequest(
            user_id=self.ID,
            user_pin=self.PIN,
            session_key=self.session_key, # 使用login时系统分配的key来提交
            sequence=self.sequence, # 使用getdata时获得的sequence
            bidasks=self.submit_pos, # 仓位变动
            prices=self.submit_price # 报单价格
        )) 
        self.accepted = response_ansr.accepted # 是否打通提交通道
        if not self.accepted:
            print(response_ansr.reason) # 未成功原因
        
    def run(self):
        try:    
            self.login()
            print(f'Log in result: {self.login_success} ...')
            response_seq = self.getdata()
            print(response_seq)
            for response in response_seq:
                if response.sequence == -2: 
                    print(f'!!{response.sequence}!!')
                    print('jobfetcher: error! ID and pin mismatch')
                    break
                if response.sequence == -3:
                    print('!!!error code -3, session key mismatch')
                    break
                print(f'Sequence now: {response.sequence} ...')
                if len(response.positions)==0:
                    self.positions = np.zeros(500)
                else:
                    self.positions = np.array(response.positions)
                self.sequence = response.sequence
                self.capital = response.capital
                self.dailystk = np.array([dstk.values[:] for dstk in response.dailystk])
                self.cur_close = self.dailystk[:,5]
                self.output()
                self.submit()
                print(f'Submit result: {self.accepted} ...')

                """
                updating training data
                """
                tick_stock_close_alphas = np.append(self.dailystk[:, [0, 1, 5]], self.alphas, axis=1)  # (500, 26)
                saver.training_df = np.append(saver.training_df, tick_stock_close_alphas, axis=0)[500:, :]
                # print(saver.training_df)

                """
                continual training
                """
                if saver.locked:
                    time.sleep(0.1)
                if not saver.training_flag:
                    self.hhf_loaded_model = saver.hhf_loaded_model
                    self.hf_loaded_model = saver.hf_loaded_model
                    print("========== new models loaded!!==========")
                    saver.training_flag = True
                    thread = training_th.TrainingThread(name='TrainingThread')      # initialize the thread
                    thread.start()
                    
        except KeyboardInterrupt:
            exit(0)

if __name__ == "__main__":    

    c = Client()
    c.run()


