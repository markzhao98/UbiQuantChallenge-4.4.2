"""
@Alpha001
6/22/2021 
Edited by Ubiquant Game
7/13/2021
"""

from typing import Sequence
import grpc
# import contest_pb2
# import contest_pb2_grpc
# import question_pb2
# import question_pb2_grpc
from contest.protos._pyprotos import contest_pb2,contest_pb2_grpc,question_pb2,question_pb2_grpc


import numpy as np
import time

class Client:
    
    # --- class attribute ---
    ID = 666 # your ID
    PIN = '666666' # your PIN
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
        
        # alpha001
        self.is_initialized = False
        
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
        
    def alpha001_ret1(self):
        if self.is_initialized == False:
            pos = np.random.randint(low=-2, high=3,size=(500)).astype(int) # 随机仓位
            pos = 0.1 *pos / np.sum(abs(pos)) # 10%持仓
            pos = [int(np.round(x)) for x in self.capital * pos / self.cur_close]
            self.submit_pos = pos - self.positions # 持仓变动
            self.submit_price = np.array([1e4 if x>0 else -1 for x in self.submit_pos])  # 最高市价单成交
            return
        else: # 在这里编写你的策略 ...
            pass
        
    def submit(self):
        response_ansr = self.stub_contest.submit_answer_make(contest_pb2.AnswerMakeRequest(
            user_id=self.ID,
            user_pin=self.PIN,
            session_key=self.session_key, # 使用login时系统分配的key来提交
            sequence=self.sequence, # 使用getdata时获得的sequence
            bidasks=self.submit_pos, # 使用alpha001_ret1中计算的pos作为变动仓位
            prices = self.submit_price # 使用alpha001_ret1中计算的prices
        )) 
        self.accepted = response_ansr.accepted # 是否打通提交通道
        if not self.accepted:
            print(response_ansr.reason) # 未成功原因
        
    def run(self):
        try:
            self.login()
            print(f'Log in result: {self.login_success} ...')
            response_seq = self.getdata()
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
                self.alpha001_ret1()
                self.submit()
                print(f'Submit result: {self.accepted} ...')
                
        except KeyboardInterrupt:
            return      

if __name__ == "__main__":
    c = Client()
    c.run()