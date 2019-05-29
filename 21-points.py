# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:34:30 2019

@author: xing
"""

from random import shuffle
from queue import Queue
from tqdm import tqdm
from utils import str_key, set_dict, get_dict

import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class Gamer():
    '''游戏者
    '''

    def __init__(self, name="", A=None, display=False):
        self.name = name
        self.cards = []  # 手中的牌
        self.display = display  # 是否显示对局文字信息
        self.policy = None  # 策略
        self.learning_method = None  # 学习方法
        self.A = A  # 行为空间

    def __str__(self):
        return self.name

    def _value_of(self, card):
        '''根据牌的字符判断牌的数值大小，A被输出为1，JQK均为10
        Arg:
            card: 牌面信息str
        Return:
            牌的大小数值int, A 返回 1
        '''
        try:
            v = int(card)
        except:
            if card == 'A':
                v = 1
            elif card in ['J', 'Q', 'K']:
                v = 10
            else:
                v = 0
        finally:
            return v

    def get_points(self):
        '''统计一手牌分值，如果使用了A的1点，同时返回True
        Arg:
            cards 庄家或玩家手中的牌 list['A', '10', '3']
        Return
            tuple(牌的总点数， 是否适用了A-11)
        '''
        num_of_useable_ace = 0  # 默认没有拿到Ace
        total_point = 0
        cards = self.cards
        if cards is None:
            return 0, False
        for card in cards:
            v = self._value_of(card)
            if v == 1:
                num_of_useable_ace += 1
                v = 11
            total_point += v
        while total_point > 21 and num_of_useable_ace > 0:
            total_point -= 10
            num_of_useable_ace -= 1

        return total_point, bool(num_of_useable_ace)

    def receive(self, cards=[]):  # 玩家获得一张或多张牌
        cards = list(cards)
        for card in cards:
            self.cards.append(card)

    def discharge_cards(self):  # 玩家把手中的牌清空
        self.cards.clear()

    def cards_info(self):  # 玩家手中牌的信息
        '''显示牌面的信息'''
        self._info("{}{}现在的牌:{}".format(self.role, self, self.cards))

    def _info(self, msg):
        if self.display:
            print(msg, end="")


class Dealer(Gamer):
    '''庄家'''

    def __init__(self, name="", A=None, display=False):
        super(Dealer, self).__init__(name, A, display)
        self.role = "庄家"
        self.policy = self.dealer_policy

    def first_card_value(self):  # 显示第一张明牌
        if self.cards is None or len(self.cards) == 0:
            return 0
        return self._value_of(self.cards[0])

    def dealer_policy(self, Dealer=None):
        action = ""
        dealer_points, _ = self.get_points()
        if dealer_points >= 17:
            action = self.A[1]  # "停止叫牌“
        else:
            action = self.A[0]  # "继续叫牌"
        return action


class Player(Gamer):
    '''玩家'''

    def __init__(self, name="", A=None, display=False):
        super(Player, self).__init__(name, A, display)
        self.policy = self.naive_policy
        self.role = "玩家"

    def get_state(self, dealer):
        dealer_first_card_value = dealer.first_card_value()
        player_points, useable_ace = self.get_points()
        return dealer_first_card_value, player_points, useable_ace

    def get_state_name(self, dealer):
        return str_key(self.get_state(dealer))

    def naive_policy(self, dealer=None):
        player_points, _ = self.get_points()
        # 玩家只要点数小于20 点就会继续叫牌
        if player_points < 20:
            action = self.A[0]
        else:
            action = self.A[1]
        return action


class Arena():
    '''负责游戏管理'''

    def __init__(self, display=None, A=None):
        self.cards = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'] * 4
        self.card_q = Queue(maxsize=52)  # 洗好的牌
        self.cards_in_pool = []  # 已经用过的公开的牌
        self.display = display
        self.episodes = []  # 产生的对局的信息列表
        self.load_cards(self.cards)  # 把初始状态的52张牌装入发牌器
        self.A = A  # 获得行为空间

    def load_cards(self, cards):
        '''把手机的牌洗一洗，重新装到发牌机器中
        Args:
            cards 需要装入发牌器的多张牌list
        Return:
            None
        '''
        shuffle(cards)  # 洗牌
        for card in cards:
            self.card_q.put(card)
        cards.clear()
        return

    def reward_of(self, dealer, player):
        '''判断玩家奖励值，附带玩家、庄家的牌点信息'''
        dealer_points, _ = dealer.get_points()
        player_points, useable_ace = player.get_points()
        if player_points > 21:
            reward = -1
        else:
            if player_points > dealer_points or dealer_points > 21:
                reward = 1
            elif player_points == dealer_points:
                reward = 0
            else:
                reward = -1
        return reward, player_points, dealer_points, useable_ace

    def serve_card_to(self, player, n=1):
        '''给庄家或玩家发牌， 如果不够则将公开牌池的牌洗一洗重新发牌
        Args:
            player 一个庄家或玩家
            n 一次连续发牌的数量
        Return:
            None
        '''
        cards = []
        for _ in range(n):
            if self.card_q.empty():
                self._info("\n发牌器没牌了,整理废牌,重新洗牌;")
                shuffle(self.cards_in_pool)
                self._info("一共整理了{}张已用牌，重新放入发牌器\n".format(len(self.cards_in_pool)))
                assert (len(self.cards_in_pool) > 20)
                self.load_cards(self.cards_in_pool)
            cards.append(self.card_q.get())  # 从发牌器发出一张牌
        self._info("发了{}张牌({})给{}{};".format(n, cards, player.role, player))
        player.receive(cards)
        player.cards_info()

    def _info(self, message):
        if self.display:
            print(message, end="")

    def recycle_cards(self, *players):
        '''回收玩家手中的牌到公开使用过的牌池中'''
        if len(players) == 0:
            return
        for player in players:
            for card in player.cards:
                self.cards_in_pool.append(card)
            player.discharge_cards()  # 玩家手中不再留这些牌

    def play_game(self, dealer, player):
        '''玩一局21点，生成一个状体序列以及最终奖励
        Args:
            dealer / player 庄家和玩家
        Returns:
            tuple: episode, reward
        '''
        self._info("===========开始新一局==========\n")
        self.serve_card_to(player, n=2)
        self.serve_card_to(dealer, n=2)
        episode = []  # 记录一个对局信息
        if player.policy is None:
            self._info("玩家需要一个策略")
            return
        if dealer.policy is None:
            self._info("庄家需要个策略")
            return
        while True:
            action = player.policy(dealer)
            # 玩家的策略产生一个行为
            self._info("{}{}选择:{};".format(player.role, player, action))
            episode.append((player.get_state_name(dealer), action))  # 记录 一个 （s, a)
            if action == self.A[0]:  # 继续叫牌
                self.serve_card_to(player)
            else:  # 停止叫牌
                break
        reward, player_points, dealer_points, useable_ace = self.reward_of(dealer, player)

        if player_points > 21:
            self._info("玩家爆点了{}输了,得分:{}\n".format(player_points, reward))
            self.recycle_cards(player, dealer)
            self.episodes.append((episode, reward))
            self._info("===========本局结束==========\n")
            return episode, reward

        self._info("\n")
        while True:
            action = dealer.policy()  # 庄家从其策略中获取一个行为
            self._info("{}{}选择:{};".format(dealer.role, dealer, action))
            # 状态只记录庄家第一章牌信息， 此时玩家不再叫牌， (s,a)不必重复记录
            if action == self.A[0]:
                self.serve_card_to(dealer)
            else:
                break
        # 双方均停止叫牌了

        self._info("\n双方均停止叫牌;\n")
        reward, player_points, dealer_points, useable_ace = self.reward_of(dealer, player)
        player.cards_info()
        dealer.cards_info()
        if reward == +1:
            self._info("玩家赢了!")
        elif reward == -1:
            self._info("玩家输了!")
        else:
            self._info("双方和局!")
        self._info("玩家{}点,庄家{}点\n".format(player_points, dealer_points))
        self._info("========= 本局结束==========\n")
        self.recycle_cards(player, dealer)  # 回收玩家和庄家手中的牌至公开牌池
        self.episodes.append((episode, reward))  # 将刚才产生的完整对局添加值状态序列列表， 蒙特卡罗控制不需要
        return episode, reward

    def play_games(self, dealer, player, num=2, show_statistic=True):
        '''一次性玩多局游戏'''
        results = [0, 0, 0]  # 玩家负、和、胜的局数
        self.episodes.clear()
        for i in tqdm(range(num)):
            episode, reward = self.play_game(dealer, player)
            results[1 + reward] += 1
            if player.learning_method is not None:
                player.learning_method(episode, reward)
        if show_statistic:
            print("共玩了{} 局， 玩家赢{} 局， 和{} 局， 输{} 局， 胜率： {:.2f},不输率:{:.2f}" \
                  .format(num, results[2], results[1], results[0], results[2] / num, (results[2] + results[1]) / num))
        return

    def _info(self, message):
        if self.display:
            print(message, end="")


################################################################
A = ["继续叫牌", "停止叫牌"]
display = False
player = Player(A=A, display=display)
dealer = Dealer(A=A, display=display)

arena = Arena(A=A, display=display)
arena.play_games(dealer, player, num=200000)


# print (arena.episodes)
# 统计各状态的价值，衰减因子为1，中间状态的即时奖励为0，递增式蒙特卡罗评估
def policy_evaluate(episodes, V, Ns):
    for episode, r in episodes:
        for s, a in episode:
            ns = get_dict(Ns, s)
            v = get_dict(V, s)
            set_dict(Ns, ns + 1, s)
            set_dict(V, v + (r - v) / (ns + 1), s)


V = {}  # 状态价值字典
Ns = {}  # 状态被访问的次数节点
policy_evaluate(arena.episodes, V, Ns)  # 学习V值


def draw_value(value_dict, useable_ace=True, is_q_dict=False, A=None):
    fig = plt.figure()  # 定义figure
    ax = Axes3D(fig)  # 将figure变为3d
    # 定义x, y
    x = np.arange(1, 11, 1)  # 庄家第一张牌
    y = np.arange(12, 22, 1)  # 玩家总分数
    # 生成网格数据
    X, Y = np.meshgrid(x, y)
    # 从V字典中检索Z轴的高度
    row, col = X.shape
    Z = np.zeros((row, col))
    if is_q_dict:
        n = len(A)
    for i in range(row):
        for j in range(col):
            state_name = str(X[i, j]) + "_" + str(Y[i, j]) + "_" + str(useable_ace)
            if not is_q_dict:
                Z[i, j] = get_dict(value_dict, state_name)
            else:
                assert (A is not None)
                for a in A:
                    new_state_name = state_name + "_" + str(a)
                    q = get_dict(value_dict, new_state_name)
                    if q >= Z[i, j]:
                        Z[i, j] = q
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color="lightgray")
    plt.show()


draw_value(V, useable_ace=True, A=A)

draw_value(V, useable_ace=False, A=A)
