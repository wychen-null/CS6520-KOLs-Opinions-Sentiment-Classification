import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from tqdm import tqdm
import pytz
from datetime import date
import random
from matplotlib import pyplot as plt
import warnings
from sklearn.metrics import accuracy_score, auc, roc_curve
from scipy.optimize import minimize

PATH = "./data/tweets_epochs-8_rank-8_0.88.csv"
OUTPUT_PATH = "./output/llama2_tweets_0.88_eval.csv"
'''
此函数计算指定时间窗口内给定股票代码的趋势和价格变动水平
symbol：要分析的股票代码。
date：分析的参考日期。
window：计算价格变动时从参考日期向前展望的天数。
threshold_trend：确定显著趋势变动的阈值。
threshold_price：确定重大价格变动的阈值。
'''
def price_moving_level(symbol, date, window, threshold_trend, threshold_price):
    # 获取一只股票在date过去一天和未来10天的股价
    stock = yf.Ticker(symbol)
    start_date = date - datetime.timedelta(days=1)
    end_date = date + datetime.timedelta(days=10)

    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    df_stock = stock.history(start=start_date, end=end_date).reset_index()

    # 股价变化趋势 股价变化率
    rtn_trend, rtn_price = -1, -1

    # 如果获取的股票数据为空，直接返回
    if df_stock.empty:
        return rtn_trend, rtn_price
    else:
        df_stock["Date"] = pd.to_datetime(df_stock["Date"])
        df_stock['date'] = df_stock['Date'].dt.date  # 去掉时间，只保留日期

        # date = pd.to_datetime(date)
        # tmp = pd.Timestamp(date)
        # date当日 股票收盘价
        p0 = df_stock[df_stock['date'] == date]['Close'].values
        # date当日 股价最高值
        h0 = df_stock[df_stock['date'] == date]['High'].values
        # date当日 股价最低值
        l0 = df_stock[df_stock['date'] == date]['Low'].values

        if p0.size == 0:
            return rtn_trend, rtn_price
        else:
            # 计算 date + window ，获取下一个目标股票的索引
            next_idx = df_stock[df_stock['date'] == date].index + window
            p1 = df_stock.loc[next_idx]['Close'].values  # 股票收盘价
            h1 = df_stock.loc[next_idx]['High'].values  # 股价最高值
            l1 = df_stock.loc[next_idx]['Low'].values  # 股价最低值

            # 通过计算相邻两天之间的最高价和最低价的差异，再除以最低价/最高价，可以得到当天的True Range
            TD_h1l0 = (h1[0] - l0[0]) / l0[0]  # True Range的升幅
            TD_l1h0 = (l1[0] - h0[0]) / h0[0]  # True Range的降幅

            # 升幅和降幅那个值绝对值更大，则代表了这段时间的波动方向，上升还是下降
            if abs(TD_h1l0) > abs(TD_l1h0):
                l = TD_h1l0
            else:
                l = TD_l1h0

            # 价格变化率
            # 当前收盘价减去前一天的收盘价，然后除以window前的收盘价，得到了价格变化的百分比
            p = (p1[0] - p0[0]) / p0[0]

            # threshold_trend作比较
            # 大幅上涨为 1 ， 大幅下跌为0， 小幅波动为2
            if l >= threshold_trend:
                rtn_trend = 1
            elif l <= -threshold_trend:
                rtn_trend = 0
            else:
                rtn_trend = 2

            # threshold_price作比较
            # 大幅上涨为 1 ， 大幅下跌为0， 小幅波动为2
            if p >= threshold_price:
                rtn_price = 1
            elif p <= -threshold_price:
                rtn_price = 0
            else:
                rtn_price = 2

            return rtn_trend, rtn_price


'''
根据特定作者的*推文*评估情绪及其与股票价格走势的相关性
author：要分析的推文的作者。
window：价格走势分析的时间窗口。
threshold_trend：趋势评估的阈值。
threshold_price：价格评估的门槛。
'''


def evaluate_tweet(author, window, threshold_trend, threshold_price):
    # 按指定作者和日期过滤推文
    df_all = tweets_df[tweets_df['source'] == author]
    dates = df_all['date'].value_counts()
    # print("df_all", df_all)

    sen_arr = []  # 情绪平均值
    trend_arr = []  # 趋势水平
    price_arr = []  # 价格水平

    # print(dates)
    # 遍历每一天
    for date, cnt_date in tqdm(dates.items()):
        # 获取某一天的所有股票的value_counts
        df_today = df_all[df_all['date'] == date]
        symbols = df_today['symbols'].value_counts()

        # ruoxin的论文中 0 1 2 代表着 负 中 正
        # tweets数据中 0 1 2 代表 负 正 中， 本次实验均以这个mapping为准
        for sym, cnt_sym in symbols.items():
            df_sen = df_today[df_today['symbols'] == sym]
            sen_sum = df_sen['pred_sen'].sum() # 关于这支股票情绪分类的和

            # 获取关于这支股票的在这个window区间内的趋势变化和价格变化水平
            trend_level, price_level = price_moving_level(sym, date, window, threshold_trend, threshold_price)
            # print("price_level", price_level)

            if price_level >= 0:
                sen_avg = sen_sum / cnt_sym  # 平均情感分类
                sen_arr.append(round(sen_avg))  # 四舍五入
                trend_arr.append(trend_level)  # 变化趋势
                price_arr.append(price_level)  # 价格趋势

    # 分别对比 涨跌变化和情感分类预测值 与 股价变化和情感分类预测值 准确度
    acc_trend = accuracy_score(trend_arr, sen_arr)
    acc_price = accuracy_score(price_arr, sen_arr)
    return acc_trend, acc_price, trend_arr, price_arr, sen_arr


def con_prob(level, price_arr, sen_arr):
    cnt_num = 0
    cnt_den = 0

    for i in range(len(price_arr)):
        if price_arr[i] == level:
            #             if sen_arr[i] <= level and sen_arr[i] > level-1:
            if sen_arr[i] == level:
                cnt_num += 1
    for s in sen_arr:
        #         if s <= level and s > level-1:
        if s == level:
            cnt_den += 1
    if cnt_den != 0:
        return cnt_num / cnt_den
    else:
        return 0

def plot_roc(validY, validProb):
    # ROC AUC
    fpr, tpr, thresholds = roc_curve(validY, validProb, pos_label=1)
    roc_auc = auc(fpr, tpr)

    print("ROC_AUC :", roc_auc)
    return roc_auc

    # fig = plt.figure()
    # plt.plot(fpr, tpr, lw=2, color='#7777cb')
    # plt.title(text + ' AUC={:.4f}'.format(roc_auc))
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    #
    # plt.grid(True)
    # plt.savefig(f'./pictures/{text}.png')
    # plt.show()
    # time.sleep(0.5)


def plot(sen_arr, trend_arr, price_arr, aut, i):
    # 检查输入数组是否为空
    if not sen_arr or not price_arr:
        # print("+++++++++++++++++++++++")
        # print(sen_arr)
        # print("=======================")
        # print(trend_arr)
        # print(">>>>>>>>>>>>>>>>>>>>>>>")
        # print(price_arr)
        print("Error: One or more input arrays are empty.")
        return

    # 确保所有数组长度相同
    length = len(sen_arr)
    if len(trend_arr) != length or len(price_arr) != length:
        print("Error: Input arrays have different lengths.")
        return

    fig, ax = plt.subplots()

    plt.xlabel('tweets')
    plt.ylabel('price level; sentiment level')

    yticks = range(-1, 4)
    ax.set_yticks(yticks)

    length = len(sen_arr)
    ax.set_ylim([-1, 4])
    ax.set_xlim([0, length])

    x = np.arange(0, length)
    plt.plot(x, sen_arr, "x-", label="sentiment")
    plt.plot(x, trend_arr,"o-",label="trend level")
    plt.plot(x, price_arr, "+-", label="price level")

    plt.grid(True)
    plt.title(f"Plot {aut} - {i}")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

    plt.show()


if __name__ == '__main__':

    warnings.simplefilter(action='ignore', category=FutureWarning)
    tz = pytz.timezone("America/New_York")

    tweets_df = pd.read_csv(PATH, on_bad_lines = 'skip')
    tweets_df["timestamp"] = pd.to_datetime(tweets_df["timestamp"])
    tweets_df ['date'] = tweets_df['timestamp'].dt.date

    new_rows = []
    idx_del = []

    for idx, row in tweets_df.iterrows():
        if '-' in row['symbols']:
            symbols =  row['symbols'].split('-')
            idx_del.append(idx)

            for s in symbols:
                nr = row.copy()
                nr['symbols'] = s

    tweets_df = tweets_df.drop(idx_del)
    new_rows_df = pd.DataFrame(new_rows, columns=tweets_df.columns)
    tweets_df = pd.concat([tweets_df, new_rows_df], ignore_index=True)

    tweet_authors = tweets_df['source'].value_counts()
    print(tweet_authors)

    columns = ['author', 'time_window', 'acc_trend', 'roc_trend', 'acc_price', 'roc_price', 'p_0_trend', 'p_1_trend', 'p_2_trend',
               'p_0_price', 'p_1_price', 'p_2_price']
    tweet_eval_df = pd.DataFrame(columns=columns)
    for i in range(1,6):
        for aut, cnt_aut in tweet_authors.items():
            if cnt_aut >= 10:
                acc_trend, acc_price ,trend_arr, price_arr, sen_arr = evaluate_tweet(aut,i,0.02,0.01)
                # 关于预测值与真实值之间的roc曲线，并返回值
                roc_trend = plot_roc(sen_arr, trend_arr)
                roc_price = plot_roc(sen_arr, price_arr)

                new_row = pd.DataFrame({'author': [aut],
                                        'time_window': [i],
                                        'acc_trend': [acc_trend],
                                        'roc_trend': [roc_trend],
                                        'acc_price': [acc_price],
                                        'roc_price': [roc_price],
                                        'p_0_trend': [con_prob(0, trend_arr, sen_arr)],
                                        'p_1_trend': [con_prob(1, trend_arr, sen_arr)],
                                        'p_2_trend': [con_prob(2, trend_arr, sen_arr)],
                                        'p_0_price': [con_prob(0, price_arr, sen_arr)],
                                        'p_1_price': [con_prob(1, price_arr, sen_arr)],
                                        'p_2_price': [con_prob(2, price_arr, sen_arr)]
                })

                tweet_eval_df = pd.concat([tweet_eval_df, new_row], ignore_index=True)
                # 这里是做实际股价与预测股价的对比图，如果需要观察，取消注释
                # plot(sen_arr, trend_arr, price_arr, aut, i)

    tweet_eval_df.to_csv(OUTPUT_PATH,index=False)

    tweet_acc={}
    tweet_acc = tweet_acc.fromkeys(tweet_authors[0:10].keys(),[0,0])
    # print(tweet_acc)
    for aut, cnt_aut in tweet_authors[0:10].items():
        trend_acc=tweet_eval_df[tweet_eval_df['author']==aut]['acc_trend'].mean()
        price_acc = tweet_eval_df[tweet_eval_df['author']==aut]['acc_price'].mean()
        tweet_acc[aut] = [trend_acc,price_acc]

    tweet_acc = sorted(tweet_acc.items(), key=lambda x:x[1][0],reverse=True)
    print(tweet_acc)