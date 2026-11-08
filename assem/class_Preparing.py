#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np


class Preparing:
    """
    Класс для подготовки и фильтрации клиентов.
    """
    
    def __init__(self, df):
        self.df = df                                        # оригинал, если понадобится
        self.df_copy = df.copy()                            # создали копию, с которой будем работать 
        
    def rename_columns(self):
        self.df_copy.rename(
            columns={"default.payment.next.month": "default", 'PAY_0':'PAY_1'}, 
            inplace = True                           
        )

    def warning(self):
        print('Рабочая версия')
    
    def select_columns(self):
        """
        Выбираем колонки с оплатой, по которым будет производиться фильтрация.
        """
        self.rename_columns()
        self.pay_columns = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'] # создаем список столбцов с платежной дисциплиной
        self.X_pay = self.df_copy.loc[:, self.pay_columns].values # создаем матрицу признаков
        self.y = self.df_copy.default.values                      # вектор целевой переменной
        
    def group_clients(self):
        """
        'removed_0': False по умолчанию (не аномальный). Клиенты со своевременными платежами PAY_<=0, но с дефолтом (y == 1). 
        Присваивается значение True, если клиент попадает под это условие. Аномальная группа.
        self.index_no_d: клиенты, которые остались после фильтрации.
        """
        self.select_columns()
        self.df_copy['removed_0'] = False                           # клиент не помечен как аномальный
        self.index = []                                             # список для аном. клиентов, которых удалим
        self.index_no_d = []                                        # список для ост. клиентов, которых оставляем
        for i,x in enumerate(self.X_pay):                            
            if (len(np.where(x<=0)[0]) == 6 and self.y[i]==1):      # все шесть месяцев PAY_<=0 => итог все равно дефолт (метка 1)
                self.index.append(i)                                # записываем индекс клиента в список "удаленных"
                self.df_copy.loc[i, ['removed_0']] = True           # все шесть месяцев PAY_<=0 и дефолт, то помечаем True      
            else:
                self.index_no_d.append(i)                           # все остальные в список не удаленных клиентов    

        self.df_new, self.X_pay_new, self.y_new = (                 
            self.df_copy.loc[self.index_no_d,:],                    # новый датафрейм без аномалий (оставшиеся клиенты и все столбцы)
            self.X_pay[self.index_no_d,:],                          # матрица признаков без аномалий
            self.y[self.index_no_d] )                               # новый вектор меток без аномалий
        self.X_all_new = self.df_copy.loc[self.index_no_d, ['LIMIT_BAL', 'PAY_1','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
           'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
           'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].values # полная матрица признаков без аномалий
        
        return self.df_new, self.X_pay_new, self.y_new              # возвращаем эти объекты

