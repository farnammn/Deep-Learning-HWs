#Python 2.7.12

import numpy as np
import gradient
import matplotlib.pyplot as plt
###############################
layer_number = 20
###############################
x=np.matrix(np.random.normal(size=(3,1)))
y=np.matrix([[0.0],[1.0],[0.0]])

dw_mean_map = {}
dw_mean_map["sigmoid"] = []
dw_mean_map["relu"] = []
for layer_number in range(1,21):
    for act in ("sigmoid","relu"):
        s=0.0
        for j in range(100):

            config=[]
            w_list=[]
            b_list=[]

            for i in range(layer_number):
                config.append({"num":3,"act_name":act})
                w_list.append(np.matrix(np.random.normal(size=(3,3))).astype("double"))
                b_list.append(np.matrix(np.random.normal(size=(3,1))).astype("double"))

            y,dw,db=gradient.compute_gradient(config,w_list,b_list,"softmax_ce",x,y)
            s+=abs(dw[i - 1]).mean()
        dw_mean_map[act].append(s)
#         print "[Activation Function = %s\tLayer = %d] : %f" %(act,layer_number,s)



print(dw_mean_map["sigmoid"])
print(dw_mean_map["relu"])

t = np.arange(1, 21, 1)

plot(t, dw_mean_map["sigmoid"], 'r--', t,dw_mean_map["relu"] ,'bs')
plt.ylabel('some numbers')
plt.show()

'''
همانطور که میبینید برای سیگموید واضحا دارد 0 میشود
دلیلش هم این است که مشتق سیگموید ماکسیمم 0.25 است و برای ورودی های بزرگ تر به صورت نمایی کوچک میشود
و برای هر لایه که پاییین می آییم گرادیان وزن ها یک بار ضرب در مشتق تابع سیگموید در نقطه ی ورودی نود ای که به آن وصل است می شود یکبار ضرب در وزن
پس انگار برای هر لایه که پایین می آییم یکبار تابع گرادیان ضرب در یک عدد کوچک خواهد شد
در نتیجه برای شبکه هایی که تعداد لایه های بیشتری دارند گرادیان لایه ی اول به تعداد بیشتری ضرب در یک عدد کوچک خواهد شد
در نتیجه تقریبا نموداری که میبینیم شبیه تابع نمایی است
اما تابع رلو مشتقش یا یک است یا 0 در نتیجه گرادیان نسبت به وزن های لایه ی اول ترتیب خاصی ندارد 
'''