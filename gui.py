import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext

import matplotlib.pyplot as plt
import numpy as np
from phe import paillier

from data_divide import read_data, standard4
from experiment_method import cross_train
from modelTrain import exp_model, old_model

T = 10
n_l = 1024


def predict_single_sample(data_str,dataSetList):
    if len(dataSetList) != 1:
        return "选择的数据集不为1"
    public_key, private_key = paillier.generate_paillier_keypair(n_length=n_l)
    adata = data_str.split(",")
    aadtaa = list(map(float, adata))
    x = np.asmatrix(aadtaa).T
    jdata, ndata, jtabel, ntabel = read_data(
        dataSetList[0], stand_method
    )
    # jnum = len(jdata)
    # nnum = len(ndata)
    data_xi = []
    data_yi = []
    j = 0
    for _ in range(1):
        data_xi.append([])
        data_yi.append([])
    for i in range(len(jdata)):
        data_stand = jdata[i]
        data_xi[j].append(data_stand.T)
        data_yi[j].append(jtabel[i])
    for i in range(len(ndata)):
        data_stand = ndata[i]
        data_xi[j].append(data_stand.T)
        data_yi[j].append(ntabel[i])
    train_x = []
    train_y = []
    train_x += data_xi[0]
    train_y += data_yi[0]
    train_num = len(train_y)
    omega, b, t, DPtime, DACtime = exp_model(
        train_x, train_y, train_num, public_key, private_key
    )
    """
    预测单独样本的函数

    参数:
    omega -- 模型参数向量
    b -- 模型偏置
    x -- 输入样本向量

    返回:
    预测结果
    """
    # 将输入样本转换为numpy数组
    x = np.array(x)

    # 计算预测值
    prediction = (omega.T * x + b)[0, 0]

    # 返回预测结果
    if prediction > 0:
        return "阳性"
    else:
        return "阴性"


def gd_svm(
    dataSetList,
    stand_method,
    evaluation_method,
    train_method,
    n_l=1024,
):
    x = np.arange(3)
    resultlist = []
    for data_set in dataSetList:
        public_key, private_key = paillier.generate_paillier_keypair(n_length=n_l)
        # generate a public key and private key pair
        # for data_set in [HDD, BCW]:
        jdata, ndata, jtabel, ntabel = read_data(data_set, stand_method)

        (
            TPsum,
            FPsum,
            FNsum,
            TNsum,
            iter_num,
            tsum,
            run_time,
            DPtime_sum,
            DACtime_sum,
        ) = evaluation_method(
            jdata, ndata, jtabel, ntabel, T, train_method, public_key, private_key
        )
        file_name = data_set.split("/")[-1]
        with open("results/" + file_name, "w") as f:
            log_message(
                "\n平均训练耗时：%fs， DP耗时：%fs， DAC耗时：%fs"
                % (run_time / 2, DPtime_sum / 2, DACtime_sum / 2),
                file=f,
            )

            total = TPsum + FPsum + FNsum + TNsum
            log_message(
                "%d次测试迭代总数：%d，平均迭代次数：%d" % (2, iter_num, iter_num / 2),
                file=f,
            )
            try:
                log_message(
                    "%d次测试样本总数：%d，分类正确总数：%d，正确率：%.3f"
                    % (2, total, tsum, tsum / total),
                    file=f,
                )
            except ZeroDivisionError:
                log_message("参数失控，超出范围", file=f)
            log_message(
                "%d次总和 TP=%d,FP=%d,FN=%d,TN=%d" % (2, TPsum, FPsum, FNsum, TNsum),
                file=f,
            )
            try:
                log_message(
                    "Precision=%f,Recall=%f"
                    % (TPsum / (TPsum + FPsum), TPsum / (TPsum + FNsum)),
                    file=f,
                )
                resultlist.append(
                    [tsum / total, TPsum / (TPsum + FPsum), TPsum / (TPsum + FNsum)]
                )  # 将每个数据集训练的结果插入结果列表中
            except ZeroDivisionError:
                log_message("division by zero", file=f)

    num_groups = len(resultlist)
    width = 0.8 / num_groups
    for i in range(len(resultlist)):
        y = np.array(resultlist[i])
        plt.bar(x + i * width, y, width=width, label=f"Set {i+1}")
    plt.xticks(x + width * (num_groups - 1) / 2, ["Accuracy", "Precision", "Recall"])
    plt.legend()
    plt.show()


dataSetList = []
train_method = None
T = 10
stand_method = standard4
evaluation_method = cross_train


dataSetList = []
train_method = None


def log_message(message, file=None):
    log_textbox.insert(tk.END, message + "\n")
    log_textbox.see(tk.END)
    file.write(message + "\n") if file else print(message)


def clear_log():
    log_textbox.delete(1.0, tk.END)


def select_files():
    global dataSetList
    file_paths = filedialog.askopenfilenames(
        filetypes=[
            ("All Files", "*.*"),
            ("CSV Files", "*.csv"),
            ("Text Files", "*.txt"),
        ]
    )
    if file_paths:
        dataSetList = list(file_paths)  # 保存到 dataSetList
        log_message("Selected files:")
        for file_path in dataSetList:
            print(file_path)


def set_train_method(method):
    global train_method
    train_method = method
    log_message(f"Selected train method: {train_method}")


def train():
    if not dataSetList:
        log_message("No files selected.")
        status_label.config(text="No files selected.")
        return
    if not train_method:
        log_message("No train method selected.")
        status_label.config(text="No train method selected.")
        return

    def run_training():
        result = gd_svm(dataSetList, stand_method, evaluation_method, train_method)
        log_message(f"Training result: {result}")
        status_label.config(text=f"Training result: {result}")

    if status_label.cget("text") == "Training in progress...":
        log_message("Training in progress...")
        return
    status_label.config(text="Training in progress...")
    training_thread = threading.Thread(target=run_training)
    training_thread.start()


# 创建主窗口
root = tk.Tk()
root.title("svm")
root.geometry("600x400")

# 添加文件选择按钮
select_button = tk.Button(root, text="Select Files", command=select_files)
select_button.pack(pady=20)

# 添加下拉菜单选择 train_method
train_methods = {"exp_model": exp_model, "Shen_model": old_model}
train_method_var = tk.StringVar(root)
train_method_var.set("exp_model")  # 设置默认值


def on_train_method_change(selected_method):
    set_train_method(train_methods[selected_method])


train_method_menu = tk.OptionMenu(
    root, train_method_var, *train_methods.keys(), command=on_train_method_change
)
train_method_menu.pack(pady=20)

# 添加训练按钮
train_button = tk.Button(root, text="Train", command=train)
train_button.pack(pady=20)

clear_log_button = tk.Button(root, text="Clear Log", command=clear_log)
clear_log_button.pack(pady=20)
# 添加状态标签
status_label = tk.Label(root, text="")
status_label.pack(pady=20)
# 添加日志文本框
log_textbox = scrolledtext.ScrolledText(root, width=70, height=10)
log_textbox.pack(pady=10)
# 添加输入框和读取按钮
input_label = tk.Label(root, text="输入乳腺癌预诊参数:")
input_label.pack(pady=5)

entry_labels = [
    "半径", "纹理", "周长", "面积", "平滑度",
    "紧凑性", "凹度", "凹点", "对称性"
]
input_entries = []

for label_text in entry_labels:
    frame = tk.Frame(root)
    frame.pack(pady=5, padx=10)  # 只适用frame的padding来提供更好的外观间距
    label = tk.Label(frame, text=label_text, width=10, anchor="w")  # 设置固定宽度和左对齐
    label.pack(side=tk.LEFT)
    entry = tk.Entry(frame, width=10)  # 固定输入框宽度
    entry.pack(side=tk.RIGHT, padx=5)  # 增加左右间距以防止拥挤
    input_entries.append(entry)

def read_input():
    input_numbers = [entry.get() for entry in input_entries]
    numbers = [float(num) for num in input_numbers if num.strip()]
    log_message(f"读取到数据: {numbers}")
    input_text = ",".join(input_numbers)
    r = predict_single_sample(input_text, dataSetList)
    log_message(f"该疾病预诊结果: {r}")
    # 在这里处理读取到的数字

read_button = tk.Button(root, text="读取数据并预诊", command=read_input)
read_button.pack(pady=20)
# 运行主循环
root.mainloop()
