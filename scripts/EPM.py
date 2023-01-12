from sklearn.metrics import auc

import numpy as np

class EffortAwareEvaluation:

    def __init__(self, prob_list, label_list, line_list):
        """
        prob_list: 各コミットの予測確率(予測結果)
        label_list: 0 or 1 でバグを含むコミットかバグを含まないコミットか(教師データ)
        line_list: 各コミットで変更された行数
        """
        self.prob_list = np.array(prob_list)  # M
        self.label_list = np.array(label_list)
        self.line_list = np.array(line_list)  # L (行数のメトリクス)
        self.all_commits = len(label_list)
        self.defective_commits = 0
        for num in label_list:
            if num != 0:
                self.defective_commits += 1

        sorted_prob_index = np.argsort(-self.prob_list)
        self.line_sorted_prob = self.line_list[sorted_prob_index]
        self.label_sorted_prob = self.label_list[sorted_prob_index]

    # IFA
    ## 最初のバグを発見するまでに発生した誤アラームの数
    ## 予測結果でラベルが0だった物
    def IFA(self):  # k
        result = 0
        for label in self.label_sorted_prob:
            if label == 0:
                result += 1
            else:  # m (最初にたどり着いた段階)
                break

        return result

    # PII@L
    ## 全インスタンスのLOCを検査した場合の検査したインスタンスの割合
    ## m / M
    def PII(self, L, prob=None):
        """
        L: number of lines or the percentage (%) of lines in the total lines
        Prob: This is a flag whether L is a percentage or number of lines
        """
        if not prob is None:
            L = int(0.01*L*sum(self.line_list))

        m = 0
        sum_line = 0
        for line in self.line_sorted_prob:
            sum_line += line
            if sum_line > L:
                break
            m += 1

        return m/self.all_commits

    # CostEffort@L
    ## 全インスタンスのL LOCを検査したときに、実際の全不良インスタンスのうち検査された不良インスタンスの割合
    ## n / N
    def CostEffort(self, L, prob=None):
        """
        L: number of lines or the percentage (%) of lines in the total lines
        Prob: This is a flag whether L is a percentage or number of lines
        """
        if not prob is None:
            L = int(0.01*L*sum(self.line_list))

        n = 0
        sum_line = 0
        for line, label in zip(self.line_sorted_prob, self.label_sorted_prob):
            sum_line += line

            if sum_line > L:
                break

            if label != 0:
                n += 1

        return n/self.defective_commits

    def _compute_auc(self, target_line_list, target_label_list, L):
        x = [0]
        y = [0]
        total_line = sum(self.line_list)
        tmp_line = 0
        tmp_commits = 0
        L = 0.01 * L
        for line, label in zip(target_line_list, target_label_list):
            tmp_line += line
            if label != 0:
                tmp_commits += 1
            if (tmp_line/total_line) > L:
                break
            x.append(tmp_line/total_line)
            y.append(tmp_commits/self.defective_commits)

        if not L < 1.0:
            assert x[-1] == 1.0, "x is not correct"
            assert y[-1] == 1.0, "y is not correct"
        else:
            if x[-1] < L:
                x.append(L)
                y.append(y[-1])

            assert x[-1] == L, "The final one is not correct"

        return auc(x, y)

    def norm_popt(self, L=100):

        defective_line = []
        clean_line = []
        for line, label in zip(self.line_sorted_prob, self.label_sorted_prob):
            if label == 0:
                clean_line.append(line)
            else:
                defective_line.append(line)

        max_auc = self._compute_auc(sorted(
            defective_line) + clean_line, ([1] * len(defective_line)) + ([0] * len(clean_line)), L)
        min_auc = self._compute_auc(clean_line + sorted(defective_line, reverse=True), ([
                                    0] * len(clean_line)) + ([1] * len(defective_line)), L)
        predict_auc = self._compute_auc(
            self.line_sorted_prob, self.label_sorted_prob, L)

        #print('mac auc')
        #print(max_auc)
        #print('min auc')
        #print(min_auc)
        #print('predict auc')
        #print(predict_auc)

        const = 0.01*L
        popt = const - (max_auc - predict_auc)
        min_popt = const - (max_auc - min_auc)

        if min_popt == 1:
            return None

        assert (popt - min_popt) / \
            (const - min_popt) <= 1.0, "Invalid norm Popt value"

        return (popt - min_popt)/(const - min_popt)

        #popt = 0.01*L*1 - (max_auc - predict_auc)
        #min_popt = 0.01*L*1 - (max_auc - min_auc)
        #return (popt - min_popt)/(0.01*L*1 - min_popt)

    def _test(self):
        print("Prob")
        print(self.prob_list)
        print("Label")
        print(self.label_list)
        print("Line")
        print(self.line_list)
        print("num commits")
        print(self.all_commits)
        print("defective commits")
        print(self.defective_commits)
        print("Sorted Prob")
        print(np.argsort(-self.prob_list))
        print("Sorted line")
        print(self.line_sorted_prob)
        print("Sorted label")
        print(self.label_sorted_prob)

if __name__=="__main__":
    prob_list = [0.3, 0.1, 0.7, 0.6, 0.2, 0.4, 0.5]
    label_list = [0, 0, 0, 1, 0, 1, 1]
    line_list = [1, 2, 3, 4, 5, 6, 7]

    ins = EffortAwareEvaluation(prob_list, label_list, line_list)

    ins._test()

    print("IFA")
    print(ins.IFA())
    assert 1==ins.IFA(), "IFA is not correct: {0}".format(ins.IFA())

    print("PII")
    print(ins.PII(20, prob=True)) # 20% lines
    print(ins.PII(20)) # 20 lines
    assert 0.143==round(ins.PII(20, prob=True), 3), "Prob PII is not correct: {0}".format(round(ins.PII(20, prob=True), 3))
    assert 0.571==round(ins.PII(20), 3), "Num PII is not correct: {0}".format(round(ins.PII(20), 3))


    print("CostEffort")
    print(ins.CostEffort(20, prob=True)) # 20% lines
    print(ins.CostEffort(20)) # 20 lines
    assert 0.0==ins.CostEffort(20, prob=True), "Prob CostEffort is not correct"
    assert 1.0==ins.CostEffort(20), "Num CostEffort is not correct"

    print("Norm popt")
    print(ins.norm_popt())
    print("==")

    print(ins.norm_popt(L=50))
    print(ins.norm_popt(L=40))

    prob_list = [0.9, 0.9, 0.9, 0.7, 0.7, 1.0, 1.0, 0.8, 0.0, 0.0]
    label_list = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    line_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    ins = EffortAwareEvaluation(prob_list, label_list, line_list)
    print("Controlled test")
    print("Norm popt")
    print(ins.norm_popt())
    assert 0.52==ins.norm_popt(), "norm popt is incorrect"

    print("Norm popt (L=20%)")
    print(ins.norm_popt(L=20))
    assert 0.0==ins.norm_popt(L=20), "norm popt 20% is incorrect"
    print("Norm popt (L=50%)")
    print(ins.norm_popt(L=50))
    assert 0.36==round(ins.norm_popt(L=50), 2), "norm popt 50% is incorrect"
    print("Norm popt (L=70%)")
    print(ins.norm_popt(L=70))
    assert 0.439==round(ins.norm_popt(L=70), 3), "norm popt 70% is incorrect"

    assert 2==ins.IFA(), "IFA is not correct: {0}".format(ins.IFA())

    assert 0.5==round(ins.PII(50, prob=True), 3), "Prob PII is not correct: {0}".format(round(ins.PII(50, prob=True), 3))
    assert 0.5==round(ins.PII(5), 3), "Num PII is not correct: {0}".format(round(ins.PII(5), 3))

    assert 0.0==ins.CostEffort(20, prob=True), "Prob CostEffort is not correct"
    assert 0.6==ins.CostEffort(5), "Num CostEffort is not correct"
    assert 0.6==ins.CostEffort(6), "Num CostEffort is not correct"
    assert 0.8==ins.CostEffort(7), "Num CostEffort is not correct"