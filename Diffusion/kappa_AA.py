import numpy as np

def aa_oa(matrix):
    accuracy = []
    b = np.sum(matrix, axis=0)
    c = 0
    on_display = []
    for i in range(0, matrix.shape[0]):
        a = matrix[i][i]/b[i]
        c += matrix[i][i]
        accuracy.append(a)
        on_display.append([b[i], matrix[i][i], a])
        print("Category:{}. Overall:{}. Correct:{}. Accuracy:{:.6f}".format(i, b[i], matrix[i][i], a))
    aa = np.mean(accuracy)
    oa = c / np.sum(b, axis=0)
    k = kappa(matrix)
    print("OA:{:.6f} AA:{:.6f} Kappa:{:.6f}".format(oa*100.0, aa*100.0, k*100.0))
    return [aa, oa, k, on_display]

def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)